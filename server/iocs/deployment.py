"""Render a host-local IOC split without changing or starting running services.

Rendered files live outside Git; existing device addresses and support paths are
read from the installed IOC. Installation and cutover are separate operations.
"""

import argparse
import json
from pathlib import Path
import re
import shlex
import shutil

from diffractometer_controls.site.mitr.ioc_specs import IOC_SPECS


BOOT_NAMES = {"main": "motors", "psd": "psd", "imaging": "imaging", "daq": "daq"}
NETWORK_NAMES = {"main": "MITR_MAIN", "psd": "MITR_PSD", "imaging": "MITR_IMAGING", "daq": "MITR_DAQ"}


def read_env(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        tokens = shlex.split(value, comments=True)
        result[key.strip()] = " ".join(tokens)
    return result


def imaging_prefixes(adcore):
    text = (Path(adcore) / "iocBoot/commonPlugins.cmd").read_text()
    names = set(re.findall(r"\bR=([A-Za-z][A-Za-z0-9]*):", text))
    names.update(("cam1", "image1", "camURL", "urlimage1", "Pva2"))
    return tuple(f"4dh4:{name}:" for name in sorted(names)) + ("4dh4:ADAutoSave",)


def owner(pv, camera_prefixes):
    if pv.startswith("4dh4:he3PSD:"):
        return "psd"
    if pv.startswith(("4dh4:USB2408:", "4dh4:USBCTR:")):
        return "daq"
    if pv.startswith(camera_prefixes):
        return "imaging"
    return "main"


def split_save(text, camera_prefixes):
    """Keep complete autosave entries, including multiline waveform arrays."""
    result = {key: [] for key in BOOT_NAMES}
    blocks = re.split(r"(?=^4dh4:[^\s]+(?:[ \t]|$))", text, flags=re.MULTILINE)
    for block in blocks:
        match = re.match(r"(4dh4:[^\s]+)(?:[ \t]|$)", block)
        if not match:
            continue
        block = block.split("<END>", 1)[0]
        # A missing-PV comment is not an entry to restore.
        block = "\n".join(line for line in block.splitlines() if not line.startswith("#")) + "\n"
        result[owner(match[1], camera_prefixes)].append(block)
    return {key: "# Partitioned from the original IOC autosave.\n" + "".join(blocks) + "<END>\n"
            for key, blocks in result.items()}


def split_inventory(text, camera_prefixes):
    result = {key: [] for key in BOOT_NAMES}
    for name in text.splitlines():
        name = name.strip()
        if name:
            result[owner(name, camera_prefixes)].append(name)
    return result


def network_config(original):
    values = read_env(original)
    used = {int(value) for key, value in values.items() if key.endswith("_PORT")}
    next_port = max(used | {5050}) + 1
    for key in ("psd", "imaging", "daq"):
        for suffix in ("CA_SERVER_PORT", "PVA_SERVER_PORT", "PVA_BROADCAST_PORT"):
            name = f"{NETWORK_NAMES[key]}_{suffix}"
            if name not in values:
                while next_port in used:
                    next_port += 1
                values[name] = str(next_port)
                used.add(next_port)
                next_port += 1
    host = values["MITR_EPICS_CA_ADDR_LIST"].split()[0].rsplit(":", 1)[0]
    for protocol, suffix in (("CA", "CA_SERVER_PORT"), ("PVA", "PVA_BROADCAST_PORT")):
        name = f"MITR_EPICS_{protocol}_ADDR_LIST"
        endpoints = values[name].split()
        for key in BOOT_NAMES:
            endpoint = f"{host}:{values[NETWORK_NAMES[key] + '_' + suffix]}"
            if endpoint not in endpoints:
                endpoints.append(endpoint)
        values[name] = " ".join(endpoints)
    return "# Private IOC endpoints and interface bindings.\n" + "".join(
        f"{key}={shlex.quote(value)}\n" for key, value in values.items())


def stats_aliases(prefix, old_prefix, stats_db):
    names = re.findall(r'(?:record\([^,]+,\s*|alias\(\s*)"\$\(IOC\):([^"$]+)"', stats_db)
    return "".join(f'alias("{prefix}:{name}", "{old_prefix}:{name}")\n' for name in names)


def refresh_autosaves(ioc_root, stage):
    root, stage = Path(ioc_root), Path(stage)
    legacy = root / "iocBoot/ioc4dh4"
    paths = parse_env_paths(legacy / "envPaths")
    camera_prefixes = imaging_prefixes(paths["ADCORE"])
    for name in ("auto_positions", "auto_settings", "built_positions", "built_settings"):
        source = legacy / "autosave" / f"{name}.sav"
        if not source.exists():
            continue
        partitions = split_save(source.read_text(errors="surrogateescape"), camera_prefixes)
        for key, content in partitions.items():
            boot = stage / "boot" / BOOT_NAMES[key]
            (boot / "autosave" / f"{name}.sav").write_text(content, errors="surrogateescape")
            if name.startswith("auto_"):
                fields = re.findall(r"^(4dh4:[^\s]+)", content, flags=re.MULTILINE)
                (boot / f"{name}.req").write_text("\n".join(fields) + "\n")
    inventory = split_inventory((legacy / "dbl-all.txt").read_text(), camera_prefixes)
    (stage / "baseline-pv-ownership.json").write_text(json.dumps(inventory, indent=2) + "\n")


def parse_env_paths(path):
    return dict(re.findall(r'epicsEnvSet\("([^"]+)",\s*"([^"]*)"\)', Path(path).read_text()))


def render(ioc_root, network, stage, checkout=None):
    root, stage = Path(ioc_root).resolve(), Path(stage).resolve()
    checkout = Path(checkout or Path(__file__).resolve().parents[2])
    legacy = root / "iocBoot/ioc4dh4"
    paths = parse_env_paths(legacy / "envPaths")
    stage.mkdir(parents=True, exist_ok=False, mode=0o700)
    # Rendered boot files contain host addresses and may contain driver credentials.
    (stage / ".gitignore").write_text("*\n")
    (stage / "units").mkdir()
    (stage / "network.env").write_text(network_config(network))
    stats_db = (Path(paths["DEVIOCSTATS"]) / "db/iocAdminSoft.db").read_text()
    # Autosave initialization is shared below. Only the instrument IOC loads
    # shared soft records; copying the monolithic common file would duplicate PVs.
    common = (legacy / "common.iocsh").read_text().split("# Access Security", 1)[1]
    common = "# Access Security" + common
    original_startup = (legacy / "st.cmd.Linux").read_text()
    daq_post = original_startup.split("# Configure the USB-CTR08", 1)[1].split("# write all the PV names", 1)[0]
    daq_post = "# Configure the USB-CTR08" + daq_post
    records = {
        "main": '< common-main.iocsh\n< iocsh/motors.iocsh\n< iocsh/ESP300.cmd\n< iocsh/AcsMotion.cmd\ndbLoadTemplate("substitutions/tomoScan.substitutions")\n',
        "psd": '< iocsh/psdPortDriver.cmd\n',
        "imaging": '< iocsh/ADZWO.iocsh\n< iocsh/ADURL.iocsh\n',
        "daq": '< iocsh/measUSB2408.iocsh\n< iocsh/measUSBCTR.iocsh\n',
    }
    for spec in IOC_SPECS:
        if spec.key not in BOOT_NAMES:
            continue
        name, env_prefix = BOOT_NAMES[spec.key], NETWORK_NAMES[spec.key]
        boot = stage / "boot" / name
        boot.mkdir(parents=True)
        (boot / "autosave").mkdir()
        # Use an immutable copy of the installed path/device configuration.
        for file in ("envPaths", "modules.lua", "settings.iocsh", "saveData.req", "scan1Menu.req"):
            shutil.copy2(legacy / file, boot / file)
        shutil.copytree(legacy / "iocsh", boot / "iocsh")
        shutil.copytree(legacy / "substitutions", boot / "substitutions")
        # motors.iocsh already owns allstop/alldone records.
        esp = boot / "iocsh/ESP300.cmd"
        esp.write_text(esp.read_text().replace(
            'iocshLoad("$(MOTOR)/iocsh/allstop.iocsh", "P=$(PREFIX)")',
            '# allstop/alldone loaded once by motors.iocsh'))
        (boot / "common-main.iocsh").write_text(common)
        if spec.key == "imaging":
            # ADCore's shared plugin script otherwise reloads the legacy global
            # autosave status records and redirects this IOC's status prefix.
            plugins = (Path(paths["ADCORE"]) / "iocBoot/commonPlugins.cmd").read_text()
            plugins = "\n".join(line for line in plugins.splitlines()
                                if not line.startswith("save_restoreSet_status_prefix(")
                                and not (line.startswith("dbLoadRecords(") and "save_restoreStatus.db" in line)) + "\n"
            (boot / "commonPlugins.cmd").write_text(plugins)
            for camera in ("ADZWO.iocsh", "ADQHY.iocsh"):
                driver = boot / "iocsh" / camera
                driver.write_text(driver.read_text().replace("$(ADCORE)/iocBoot/commonPlugins.cmd", "commonPlugins.cmd"))
        interface = "$(MITR_EPICS_SERVER_INTERFACE)"
        (boot / "network.iocsh").write_text(f'''# Configure servers before registering drivers/PVA providers.
epicsEnvSet("EPICS_CA_SERVER_PORT", "$({env_prefix}_CA_SERVER_PORT)")
epicsEnvSet("EPICS_CAS_SERVER_PORT", "$({env_prefix}_CA_SERVER_PORT)")
epicsEnvSet("EPICS_CAS_INTF_ADDR_LIST", "{interface}")
epicsEnvSet("EPICS_CAS_AUTO_BEACON_ADDR_LIST", "NO")
epicsEnvSet("EPICS_PVAS_SERVER_PORT", "$({env_prefix}_PVA_SERVER_PORT)")
epicsEnvSet("EPICS_PVAS_BROADCAST_PORT", "$({env_prefix}_PVA_BROADCAST_PORT)")
epicsEnvSet("EPICS_PVAS_INTF_ADDR_LIST", "{interface}")
epicsEnvSet("EPICS_PVAS_AUTO_BEACON_ADDR_LIST", "NO")
''')
        autosave_prefix = "4dh4:" if spec.key == "main" else spec.stats_prefix + ":"
        startup = f'''# Independent {name} IOC. Device PV names are unchanged.
< envPaths
epicsEnvSet("IOC", "ioc4dh4-{name}")
< network.iocsh
errlogInit(20000)
dbLoadDatabase("$(TOP)/dbd/ioc4dh4Linux.dbd")
ioc4dh4Linux_registerRecordDeviceDriver(pdbbase)
< settings.iocsh
epicsEnvSet("TZ", "EST5EDT")
iocshLoad("$(AUTOSAVE)/iocsh/autosave_settings.iocsh", "PREFIX={autosave_prefix},SAVE_PATH=$(TOP)/iocBoot/ioc4dh4-{name}")
luaCmd("modules=require('modules'); for mod,path in pairs(modules) do set_requestfile_path(path .. '/db') end")
iocshLoad("$(AUTOSAVE)/iocsh/save_restore.iocsh", "PREFIX=$(PREFIX),POSITIONS_FILE=auto_positions,SETTINGS_FILE=auto_settings")
iocshLoad("$(AUTOSAVE)/iocsh/autosaveBuild.iocsh", "PREFIX=$(PREFIX),BUILD_PATH=autosave")
{records[spec.key]}
dbLoadRecords("$(DEVIOCSTATS)/db/iocAdminSoft.db", "IOC={spec.stats_prefix}")
'''
        if spec.key == "main":
            (boot / "stats-compat.db").write_text(stats_aliases(spec.stats_prefix, "4dh4", stats_db))
            startup += '< stats-compat.db\n'
        if spec.key != "imaging":
            startup += 'startPVAServer\n'
        startup += 'iocInit\n'
        if spec.key == "daq":
            startup += daq_post
            startup += 'dbpf $(PREFIX)$(USBPREFIX)WaveDigDwell.PROC 1\ndbpf $(PREFIX)$(USBPREFIX)WaveGenUserDwell.PROC 1\n'
        if spec.key == "imaging":
            startup += 'dbpf "4dh4:camURL:Acquire" 1\n'
        startup += 'dbl > dbl-all.txt\ndate\n'
        # Alias is an IOC-shell database command, not a separate script command.
        startup = startup.replace('< stats-compat.db', 'dbLoadRecords("stats-compat.db", "")')
        (boot / "st.cmd").write_text(startup)
        if spec.key == "imaging":
            (boot / "st-qhy.cmd").write_text(startup.replace('< iocsh/ADZWO.iocsh', '< iocsh/ADQHY.iocsh'))
        installed_boot = root / "iocBoot" / f"ioc4dh4-{name}"
        (boot / "run-ioc").write_text(f'''#!/usr/bin/env bash
set -euo pipefail
set -a
if [[ -r "$HOME/.config/diffractometer-controls/control.env" ]]; then
    source "$HOME/.config/diffractometer-controls/control.env"
fi
source "$HOME/.config/epics/network.env"
export EPICS_CA_AUTO_ADDR_LIST=YES EPICS_PVA_AUTO_ADDR_LIST=YES
export EPICS_CA_ADDR_LIST="${{MITR_EPICS_CA_ADDR_LIST}}"
export EPICS_PVA_ADDR_LIST="${{MITR_EPICS_PVA_ADDR_LIST}}"
set +a
cd {shlex.quote(str(installed_boot))}
startup=st.cmd
'''+ ('''case "${MITR_CAMERA_BACKEND:-zwo}" in
    zwo) ;;
    qhy) startup=st-qhy.cmd ;;
    *) echo "MITR_CAMERA_BACKEND must be zwo or qhy" >&2; exit 1 ;;
esac
''' if spec.key == "imaging" else "") + f'exec {shlex.quote(str(root / "bin/linux-x86_64/4dh4"))} --service "$startup"\n')
        (boot / "run-ioc").chmod(0o755)
        (stage / "units" / spec.service).write_text(f'''[Unit]
Description=4DH4 {spec.title}

[Service]
Type=exec
WorkingDirectory={installed_boot}
ExecStart=/bin/bash {installed_boot}/run-ioc
Restart=no
KillSignal=SIGTERM
KillMode=mixed
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal

# Manual start only. No [Install] target and no automatic IOC dependencies.
''')
    (stage / "reolink-stats-compat.db").write_text(stats_aliases("4dh4:IOC:Reolink", "4dh4:ReolinkIOC", stats_db))
    refresh_autosaves(root, stage)
    (stage / "deployment.json").write_text(json.dumps({
        "ioc_root": str(root), "checkout": str(checkout),
        "services": [spec.service for spec in IOC_SPECS],
        "camera_backend": "MITR_CAMERA_BACKEND in control.env; default zwo",
    }, indent=2) + "\n")
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ioc-root", type=Path, required=True)
    parser.add_argument("--network", type=Path, default=Path.home() / ".config/epics/network.env")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--refresh-autosaves", action="store_true")
    args = parser.parse_args()
    if args.refresh_autosaves:
        refresh_autosaves(args.ioc_root, args.stage)
    else:
        print(render(args.ioc_root, args.network, args.stage))


if __name__ == "__main__":
    main()
