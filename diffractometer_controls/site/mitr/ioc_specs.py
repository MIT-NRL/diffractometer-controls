"""IOC identities shared by the control panel and local deployment tools."""

from typing import NamedTuple


class IOCSpec(NamedTuple):
    key: str
    title: str
    heartbeat: str
    uptime: str
    stats_prefix: str
    transport: str
    service: str


def _spec(key, title, suffix, service):
    prefix = f"4dh4:IOC:{suffix}"
    return IOCSpec(key, title, f"{prefix}:HEARTBEAT", f"{prefix}:UPTIME", prefix,
                   f"systemd {service} • private CA/PVA endpoints", service)


IOC_SPECS = (
    _spec("main", "Motors and instrument IOC", "Motors", "4dh4-motors.service"),
    _spec("psd", "He3 PSD detector IOC", "PSD", "4dh4-psd.service"),
    _spec("imaging", "ZWO / QHY imaging IOC", "Imaging", "4dh4-imaging.service"),
    _spec("daq", "Counter and analog / digital IOC", "DAQ", "4dh4-daq.service"),
    _spec("camera", "Reolink camera IOC", "Reolink", "reolinkioc.service"),
)


def ioc_spec(key):
    for spec in IOC_SPECS:
        if spec.key == key:
            return spec
    raise ValueError(f"Unknown IOC: {key}")
