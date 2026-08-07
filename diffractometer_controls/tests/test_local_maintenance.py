import tempfile
import unittest
from pathlib import Path

from diffractometer_controls.local_maintenance import (
    LOCAL_MAINTENANCE_ENV,
    local_maintenance_allowed,
)


class LocalMaintenanceTests(unittest.TestCase):
    def test_permission_is_fail_closed_when_unconfigured(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.env"
            self.assertFalse(
                local_maintenance_allowed(
                    environ={},
                    control_env=missing,
                    system_name="Linux",
                )
            )

    def test_linux_control_host_can_opt_in_through_config_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "control.env"
            config.write_text(
                f"# Per-machine configuration\n{LOCAL_MAINTENANCE_ENV}='yes'\n",
                encoding="utf-8",
            )
            self.assertTrue(
                local_maintenance_allowed(
                    environ={},
                    control_env=config,
                    system_name="Linux",
                )
            )

    def test_process_environment_overrides_config_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "control.env"
            config.write_text(f"{LOCAL_MAINTENANCE_ENV}=1\n", encoding="utf-8")
            self.assertFalse(
                local_maintenance_allowed(
                    environ={LOCAL_MAINTENANCE_ENV: "0"},
                    control_env=config,
                    system_name="Linux",
                )
            )

    def test_non_linux_hosts_remain_disabled_even_if_configured(self):
        self.assertFalse(
            local_maintenance_allowed(
                environ={LOCAL_MAINTENANCE_ENV: "1"},
                system_name="Windows",
            )
        )


if __name__ == "__main__":
    unittest.main()
