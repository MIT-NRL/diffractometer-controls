import unittest
from types import SimpleNamespace
from unittest import mock


try:
    from diffractometer_controls.tomography_gui import MainScreen

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only used without GUI dependencies
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"GUI dependencies unavailable: {_IMPORT_ERROR}")
class TomographyStartupLevelTests(unittest.TestCase):
    def test_startup_autoscale_waits_for_several_valid_passes(self):
        timer = mock.Mock()
        harness = SimpleNamespace(
            _startup_autoscale_attempts=0,
            _startup_autoscale_max_attempts=30,
            _startup_autoscale_valid_attempts=0,
            _startup_autoscale_settle_attempts=5,
            _startup_autoscale_timer=timer,
            _last_image=None,
        )

        def apply_levels():
            harness._last_image = object()

        harness._apply_robust_normalization = apply_levels

        for _ in range(4):
            MainScreen._startup_autoscale_tick(harness)
        timer.stop.assert_not_called()

        MainScreen._startup_autoscale_tick(harness)
        timer.stop.assert_called_once_with()

    def test_startup_autoscale_still_times_out_without_an_image(self):
        timer = mock.Mock()
        harness = SimpleNamespace(
            _startup_autoscale_attempts=29,
            _startup_autoscale_max_attempts=30,
            _startup_autoscale_valid_attempts=0,
            _startup_autoscale_settle_attempts=5,
            _startup_autoscale_timer=timer,
            _last_image=None,
            _apply_robust_normalization=mock.Mock(),
        )

        MainScreen._startup_autoscale_tick(harness)

        timer.stop.assert_called_once_with()

    def test_deferred_reapply_respects_auto_levels_toggle(self):
        checkbox = mock.Mock()
        apply_levels = mock.Mock()
        harness = SimpleNamespace(
            _auto_levels_reapply_pending=True,
            auto_levels_checkbox=checkbox,
            _auto_levels_from_current_image=apply_levels,
        )

        checkbox.isChecked.return_value = True
        MainScreen._reapply_auto_levels_after_image_update(harness)
        apply_levels.assert_called_once_with()
        self.assertFalse(harness._auto_levels_reapply_pending)

        apply_levels.reset_mock()
        harness._auto_levels_reapply_pending = True
        checkbox.isChecked.return_value = False
        MainScreen._reapply_auto_levels_after_image_update(harness)
        apply_levels.assert_not_called()
        self.assertFalse(harness._auto_levels_reapply_pending)


if __name__ == "__main__":
    unittest.main()
