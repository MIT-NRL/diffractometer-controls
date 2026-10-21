"""Stable history controls, live-follow transitions and theme contrast."""

import unittest
from unittest.mock import patch

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from control_ui.widgets.profile_colors import HISTORY_ALPHA, contrast_ratio
from diffractometer_controls.screens.diffraction import diffraction_live_plot as live
from diffractometer_controls.screens.diffraction.diffraction_live_plot_pyqtgraph import DiffractionPlotWidgetPyQtGraph
from diffraction_analysis.history import build_history_snapshot
from tests.test_diffraction_history_window import make_psd


class ControlsPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def flush(self):
        for _ in range(3):
            self.app.processEvents()

    def test_long_run_names_and_status_do_not_change_control_geometry(self):
        plot = DiffractionPlotWidgetPyQtGraph()
        widget = live.DiffractionHistoryViewer(plot)
        widget.resize(1250, 850)
        widget.show()
        try:
            controls = (widget._tiled_checkbox, widget._previous_button, widget._run_combo,
                        widget._next_button, widget._live_checkbox, widget._analysis_button, widget._status_label)
            state = {'entries': [{'uid': 'run', 'label': 'Short'}], 'selected_uid': 'run',
                     'tiled_enabled': True, 'history_enabled': True, 'status_text': 'Ready'}
            widget.set_history_state(state)
            self.flush()
            original = [control.geometry() for control in controls]
            size = widget.size()
            name = 'A very long run file name ' * 20
            widget.set_history_state({**state, 'entries': [{'uid': 'run', 'label': name}], 'status_text': name})
            plot.reset({'run_title': name})
            self.flush()
            self.assertEqual(widget.size(), size)
            self.assertEqual([control.geometry() for control in controls], original)
            self.assertTrue(all(control.geometry().right() < widget.width() for control in controls))
            self.assertIn(name, widget._run_combo.toolTip())
            self.assertEqual(widget._status_label.toolTip(), name)
            self.assertEqual(plot._title_label.toolTip(), name)
            self.assertLess(len(widget._status_label.text()), len(name))
        finally:
            widget.close()

    def test_live_follow_holds_then_resumes_without_tiled_and_ignores_stale_history(self):
        plot = DiffractionPlotWidgetPyQtGraph()
        widget = live.DiffractionHistoryViewer(plot)
        controller = live.DiffractionLivePlot(widget)
        try:
            controller.show_history_snapshot(build_history_snapshot(make_psd('old')))
            self.flush()
            self.assertTrue(widget._live_checkbox.isEnabled())
            self.assertFalse(widget._live_checkbox.isChecked())
            before = dict(plot._summary_y)
            start = {'uid': 'active', 'experiment_type': 'diffraction', 'data_type': '1d', 'detectors': ['psd']}
            descriptor = {'uid': 'descriptor', 'run_start': 'active', 'name': 'primary',
                          'data_keys': {'psd_counts': {'dtype': 'array', 'shape': [3]},
                                        'psd_position_x': {'dtype': 'array', 'shape': [3]}}}
            controller.on_document('start', start)
            controller.on_document('descriptor', descriptor)
            controller.on_document('event', {'descriptor': 'descriptor', 'seq_num': 1,
                                            'data': {'psd_counts': [8, 9, 10], 'psd_position_x': [-1, 0, 1]}})
            self.flush()
            self.assertEqual(plot._summary_y, before)
            old_token = controller._history_load_token
            with patch.object(live, '_safe_caget', side_effect=lambda pv, **kwargs: 'RUNNING' if pv == live.RUN_STATE_STATUS_PV else 'active'):
                widget._live_checkbox.click()
            self.flush()
            self.assertTrue(controller._live_follow_enabled)
            self.assertEqual(controller._run_uid, 'active')
            controller._on_history_load_completed(old_token, {'uid': 'old', 'request_kind': 'history',
                                                             'snapshot': build_history_snapshot(make_psd('old'))})
            self.assertTrue(controller._live_follow_enabled)
            controller.on_document('event', {'descriptor': 'descriptor', 'seq_num': 2,
                                            'data': {'psd_counts': [10, 11, 12], 'psd_position_x': [-1, 0, 1]}})
            self.flush()
            self.assertEqual(plot._summary_y['psd'], [33])
            self.assertTrue(widget._live_checkbox.isChecked())
        finally:
            controller.shutdown()
            widget.close()

    def test_finished_uid_is_not_active_and_stops_are_tracked_while_viewing_history(self):
        plot = DiffractionPlotWidgetPyQtGraph()
        controller = live.DiffractionLivePlot(plot)
        try:
            controller._current_live_run_uid = 'finished'
            with patch.object(live, '_safe_caget', side_effect=lambda pv, **kwargs: 'IDLE' if pv == live.RUN_STATE_STATUS_PV else 'finished'):
                self.assertEqual(controller._read_active_run_uid(), '')
            with patch.object(live, '_safe_caget', return_value=None):
                self.assertEqual(controller._read_active_run_uid(), '')
                controller.on_document('start', {'uid': 'active', 'experiment_type': 'diffraction'})
                self.assertEqual(controller._read_active_run_uid(), 'active')
                controller._on_live_follow_toggled(False)
                controller.on_document('stop', {'run_start': 'active', 'exit_status': 'success'})
                self.assertEqual(controller._read_active_run_uid(allow_cached=True), '')
                self.assertFalse(controller._live_run_active)
        finally:
            controller.shutdown()
            plot.close()

    def test_run_state_is_read_as_an_epics_enum_label(self):
        plot = DiffractionPlotWidgetPyQtGraph()
        controller = live.DiffractionLivePlot(plot)
        try:
            def read(pv, **kwargs):
                if pv == live.RUN_STATE_STATUS_PV:
                    return 'RUNNING' if kwargs.get('as_string') else 1
                return 'active'
            with patch.object(live, 'caget', side_effect=read) as get:
                self.assertEqual(controller._read_active_run_uid(), 'active')
                self.assertTrue(get.call_args_list[0].kwargs['as_string'])
        finally:
            controller.shutdown()
            plot.close()

    def test_disabling_tiled_restores_live_follow(self):
        plot = DiffractionPlotWidgetPyQtGraph()
        controller = live.DiffractionLivePlot(plot)
        try:
            controller._tiled_enabled = True
            controller._live_follow_enabled = False
            controller._display_mode = 'history'
            with patch.object(live, '_safe_caget', return_value=None):
                controller._on_tiled_enabled_toggled(False)
            self.assertTrue(controller._live_follow_enabled)
            self.assertEqual(controller._display_mode, 'live')
        finally:
            controller.shutdown()
            plot.close()

    def test_new_live_start_discards_a_pending_latest_history_backfill(self):
        plot = DiffractionPlotWidgetPyQtGraph()
        controller = live.DiffractionLivePlot(plot)
        try:
            token = controller._history_load_token
            controller.on_document('start', {'uid': 'new', 'experiment_type': 'diffraction'})
            controller._on_history_load_completed(token, {'uid': 'old', 'request_kind': 'latest',
                                                         'snapshot': build_history_snapshot(make_psd('old'))})
            self.assertEqual(controller._run_uid, 'new')
            self.assertTrue(controller._live_follow_enabled)
        finally:
            controller.shutdown()
            plot.close()

    def test_both_backends_recolor_existing_and_preview_spectra_for_dark_theme(self):
        original = QtGui.QPalette(self.app.palette())
        try:
            for cls in (DiffractionPlotWidgetPyQtGraph, live.DiffractionPlotWidget):
                with self.subTest(backend=cls.__name__):
                    self.app.setPalette(original)
                    plot = cls()
                    try:
                        for detector in ('psd0', 'psd1'):
                            for index in range(12):
                                plot.set_profile(detector, [-1, 0, 1], [index, index + 1, index])
                            plot.update_live_profile(detector, [-1, 0, 1], [20, 21, 20])
                        data_ids = [id(item) for items in plot._profile_history.values() for item in items]
                        dark = QtGui.QPalette(original)
                        dark.setColor(QtGui.QPalette.Base, QtGui.QColor('#202124'))
                        dark.setColor(QtGui.QPalette.Window, QtGui.QColor('#292a2d'))
                        self.app.setPalette(dark)
                        plot._apply_theme_from_palette()
                        self.assertEqual([id(item) for items in plot._profile_history.values() for item in items], data_ids)
                        background = tuple(c / 255 for c in (32, 33, 36))
                        for color_map in plot._profile_colormaps.values():
                            for position in np.linspace(0.15, 0.9, 20):
                                self.assertGreaterEqual(contrast_ratio(color_map(position)[:3], background, HISTORY_ALPHA), 3)
                        a = plot._profile_colormaps['psd0'](0.9)
                        b = plot._profile_colormaps['psd1'](0.9)
                        self.assertGreater(np.linalg.norm(np.array(a) - b), 0.2)
                        older_green = plot._profile_colormaps['psd0'](0.15)
                        older_red = plot._profile_colormaps['psd1'](0.15)
                        self.assertGreater(a[2], a[1])  # Latest is blue.
                        self.assertGreater(older_green[1], max(older_green[0], older_green[2]))
                        self.assertGreater(b[0], b[1])  # Latest is orange.
                        self.assertGreater(b[1], b[2])
                        self.assertGreater(b[1], older_red[1])
                        self.assertGreater(older_red[0], max(older_red[1], older_red[2]))
                        old = plot._profile_history['psd0'][0]
                        if isinstance(plot, DiffractionPlotWidgetPyQtGraph):
                            color = old.opts['pen'].color()
                            actual = (color.redF(), color.greenF(), color.blueF())
                            alpha = color.alphaF()
                        else:
                            actual = old.get_color()[:3]
                            alpha = old.get_alpha()
                        self.assertGreaterEqual(contrast_ratio(actual, background, alpha), 2.95)
                    finally:
                        plot.close()
        finally:
            self.app.setPalette(original)
            self.flush()
