import copy
import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from qtpy import QtCore, QtGui, QtTest, QtWidgets

from diffraction_analysis.batch import export_batch, fit_snapshot, parameter_trend
from diffraction_analysis.fitting import FitConfig
from diffraction_analysis.history import build_history_snapshot
from diffraction_analysis.window import DiffractionAnalysisWindow
from tests.test_diffraction_history import Run, Stream
from tests.test_diffraction_history_window import Catalog


def scan_run(points=5, flat=None):
    x = np.linspace(-5, 5, 151)
    motor = np.linspace(-2, 2, points)
    profiles = [5 + (80 + 3 * m) * np.exp(-0.5 * ((x - (0.3 + 0.2 * m)) / (0.5 + 0.02 * m))**2) for m in motor]
    if flat is not None:
        profiles[flat] = np.full_like(x, 5)
    stream = Stream(xr.Dataset({'intensity': (('event','bin'), profiles), 'angle': (('event','bin'), np.tile(x,(points,1))), 'motor': ('event', motor)}), {
        'intensity': {'dtype':'array','shape':[len(x)],'units':'counts'},
        'angle': {'dtype':'array','shape':[len(x)],'units':'deg'},
        'motor': {'dtype':'number','shape':[],'units':'mm'},
    })
    run = Run(stream)
    run.metadata['start'].update(uid='scan', data_type='1d', motors=['motor'], plan_name='diffraction_scan',
        live_plot_fields={'intensity':{'role':'signal'}, 'angle':{'role':'coordinate'}})
    return run


class BatchCoreTests(unittest.TestCase):
    def test_fit_all_points_returns_parameters_against_recorded_motor(self):
        snapshot = build_history_snapshot(scan_run())
        result = fit_snapshot(snapshot, 'intensity', 'angle', FitConfig(xmin=-2, xmax=2))
        self.assertEqual(len(result['records']), 5)
        self.assertTrue(all(record['fit'] is not None for record in result['records']))
        x, center, errors = parameter_trend(result['records'], coordinate='motor')
        np.testing.assert_allclose(x, [-2,-1,0,1,2])
        np.testing.assert_allclose(center, 0.3 + 0.2 * x, atol=1e-5)
        self.assertEqual(errors.shape, center.shape)
        _, width, _ = parameter_trend(result['records'], 'FWHM', coordinate='motor')
        np.testing.assert_allclose(width, 2.354820045 * (0.5 + 0.02*x), atol=1e-5)
        _, height, _ = parameter_trend(result['records'], 'Peak height', coordinate='motor')
        np.testing.assert_allclose(height, 80+3*x, atol=1e-4)
        self.assertEqual(json.loads(json.dumps(export_batch(result),allow_nan=False))['uid'], 'scan')

    def test_failed_fits_and_missing_coordinates_are_not_zeroed(self):
        snapshot = build_history_snapshot(scan_run(flat=2))
        snapshot['rows'][3]['motor'] = None
        result = fit_snapshot(snapshot,'intensity','angle',FitConfig())
        self.assertIn('constant', result['records'][2]['error'])
        x,y,_ = parameter_trend(result['records'],coordinate='motor')
        self.assertTrue(np.isnan(y[2]))
        self.assertTrue(np.isnan(x[3]))
        self.assertEqual(len(result['records']),5)
        result['records'][0]['fit']['parameters']['p1_center']['stderr'] = None
        _,_,error = parameter_trend(result['records'],coordinate='motor')
        self.assertTrue(np.isnan(error[0]))

    def test_cancellation_keeps_completed_points_and_progress(self):
        snapshot = build_history_snapshot(scan_run())
        cancelled = threading.Event();progress=[]
        def tick(done,total):
            progress.append((done,total))
            if done == 2:cancelled.set()
        result = fit_snapshot(snapshot,'intensity','angle',FitConfig(),progress=tick,cancelled=cancelled.is_set)
        self.assertTrue(result['cancelled'])
        self.assertEqual(len(result['records']),2)
        self.assertEqual(progress,[(1,5),(2,5)])


class RunAndFitTabsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait_for(self,predicate):
        end=time.monotonic()+8
        while time.monotonic()<end:
            self.app.processEvents()
            if predicate():
                self.app.processEvents();return
            QtTest.QTest.qWait(10)
        self.fail('Viewer operation timed out')

    def window(self):
        w=DiffractionAnalysisWindow({'runs':Catalog(scan=scan_run())},catalogs={'Runs':'runs'})
        self.wait_for(lambda:w.data_view.snapshot is not None)
        return w

    def close(self,w):
        w.close();self.app.processEvents();self.app.sendPostedEvents(None,QtCore.QEvent.DeferredDelete)

    def test_data_tab_defaults_to_overlay_and_has_position_navigation(self):
        w=self.window()
        try:
            v=w.data_view
            self.assertEqual(w.workspace_tabs.currentIndex(),0)
            self.assertTrue(v.show_all.isChecked())
            self.assertEqual(len(v.curves),5)
            self.assertFalse(w.details_tabs.isTabVisible(0))
            v.next_button.click()
            self.assertFalse(v.show_all.isChecked())
            self.assertEqual(len(v.curves),1)
            self.assertEqual(v.curves[0]['point'],1)
            v.position_combo.setCurrentIndex(3)
            self.assertEqual(v.curves[0]['point'],3)
            self.assertIn('motor = 1.0',v.summary.text())
            w.workspace_tabs.setCurrentIndex(1)
            self.assertEqual(w.analysis.curve['point'],3)
            self.assertTrue(w.details_tabs.isTabVisible(0))
        finally:self.close(w)

    def test_all_profile_signals_are_overlaid_and_scalar_traces_are_not_repeated(self):
        run=scan_run()
        stream=run['primary']
        stream.dataset['second_intensity']=stream.dataset['intensity']*2
        stream.metadata['data_keys']['second_intensity']=dict(stream.metadata['data_keys']['intensity'])
        run.metadata['start']['live_plot_fields']['intensity']['label']='Detector counts'
        run.metadata['start']['live_plot_fields']['second_intensity']={'role':'signal','label':'Detector counts'}
        w=DiffractionAnalysisWindow({'runs':Catalog(scan=run)},catalogs={'Runs':'runs'})
        try:
            self.wait_for(lambda:w.data_view.snapshot is not None)
            v=w.data_view
            self.assertEqual(len(v.curves),10)
            self.assertIn('[intensity]',v.signal_combo.itemText(1))
            self.assertIn('[second_intensity]',v.signal_combo.itemText(2))
            latest_color = next(item.opts['pen'].color() for item, curve in v._curve_items
                                if curve['field'] == 'second_intensity' and curve['point'] == 4)
            v.next_button.click()
            self.assertEqual(len(v.curves),2)
            self.assertTrue(all(curve['point']==1 for curve in v.curves))
            # Isolating detector 2 must preserve its ramp, rather than
            # renumbering the selected detector and changing it to blue.
            v.signal_combo.setCurrentIndex(2)
            v.point_combo.setCurrentIndex(4)
            self.assertEqual(v._curve_items[0][0].opts['pen'].color(), latest_color)
            w.workspace_tabs.setCurrentIndex(1)
            p = w.analysis
            self.assertEqual(p.data_line.opts['pen'].color(), latest_color)
            batch = fit_snapshot(p.snapshot, 'second_intensity', 'angle', FitConfig(xmin=-2, xmax=2))
            p.show_batch(batch)
            self.assertEqual(p.trends.line.opts['pen'].color(), QtGui.QColor('#ff7f0e'))
            for base in ('#202124', '#ffffff'):
                palette = QtGui.QPalette(w.palette())
                palette.setColor(QtGui.QPalette.Base, QtGui.QColor(base))
                w.setPalette(palette)
                self.app.processEvents()
                self.assertEqual(v._curve_items[0][0].opts['pen'].color(), p.data_line.opts['pen'].color())
                self.assertIs(p.batch, batch)
                self.assertEqual(v.point_combo.currentIndex(), 4)
        finally:self.close(w)

    def test_batch_results_survive_point_navigation_and_plot_motor_trends(self):
        w=self.window()
        try:
            w.workspace_tabs.setCurrentIndex(1)
            p=w.analysis
            p.min_spin.setValue(-2);p.max_spin.setValue(2)
            p.batch_button.click()
            self.wait_for(lambda:p.batch is not None and len(p.batch['records'])==5)
            self.assertEqual(p.trends.coordinate_combo.currentData(),'motor')
            np.testing.assert_allclose(p.trends.line.xData,[-2,-1,0,1,2])
            np.testing.assert_allclose(p.trends.line.yData,[-0.1,0.1,0.3,0.5,0.7],atol=1e-5)
            self.assertIsNone(p.trends.errors.opts['height'])
            p.trends.errors.boundingRect()  # Exercise the paint-time geometry.
            p.trends.metric_combo.setCurrentText('Peak height')
            p.trends.errors.boundingRect()
            self.assertEqual(p.trends.table.rowCount(),5)
            p.point_combo.setCurrentIndex(3)
            self.assertEqual(len(p.batch['records']),5)
            self.assertAlmostEqual(p.fit_result['parameters']['p1_center']['value'],0.5,places=4)
            self.assertEqual(p.min_spin.value(),-2)
            p.trends.table.cellDoubleClicked.emit(1,0)
            self.assertEqual(p.curve['point'],1)
            p.copy_button.click()
            self.assertEqual(len(json.loads(self.app.clipboard().text())['records']),5)
            p.background_combo.setCurrentText('Linear')
            self.assertIsNone(p.batch)
            self.assertEqual(p.trends.table.rowCount(),0)
        finally:self.close(w)

    def test_separate_single_fits_accumulate_and_can_be_revisited(self):
        w=self.window()
        try:
            w.workspace_tabs.setCurrentIndex(1);p=w.analysis
            p.fit_button.click();self.wait_for(lambda:p.fit_result is not None)
            p.point_combo.setCurrentIndex(2)
            self.assertIsNone(p.fit_result)
            self.assertEqual(len(p.batch['records']),1)
            p.fit_button.click();self.wait_for(lambda:p.fit_result is not None)
            self.assertEqual(len(p.batch['records']),2)
            p.point_combo.setCurrentIndex(0)
            self.assertAlmostEqual(p.fit_result['parameters']['p1_center']['value'],-0.1,places=4)
        finally:self.close(w)

    def test_stop_preserves_partial_batch_and_new_settings_discard_running_batch(self):
        import diffraction_analysis.window as window_module
        original=window_module.fit_snapshot
        for change_settings in (False,True):
            started,release=threading.Event(),threading.Event()
            def slow_batch(snapshot,field,x_field,config,*,progress,cancelled):
                def tick(done,total):
                    progress(done,total)
                    if done==1:
                        started.set();release.wait(5)
                return original(snapshot,field,x_field,config,progress=tick,cancelled=cancelled)
            with self.subTest(change_settings=change_settings),patch.object(window_module,'fit_snapshot',side_effect=slow_batch):
                w=self.window()
                try:
                    w.workspace_tabs.setCurrentIndex(1);p=w.analysis
                    p.batch_button.click();self.wait_for(started.is_set)
                    if change_settings:p.background_combo.setCurrentText('Linear')
                    else:p.stop_button.click()
                    release.set()
                    self.wait_for(lambda:all(future.done() for future in w._futures))
                    self.app.processEvents()
                    if change_settings:self.assertIsNone(p.batch)
                    else:
                        self.assertTrue(p.batch['cancelled'])
                        self.assertEqual(len(p.batch['records']),1)
                    self.assertFalse(p.stop_button.isEnabled())
                finally:
                    release.set();self.close(w)

    def test_fit_region_drag_coalesces_preview_without_changing_plot_range(self):
        w=self.window()
        try:
            w.workspace_tabs.setCurrentIndex(1);p=w.analysis
            p.fit_button.click()
            self.wait_for(lambda:p.fit_result is not None)
            result_rows=p.results.rowCount()
            self.app.processEvents()
            original_range=np.array(p.plot.viewRange())
            invalidated=QtTest.QSignalSpy(p.context_changed)
            fitted=QtTest.QSignalSpy(p.fit_requested)
            updates=QtTest.QSignalSpy(p.min_spin.valueChanged)
            # Moving a handle emits in-progress changes, not a completed drag.
            for value in np.linspace(-4,-1,80):p.region.lines[0].setValue(float(value))
            self.app.processEvents();QtTest.QTest.qWait(40);self.app.processEvents()
            self.assertEqual(len(invalidated),0)
            self.assertEqual(len(fitted),0)
            self.assertEqual(len(updates),0)
            self.assertAlmostEqual(p.min_spin.value(),-1)
            self.assertEqual(p.results.rowCount(),result_rows)
            self.assertIsNotNone(p.fit_result)
            np.testing.assert_allclose(p.plot.viewRange(),original_range)
            p.region.sigRegionChangeFinished.emit(p.region)
            self.assertEqual(len(invalidated),1)
            self.assertIsNone(p.fit_result)
            self.assertFalse(p._window_timer.isActive())
        finally:self.close(w)
