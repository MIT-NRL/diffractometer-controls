"""MITR composition adapter for the instrument-independent workspace."""

from qtpy import QtCore, QtWidgets

from control_ui.layouts.experiment_workspace import ExperimentWorkspace
from diffractometer_controls.display import MITRDisplay


class _SharedDisplayOwner:
    """Own the callbacks/timers introduced by one shared display factory."""

    def __init__(self, factory, services):
        events = services.re_client.events
        before = {name: getattr(events, name).callbacks for name in events}
        self.widget = factory()
        self._connections = [
            (getattr(events, name), callback)
            for name in events
            for callback in getattr(events, name).callbacks
            if callback not in before[name]
        ]
        self._timers = []
        self._active = True
        self._shutdown = False

    def deactivate(self):
        if not self._active:
            return
        self.widget.deactivate_display()
        self._timers = [(timer, timer.interval())
                        for timer in self.widget.findChildren(QtCore.QTimer)
                        if timer.isActive()]
        for timer, _ in self._timers:
            timer.stop()
        for emitter, callback in self._connections:
            emitter.disconnect(callback)
        self._active = False

    def activate(self):
        if self._active or self._shutdown:
            return
        for emitter, callback in reversed(self._connections):
            emitter.connect(callback)
        for timer, interval in self._timers:
            timer.start(interval)
        self.widget.activate_display()
        self._active = True

    def shutdown(self):
        if self._shutdown:
            return
        self.deactivate()
        cleanup = getattr(self.widget, "prepare_for_detach", None)
        if callable(cleanup):
            cleanup()
        self._shutdown = True


class _ExperimentOwner:
    def __init__(self, screen):
        self.screen = screen

    def activate(self):
        self.screen.activate_display()

    def deactivate(self):
        self.screen.deactivate_display()

    def shutdown(self):
        self.screen.shutdown_experiment()


class ExperimentScreen(MITRDisplay):
    def build_workspace(self, *, viewer_title="Viewer", additional_tabs=()):
        from diffractometer_controls.reactor_power import ReactorPowerDisplay
        from diffractometer_controls.re_control_panel import REControlPanel
        from diffractometer_controls.re_plans import REPlans
        from diffractometer_controls.re_extras import REPlans as REExtras

        macros = self.macros()
        self.workspace = ExperimentWorkspace(self.services, self)
        owners = [
            _SharedDisplayOwner(lambda: cls(macros=macros, services=self.services), self.services)
            for cls in (ReactorPowerDisplay, REControlPanel, REPlans, REExtras)
        ]
        source, run_engine, plans, extras = [owner.widget for owner in owners]
        self.workspace.install_shared_controls(
            source_status=source,
            run_engine=run_engine, plans=plans, console_history=extras,
            plan_editor=plans.re_plan_editor,
        )
        self.workspace.install_experiment(
            viewer=self.ui.experimentViewer, controls=self.ui.epicsControls,
            acquisition=self.ui.acquisitionStrip,
        )
        self.workspace.viewerTabs.setTabText(0, viewer_title)
        for name, title in additional_tabs:
            self.workspace.add_tab(getattr(self.ui, name), title)
        # The fragment root now holds only the shared shell. Designer files
        # still expose all instrument widget handles to the original methods.
        self.ui.layout().addWidget(self.workspace)
        for owner in owners:
            self.workspace.register_component(owner)
        self.workspace.register_component(_ExperimentOwner(self))
        self.destroyed.connect(self.workspace.shutdown)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.workspace.shutdown)

    def cleanup_before_navigation(self):
        if not getattr(self, "_navigation_active", True):
            return
        self.workspace.deactivate()
        self._navigation_active = False

    def restore_after_navigation(self):
        if getattr(self, "_navigation_active", False):
            return
        self.workspace.activate()
        self._navigation_active = True

    def shutdown_experiment(self):
        """Subclasses release document subscriptions or scientific workers."""
        self.deactivate_display()
