"""Compose instrument widgets without importing an application or a site."""

from pathlib import Path

from qtpy import QtCore, QtWidgets, uic


class ExperimentWorkspace(QtWidgets.QWidget):
    def __init__(self, services, parent=None):
        super().__init__(parent)
        self.services = services
        uic.loadUi(str(Path(__file__).with_suffix(".ui")), self)
        self.plan_editor = None
        self._components = []
        self._active = True
        self._shutdown = False
        self.contentSplitter.setStretchFactor(0, 2)
        self.contentSplitter.setStretchFactor(1, 3)
        self.contentSplitter.setStretchFactor(2, 0)
        QtCore.QTimer.singleShot(0, self._set_initial_sizes)

    def _set_initial_sizes(self):
        self.contentSplitter.setSizes([500, 900, 300])

    @staticmethod
    def _install(slot, widget):
        if slot.layout().count():
            raise ValueError(f"Slot {slot.objectName()} already contains a widget")
        slot.layout().addWidget(widget)

    def install_shared_controls(self, *, source_status, run_engine, plans, console_history, plan_editor):
        self._install(self.sourceStatusSlot, source_status)
        self._install(self.runEngineSlot, run_engine)
        self._install(self.plansSlot, plans)
        self._install(self.consoleSlot, console_history)
        self.plan_editor = plan_editor

    def install_bluesky_controls(self, *, source_status_factory, editor_options=None,
                                 estimation_context=None, macros=None):
        """Install and own the reusable Run Engine, editor, queue and console."""
        from control_ui.core.lifecycle import DisplayOwner
        from control_ui.widgets.re_control_panel import REControlPanel
        from control_ui.widgets.re_plans import REPlans
        from control_ui.widgets.re_extras import REPlans as ConsoleHistory

        factories = (
            source_status_factory,
            lambda: REControlPanel(services=self.services, macros=macros),
            lambda: REPlans(services=self.services, macros=macros,
                            editor_options=editor_options, estimation_context=estimation_context),
            lambda: ConsoleHistory(services=self.services, macros=macros),
        )
        owners = [DisplayOwner(factory, self.services) for factory in factories]
        source, self.run_engine_controls, self.plan_controls, self.console_history = (
            owner.widget for owner in owners
        )
        self.install_shared_controls(
            source_status=source, run_engine=self.run_engine_controls,
            plans=self.plan_controls, console_history=self.console_history,
            plan_editor=self.plan_controls.re_plan_editor,
        )
        for owner in owners:
            self.register_component(owner)

    def install_experiment(self, *, viewer, controls, acquisition):
        self._install(self.viewerSlot, viewer)
        self._install(self.controlsSlot, controls)
        self._install(self.acquisitionSlot, acquisition)

    def add_tab(self, widget, title, *, index=None):
        if index is None:
            return self.viewerTabs.addTab(widget, title)
        return self.viewerTabs.insertTab(index, widget, title)

    def load_proposed_plan(self, item):
        if self.plan_editor is None:
            raise RuntimeError("No plan editor is installed")
        return self.plan_editor.load_new_plan_item(dict(item), preserve_existing=True)

    def register_component(self, component):
        """Register an owner exposing activate/deactivate/shutdown hooks."""
        if component in self._components:
            raise ValueError("Component is already registered")
        self._components.append(component)

    def deactivate(self):
        if not self._active or self._shutdown:
            return
        for component in reversed(self._components):
            component.deactivate()
        self._active = False

    def activate(self):
        if self._active or self._shutdown:
            return
        for component in self._components:
            component.activate()
        self._active = True

    def shutdown(self):
        if self._shutdown:
            return
        self.deactivate()
        for component in reversed(self._components):
            component.shutdown()
        self._shutdown = True
