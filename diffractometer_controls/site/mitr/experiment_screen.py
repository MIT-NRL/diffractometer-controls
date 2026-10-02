"""MITR composition adapter for the instrument-independent workspace."""

from qtpy import QtWidgets
from diffractometer_controls.site.mitr import profile

from control_ui.layouts.experiment_workspace import ExperimentWorkspace
from diffractometer_controls.site.mitr.display import MITRDisplay




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
        from diffractometer_controls.site.mitr.reactor_power import ReactorPowerDisplay
        macros = self.macros()
        self.workspace = ExperimentWorkspace(self.services, self)
        self.workspace.install_bluesky_controls(
            source_status_factory=lambda: ReactorPowerDisplay(macros=macros, services=self.services),
            editor_options=profile.editor_options(), estimation_context=profile.estimation_context,
            macros=macros,
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
