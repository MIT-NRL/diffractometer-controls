"""MITR catalog adapter for the reusable diffraction analysis window."""

from diffraction_analysis.window import DiffractionAnalysisWindow


class DiffractionHistoryWindow(DiffractionAnalysisWindow):
    def __init__(self, client, *, mode="production", selected_uid="", parent=None):
        super().__init__(
            client, catalogs={"Production": "4dh4", "Test": "testdb"},
            catalog="Test" if mode == "test" else "Production",
            selected_uid=selected_uid, parent=parent,
        )
