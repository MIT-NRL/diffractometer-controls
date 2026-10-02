"""Enforce reusable package boundaries and source-relative Designer resources."""

import ast
import importlib
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("control_ui", "diffractometer_controls", "server", "vendor")


class RepositoryStructureTests(unittest.TestCase):
    def test_shared_controls_do_not_import_instrument_or_server(self):
        for path in (ROOT / "control_ui").rglob("*.py"):
            with self.subTest(path=path.relative_to(ROOT)):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                               else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                    self.assertFalse(any(name.split(".")[0] in ("diffractometer_controls", "server")
                                         for name in imports), imports)

    def test_package_initializers_have_no_imports_or_execution(self):
        for package in PACKAGES[:3]:
            for path in (ROOT / package).rglob("__init__.py"):
                with self.subTest(path=path.relative_to(ROOT)):
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                    self.assertTrue(all(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                                        and isinstance(node.value.value, str) for node in tree.body))

    def test_all_designer_imports_and_bundled_paths_resolve(self):
        files = [path for package in PACKAGES for path in (ROOT / package).rglob("*.ui")]
        bundled_names = {path.name for package in PACKAGES
                         for path in (ROOT / package).rglob("*") if path.is_file()}
        external = json.loads((ROOT / "vendor/displays/external_references.json").read_text())
        headers = set()
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                tree = ET.parse(path)
                for custom in tree.findall(".//customwidget"):
                    headers.add((custom.findtext("header"), custom.findtext("class")))
                for node in tree.iter():
                    value = (node.text or "").strip()
                    if not value.endswith((".ui", ".adl", ".py", ".svg", ".png")) or "$" in value or "\n" in value:
                        continue
                    if (path.parent / value).exists():
                        continue
                    if "/" in value or "\\" in value:
                        self.fail(f"Missing relative resource: {value}")
                    self.assertTrue(value in bundled_names or value in external["references"], value)
        for header, widget_class in headers:
            with self.subTest(header=header, widget_class=widget_class):
                self.assertTrue(hasattr(importlib.import_module(header), widget_class))


if __name__ == "__main__":
    unittest.main()
