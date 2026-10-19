"""Original ADL lookup must work without the archived UI conversions."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydm.utilities import find_file
from diffractometer_controls.site.mitr import profile


class DisplayPathTests(unittest.TestCase):
    def test_user_paths_and_active_module_versions_survive_existing_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support, github, ioc = root / 'support', root / 'github', root / 'ioc'
            current = support / 'areaDetector-current/ADCore/ADApp/op/adl'
            older = support / 'areaDetector-older/ADCore/ADApp/op/adl'
            custom = github / 'ADZWO/zwoApp/op/PyDM'
            override = root / 'custom-displays'
            boot = ioc / 'iocBoot/ioc4dh4-motors'
            for path in (current, older, custom, override, boot):path.mkdir(parents=True)
            for path in (current, older):(path / 'ADBase.adl').write_text('test ADL source')
            (custom / 'ADZWO.ui').write_text('custom camera screen')
            (boot / 'envPaths').write_text(f'epicsEnvSet("ADCORE","{current.parents[2]}")\n')
            archive = profile.CHECKOUT_ROOT / 'vendor/displays'
            environment = {'MITR_IOC_ROOT': str(ioc), 'MITR_EPICS_SUPPORT': str(support),
                           'PYDM_DISPLAYS_PATH': os.pathsep.join(map(str, (override, archive, older)))}
            with patch.dict(os.environ, environment):
                paths = profile.configure_display_paths(github)
                self.assertEqual(paths[0], str(override))
                self.assertNotIn(str(archive), paths)
                self.assertLess(paths.index(str(current)), paths.index(str(older)))
                self.assertEqual(Path(find_file('ADBase.adl')), current / 'ADBase.adl')
                # PyDM's fallback also covers nested .ui links emitted by
                # adl2pydm and links from the custom detector UIs.
                self.assertEqual(Path(find_file('ADBase.ui')), current / 'ADBase.adl')
                self.assertEqual(Path(find_file('ADZWO.ui')), custom / 'ADZWO.ui')
                self.assertEqual(profile.configure_display_paths(github), paths)

    def test_site_panels_never_add_the_conversion_archive(self):
        self.assertFalse(any('vendor' in path.parts for path in profile.display_directories()))


if __name__ == '__main__':
    unittest.main()
