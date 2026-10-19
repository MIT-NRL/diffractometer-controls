# Archived generated display resources

These files are retained for later removal. Runtime lookup uses original ADLs
from installed EPICS support modules and custom screens from their own
repositories. This directory is excluded from the GUI display search path.
The active external-display manifest is
`diffractometer_controls/site/mitr/external_displays.json`; the JSON here is a
historical record.

These existing areaDetector, motor and IOC-statistics displays moved from
`extra_ui/autoconvert/`; their layout content and macros are preserved. Original
source revision: `806496621ba808f7c9c5994030f71788a3350cb1`. The repository did not
record a converter version or complete upstream revision; none is inferred.

Maintain hand-edited components beside their Python files under `control_ui`
or the instrument/site directory. Vendor regeneration is separate work.

Some related displays have always come from installed EPICS
synApps/areaDetector/motor support or custom detector repositories.
`external_references.json` records 83 unbundled filenames and their original
baseline references. Supply them through the host's display search paths.
The launcher retains user `PYDM_DISPLAYS_PATH` overrides, excludes this archive,
and adds project and configured EPICS support paths.

Structural tests check active project panels against the current manifest;
they no longer depend on this archive.
