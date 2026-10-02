# Generated display resources

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
The launcher retains existing `PYDM_DISPLAYS_PATH` before adding project and
configured EPICS support paths.

Structural tests resolve bundled paths/custom imports and allow only recorded
external filenames. Linux acceptance must verify those files and their macros
against the installed support tree.
