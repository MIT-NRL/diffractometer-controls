from server.context import StartupContext
from server.plans import helpers, scalar, he3psd, imaging, adaptive_imaging
from ophyd import Signal
frame_type_sig = Signal(name="frame_type_sig", value=0)
_context = StartupContext(globals())
for _module in (helpers, scalar, he3psd, imaging, adaptive_imaging):
    _context.publish(_module.create_plans(_context))
