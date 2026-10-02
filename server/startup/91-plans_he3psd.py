from server.context import StartupContext
from server.plans.he3psd import create_plans
from ophyd import EpicsSignal
frame_type_sig = EpicsSignal("4dh4:TS:FrameType", name="frame_type_sig")

_context = StartupContext(globals())
_context.publish(create_plans(_context))
for _legacy_plan_name in (
    "count_he3",
    "scan_he3",
    "scan_parallel_he3",
    "scan_list_he3",
    "scan2D_he3",
):
    register_plan(_legacy_plan_name, exclude=True)
