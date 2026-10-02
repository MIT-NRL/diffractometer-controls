from server.context import StartupContext
from server.plans.helpers import create_plans

_context = StartupContext(globals())
_context.publish(create_plans(_context))
