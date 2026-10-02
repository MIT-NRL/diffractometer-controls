from server.context import StartupContext
from server.services.file_directories import create_service

_context = StartupContext(globals())
_context.publish(create_service(_context))
atexit.register(_stop_file_dir_choices_stream)
_start_file_dir_choices_stream()
