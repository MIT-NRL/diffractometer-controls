"""Quiet routine HTTP traffic without changing application log levels."""

import logging
import os


def configure_http_logging():
    """Keep warnings/errors by default; opt in with TILED_HTTP_DEBUG=1."""
    debug = os.environ.get('TILED_HTTP_DEBUG', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    level = logging.DEBUG if debug else logging.WARNING
    for name in ('httpx', 'httpcore'):
        logging.getLogger(name).setLevel(level)
    return debug
