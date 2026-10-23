"""Exercise HTTPX logging with real success/failure responses and no sockets."""

import io
import logging
import os
import unittest
from unittest.mock import patch

import httpx

from control_ui.core.logging_config import configure_http_logging


class HttpLoggingTests(unittest.TestCase):
    def setUp(self):
        self.names = ('httpx', 'httpcore', 'httpcore.http11', 'test-application')
        self.original = {name: logging.getLogger(name).level for name in self.names}
        self.addCleanup(self.restore)

    def restore(self):
        for name, level in self.original.items():
            logging.getLogger(name).setLevel(level)

    def test_default_hides_request_info_preserves_warnings_and_application_errors(self):
        with patch.dict(os.environ, {'TILED_HTTP_DEBUG': ''}):
            self.assertFalse(configure_http_logging())
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        root = logging.getLogger()
        root.addHandler(handler)
        self.addCleanup(root.removeHandler, handler)
        # Native HTTPX emits INFO for both 200 and 500, so request logging is
        # not an error reporting mechanism. HTTP errors still raise normally.
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
            client.get('http://test-server')
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                client.get('http://test-server').raise_for_status()
        logging.getLogger('httpx').warning('HTTP warning retained')
        logging.getLogger('httpcore.http11').error('Transport error retained')
        logging.getLogger('test-application').error('Application error retained')
        self.assertNotIn('HTTP Request:', output.getvalue())
        for message in ('HTTP warning retained', 'Transport error retained', 'Application error retained'):
            self.assertIn(message, output.getvalue())
        self.assertEqual(logging.getLogger('test-application').level, self.original['test-application'])

    def test_debug_opt_in_restores_successful_request_messages(self):
        with patch.dict(os.environ, {'TILED_HTTP_DEBUG': '1'}):
            self.assertTrue(configure_http_logging())
        with self.assertLogs('httpx', level=logging.DEBUG) as logs:
            with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
                client.get('http://test-server')
        self.assertTrue(any('HTTP Request:' in entry for entry in logs.output))
        self.assertEqual(logging.getLogger('httpcore').level, logging.DEBUG)
