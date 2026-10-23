"""Run the independent Tiled Manager without loading instrument controls."""

import argparse
import logging
import os
import sys

from qtpy import QtWidgets

from . import auth
from .window import TiledManagerWindow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uri', help='Tiled endpoint; defaults to environment or installed reader settings')
    parser.add_argument('--catalog', action='append', metavar='LABEL=PATH', help='Repeat for each catalog')
    args = parser.parse_args(argv)
    from control_ui.core.logging_config import configure_http_logging
    if configure_http_logging():
        logging.basicConfig(level=logging.DEBUG)
    config = auth.read_private_config(auth.READER_PATH) or {}
    uri = args.uri or os.environ.get('TILED_URI') or os.environ.get('MITR_TILED_URI') or config.get('uri') or 'http://localhost:8000'
    catalogs = {}
    for item in args.catalog or ['Production=4dh4', 'Test=testdb']:
        label, separator, path = item.partition('=')
        if not separator or not label or not path:
            parser.error('--catalog must be LABEL=PATH')
        catalogs[label] = path
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = TiledManagerWindow(uri, catalogs=catalogs)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
