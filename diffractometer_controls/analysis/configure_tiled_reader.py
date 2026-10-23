"""Install a shared or individual read-only Tiled key on this computer."""

import argparse
import getpass
from pathlib import Path
import socket
import warnings

from . import tiled_auth
from .reader_config import write_private_config


READER_SCOPES = frozenset({"read:metadata", "read:data"})


def install_reader_key(uri, api_key, *, path=None, replace=False):
    """Validate the reader key without creating keys or changing the server."""
    tiled_auth._reader_server_id(uri)
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("A read-only Tiled API key is required.")
    api_key = api_key.strip()
    path = Path(path) if path is not None else tiled_auth.LOCAL_READER_CONFIG
    if path.exists() and not replace:
        raise FileExistsError("Reader configuration already exists; use --replace to change it.")
    if tiled_auth.Context is None:
        raise RuntimeError("Tiled authentication helpers are unavailable.")
    context, _parts = tiled_auth.Context.from_any_uri(uri, api_key=api_key)
    try:
        details = context.which_api_key()
        if set(details.get("scopes", [])) != READER_SCOPES:
            raise ValueError("Use a key restricted to read:metadata and read:data.")
    finally:
        context.close()
    write_private_config(path, {"hostname": socket.gethostname(), "uri": str(uri),
                                "api_key": api_key}, replace=replace)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True, help="Tiled server URL, without credentials")
    parser.add_argument("--replace", action="store_true", help="Replace this user's existing local reader settings")
    args = parser.parse_args(argv)
    try:
        # Do not fall back to an input mode that might echo the secret.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("Read-only Tiled API key (hidden): ")
        path = install_reader_key(args.uri, key, replace=args.replace)
    except getpass.GetPassWarning:
        parser.exit(1, "Run this command in a terminal that supports hidden input.\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    except Exception:
        parser.exit(1, "Could not verify the reader key. Check the server URL, connection, and key.\n")
    print(f"Reader access saved to {path}. Restart the GUI to use it.")


if __name__ == "__main__":
    main()
