"""User-private local reader settings on POSIX and Windows."""

import base64
import json
import os
from pathlib import Path
import stat
import tempfile


def _platform():
    return os.name


def _windows_protect(data, *, decrypt=False):
    """Use current-user DPAPI, protecting the endpoint as well as the key."""
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = Blob(len(data), buffer)
    result = Blob()
    library = ctypes.WinDLL("crypt32", use_last_error=True)
    function = library.CryptUnprotectData if decrypt else library.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    # CRYPTPROTECT_UI_FORBIDDEN; deliberately omit CRYPTPROTECT_LOCAL_MACHINE.
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise OSError("Windows could not protect or unlock the local reader configuration.")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        kernel.LocalFree(result.data)


def read_private_config(path):
    """Reject public POSIX files and unprotected/unreadable Windows settings."""
    try:
        with Path(path).open("rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return None
            platform = _platform()
            if platform == "posix":
                if info.st_uid != os.getuid() or info.st_mode & 0o077:
                    return None
            elif platform != "nt":
                return None
            config = json.load(stream)
        if platform == "nt":
            if not isinstance(config, dict) or set(config) != {"dpapi"}:
                return None
            encrypted = base64.b64decode(config["dpapi"], validate=True)
            config = json.loads(_windows_protect(encrypted, decrypt=True))
        return config if isinstance(config, dict) else None
    except (OSError, ValueError, TypeError, UnicodeError):
        return None


def write_private_config(path, config, *, replace=False):
    """Publish only a complete, protected file; preserve existing settings by default."""
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError("Reader configuration already exists; use --replace to change it.")
    payload = json.dumps(config, indent=2).encode("utf-8")
    platform = _platform()
    if platform == "nt":
        payload = json.dumps({"dpapi": base64.b64encode(_windows_protect(payload)).decode("ascii")}).encode("utf-8")
    elif platform != "posix":
        raise OSError("Local reader configuration is unsupported on this platform.")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".tiled-reader-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        elif platform == "nt":
            # Windows rename refuses an existing destination, including when
            # another setup process creates it after the initial check.
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
