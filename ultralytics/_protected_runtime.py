"""Load the bundled FSHR modules when the source checkout is encrypted."""

import importlib.abc
import importlib.util
import platform
import sys
from pathlib import Path


def enable_protected_runtime():
    root = Path(__file__).resolve().parent
    source = root / "nn" / "modules" / "__init__.py"
    signature = b"\x00GITCRYPT\x00"
    with source.open("rb") as handle:
        locked = handle.read(len(signature)) == signature
    if not locked:
        return

    system = {"Windows": "win_amd64", "Linux": "linux_x86_64"}.get(platform.system())
    supported = (
        sys.implementation.name == "cpython"
        and sys.version_info[:2] == (3, 11)
        and platform.machine().lower() in {"amd64", "x86_64"}
        and sys.maxsize > 2**32
        and system is not None
    )
    if not supported:
        raise ImportError("The bundled FSHR runtime requires 64-bit CPython 3.11 on Windows or Linux (x86_64).")

    folder = root / "_fshr_runtime" / f"cp311-{system}"
    suffix = ".cp311-win_amd64.pyd" if system == "win_amd64" else ".cpython-311-x86_64-linux-gnu.so"
    entry = folder / ("__init__" + suffix)
    if not entry.is_file():
        raise ImportError("The FSHR runtime is missing. Download the complete FSHR-Net repository.")

    class RuntimeFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "ultralytics.nn.modules":
                return importlib.util.spec_from_file_location(
                    fullname, entry, submodule_search_locations=[str(folder)]
                )
            return None

    sys.meta_path.insert(0, RuntimeFinder())
