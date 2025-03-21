# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
__version__ = "0.0.1"

import importlib
import pkgutil
from pathlib import Path
import subprocess
import sys


def ensure_package(package_name):
    try:
        importlib.import_module(package_name)
    except ImportError:
        print(f"[INFO] {package_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"[INFO] {package_name} installed successfully.")


def _import_dataclass():
    data_pkg = __name__ + ".data"
    package_dir = Path(__file__).parent / "data"
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if not module_info.name.startswith("_"):
            full_module_name = f"{data_pkg}.{module_info.name}"
            importlib.import_module(full_module_name)


_import_dataclass()
