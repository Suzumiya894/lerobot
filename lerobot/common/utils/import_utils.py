#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import logging
import json
from pathlib import Path
import lerobot
import os


def is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    """Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py
    Check if the package spec exists and grab its version to avoid importing a local directory.
    **Note:** this doesn't work for all packages.
    """
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)

        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            elif pkg_name == "grpc":
                package = importlib.import_module(pkg_name)
                package_version = getattr(package, "__version__", "N/A")
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logging.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists

def load_local_config(cls):
    """
    一个类装饰器，用于从 local_config.json 文件加载配置并覆盖类的默认属性。
    """
    project_root = Path(lerobot.__file__).parent.parent
    local_config_path = os.path.join(project_root, "local_config.json")
    config_key = cls.__name__  # 例如 "Stretch3RobotConfig"

    if os.path.isfile(local_config_path):
        print(f"Found local_config.json, attempting to override defaults for '{config_key}'.")
        with open(local_config_path, "r") as f:
            try:
                config_data = json.load(f).get(config_key, {})
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        print(f"  - Overriding '{key}' with value '{value}'")
                        setattr(cls, key, value)
                    else:
                        print(f"  - Warning: Key '{key}' from json is not a field in '{config_key}'.")
            except Exception as e:
                print(f"  - Error loading local config: {e}")
    
    # 必须返回这个类
    return cls

_torch_available, _torch_version = is_package_available("torch", return_version=True)
_gym_xarm_available = is_package_available("gym_xarm")
_gym_aloha_available = is_package_available("gym_aloha")
_gym_pusht_available = is_package_available("gym_pusht")
