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
import os
import json
from pathlib import Path
from dataclasses import dataclass

import lerobot
from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("stretch3")
@dataclass
class Stretch3GamePadConfig(TeleoperatorConfig):
    mock: bool = False

    control_action_base_only_x: bool = True


project_root = Path(lerobot.__file__).parent.parent
if os.path.isfile(os.path.join(project_root, "local_config.json")):
    with open(os.path.join(project_root, "local_config.json"), "r") as f:
        config_data = json.load(f)
        try:
            print("Use local configuration for Stretch3GamePadConfig.")
            for key, value in config_data["Stretch3GamePadConfig"].items():
                if hasattr(Stretch3GamePadConfig, key):
                    setattr(Stretch3GamePadConfig, key, value)
                    print(f"Set {key} to {value} in Stretch3GamePadConfig.")
        except:
            print("Failed to load local configuration for Stretch3GamePadConfig. Please check the format of local_config.json.")
