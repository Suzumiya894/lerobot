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
from dataclasses import dataclass, field
import lerobot

from lerobot.common.cameras import CameraConfig
from lerobot.common.cameras.opencv import OpenCVCameraConfig
from lerobot.common.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("stretch3")
@dataclass
class Stretch3RobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "navigation": OpenCVCameraConfig(
                index_or_path="/dev/hello-nav-head-camera",
                fps=10,
                width=1280,
                height=720,
                rotation=-90,
            ),
            "head": RealSenseCameraConfig(
                serial_number_or_name="Intel RealSense D435IF",
                fps=30,
                width=640,
                height=480,
                rotation=90,
            ),
            "wrist": RealSenseCameraConfig(
                serial_number_or_name="Intel RealSense D405",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False

    is_remote_server: bool = False
    server_port: int = 65432
    control_mode: str = "pos" # ['pos', 'vel']
    control_action_use_head: bool = False
    control_action_base_only_x: bool = True

project_root = Path(lerobot.__file__).parent.parent
if os.path.isfile(os.path.join(project_root, "local_config.json")):
    with open(os.path.join(project_root, "local_config.json"), "r") as f:
        config_data = json.load(f)
        try:
            print("Use local configuration for Stretch3RobotConfig.")
            for key, value in config_data["Stretch3RobotConfig"].items():
                if hasattr(Stretch3RobotConfig, key):
                    setattr(Stretch3RobotConfig, key, value)
                    print(f"Set {key} to {value} in Stretch3RobotConfig.")
        except:
            print("Failed to load local configuration for Stretch3RobotConfig. Please check the format of local_config.json.")
