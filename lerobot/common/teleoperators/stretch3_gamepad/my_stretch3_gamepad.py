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

from functools import cached_property
from typing import Any

from stretch_body.robot import Robot as StretchAPI
from stretch_body.gamepad_teleop import GamePadTeleop
from stretch_body.robot_params import RobotParams

from ..teleoperator import Teleoperator
from .configuration_stretch3 import Stretch3GamePadConfig

class Stretch3GamePad(Teleoperator):
    config_class = Stretch3GamePadConfig
    name = "stretch3"
    GAMEPAD_BUTTONS = [
        "middle_led_ring_button_pressed",
        "left_stick_x",
        "left_stick_y",
        "right_stick_x",
        "right_stick_y",
        "left_stick_button_pressed",
        "right_stick_button_pressed",
        "bottom_button_pressed",
        "top_button_pressed",
        "left_button_pressed",
        "right_button_pressed",
        "left_shoulder_button_pressed",
        "right_shoulder_button_pressed",
        "select_button_pressed",
        "start_button_pressed",
        "left_trigger_pulled",
        "right_trigger_pulled",
        "bottom_pad_pressed",
        "top_pad_pressed",
        "left_pad_pressed",
        "right_pad_pressed",
    ]
    def __init__(self, config: Stretch3GamePadConfig):
        super().__init__(config)
        self.config = config

        self.api = GamePadTeleop(robot_instance=False)
        self.robot = None
        self._is_connected = False

        RobotParams.set_logging_level("WARNING")
        RobotParams.set_logging_formatter("brief_console_formatter")

    def set_robot(self, robot: StretchAPI) -> None:
        """Set the robot instance for the teleoperator. Should be called before connecting."""
        self.robot = robot


    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self.GAMEPAD_BUTTONS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        assert self.robot is not None and isinstance(self.robot, StretchAPI), "Robot instance must be set before connecting. Use `set_robot(robot)` to set it."
        self.api.startup(self.robot)
        self.api._update_state()  # Check controller can be read & written
        self.api._update_modes()
        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        # TODO(yew): ?Why do Teleoperator and Robot both have a calibrate method?
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        action = self.api.gamepad_controller.get_state()
        if self.config.control_action_base_only_x:
            # 限制左摇杆只能前后移动，即机器人沿x轴移动（手柄左摇杆的y轴对应机器人的x轴）
            action['left_stick_x'] = 0.0
        return action
    
    def send_action(self, action_args: dict[str, Any]) -> None:
        """
        Send action to control the robot.
        """
        if not self.is_connected:
            raise RuntimeError("Teleoperator is not connected. Call `connect()` before sending actions.")
        
        self.api.do_motion(state=action_args, robot=self.robot)
        self.robot.push_command()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def safety_stop(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Teleoperator is not connected. Call `connect()` before sending actions.")
        self.api._safety_stop(robot=self.robot)

    def disconnect(self) -> None:
        self.api.gamepad_controller.stop()
        self.api.stop()
        self._is_connected = False