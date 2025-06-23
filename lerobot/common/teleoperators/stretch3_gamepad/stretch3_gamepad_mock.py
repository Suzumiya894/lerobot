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

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self.GAMEPAD_BUTTONS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        pass

    def set_robot(self, robot: Any) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        pass

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        pass