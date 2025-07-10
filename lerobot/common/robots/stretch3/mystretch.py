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

import time
import math
from dataclasses import replace
from functools import cached_property
from typing import Any
import numpy as np
from stretch_body.gamepad_teleop import GamePadTeleop
from stretch_body.robot import Robot as StretchAPI
from stretch_body.robot_params import RobotParams

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import OBS_IMAGES, OBS_STATE
from lerobot.common.datasets.utils import get_nested_item

from ..robot import Robot
from .configuration_stretch3 import Stretch3RobotConfig

class MyStretchRobot(Robot):
    """My Implementation of Stretch3Robot"""
    config_class = Stretch3RobotConfig
    name = "stretch3"

    STRETCH_STATE = ["head_pan", "head_tilt", "lift", "arm", "wrist_pitch", "wrist_roll", "wrist_yaw", "gripper", "base_x", "base_y", "base_theta"]
    def __init__(self, config: Stretch3RobotConfig):
        super().__init__(config)
        self.config = config

        self.robot_type = self.config.type
        self.cameras_configs = self.config.cameras

        self.api = StretchAPI()
        self.cameras = make_cameras_from_configs(self.cameras_configs)

        self._is_connected = False
        self.teleop = None
        self.logs = {}

        # TODO(aliberts): test this
        RobotParams.set_logging_level("WARNING")
        RobotParams.set_logging_formatter("brief_console_formatter")

        self.state_keys = None
        self.action_keys = None

        self.fast_reset_count = 0
        self.control_mode = self.config.control_mode
        self.control_action_use_head = config.control_action_use_head
        self.control_action_base_only_x = config.control_action_base_only_x

        self.observation_states = [i + ".pos" for i in self.STRETCH_STATE]
        self.action_spaces = [i + ".next_pos" if self.control_mode == "pos" else i + ".vel" for i in self.STRETCH_STATE]
        if not self.control_action_use_head:
            self.observation_states = self.observation_states[2:]
            self.action_spaces = self.action_spaces[2:]
        if self.control_action_base_only_x:
            self.observation_states = self.observation_states[:-2]
            self.action_spaces = self.action_spaces[:-2]

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        # return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}
        return {"navigation":(1280, 720, 3), "head":(640, 480, 3), "wrist":(480, 640, 3)}   # 考虑旋转之后的features
    
    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            self.observation_states, float
        )
    
    @cached_property
    def action_features(self) -> dict:
        return dict.fromkeys(
            self.action_spaces, float
        )

    @cached_property
    def observation_features(self) -> dict:
        return {**self._state_ft, **self._cameras_ft}

    def connect(self) -> None:
        self._is_connected = self.api.startup()
        if not self._is_connected:
            print("Another process is already using Stretch. Try running 'stretch_free_robot_process.py'")
            raise ConnectionError()

        for name in self.cameras:
            self.cameras[name].connect()
            self._is_connected = self._is_connected and self.cameras[name].is_connected

        if not self._is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.calibrate()

    def calibrate(self) -> None:
        if not self.api.is_homed():
            self.api.home()
        self.head_look_at_end()
        self.api.base.reset_odometry()  # 重置底盘位置
        self.api.base.pull_status()     # 确保底盘坐标重置为0
    
    @property
    def is_calibrated(self) -> bool:
        # TODO
        return self.api.is_homed()

    def configure(self):
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        在记录数据时，将各个关节的速度作为action存下来，而不是手柄输入。手柄输入现在仅用于控制机器人。
        """
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self._is_connected:
            raise ConnectionError()

        if self.teleop is None:
            self.teleop = GamePadTeleop(robot_instance=False)
            self.teleop.startup(robot=self)

        before_read_t = time.perf_counter()
        state, velocity = self._get_state_and_velocity()  # 获取状态和速度
        action = self.teleop.gamepad_controller._get_state()

        # 限制左摇杆只能前后移动，即机器人沿x轴移动
        action['left_stick_x'] = 0.0
        
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        before_write_t = time.perf_counter()
        # self.teleop.do_motion(robot=self)       
        self.teleop.do_motion(state=action, robot=self)     # 使用之前读到手柄输入，而不是重新等待
        self.api.push_command()
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if self.state_keys is None:
            self.state_keys = list(state)

        if not record_data:
            return

        state = np.array(list(state.values()))
        velocity = np.array(list(velocity.values()))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        action_dict = {}
        obs_dict = {**state, **images}
        action_dict["action"] = velocity    # 使用关节的速度作为action
        for name in self.cameras:
            # obs_dict[f"{OBS_IMAGES}.{name}"] = images[name]
            obs_dict[name] = images[name]

        return obs_dict, action_dict

    def _get_state(self) -> dict:
        status = self.api.get_status()
        return {
            "head_pan.pos": status["head"]["head_pan"]["pos"],
            "head_tilt.pos": status["head"]["head_tilt"]["pos"],
            "lift.pos": status["lift"]["pos"],
            "arm.pos": status["arm"]["pos"],
            "wrist_pitch.pos": status["end_of_arm"]["wrist_pitch"]["pos"],
            "wrist_roll.pos": status["end_of_arm"]["wrist_roll"]["pos"],
            "wrist_yaw.pos": status["end_of_arm"]["wrist_yaw"]["pos"],
            "gripper.pos": status["end_of_arm"]["stretch_gripper"]["pos"],
            "base_x.pos": status["base"]["x"],
            "base_y.pos": status["base"]["y"],
            "base_theta.pos": status["base"]["theta"],
        }
    
    def _get_state_and_velocity(self) -> tuple[dict, dict]:
        """
        只调用一次self.get_status()，确保velocity和position是同一时刻的
        """
        status = self.api.get_status()
        state = {
            "head_pan.pos": status["head"]["head_pan"]["pos"],
            "head_tilt.pos": status["head"]["head_tilt"]["pos"],
            "lift.pos": status["lift"]["pos"],
            "arm.pos": status["arm"]["pos"],
            "wrist_pitch.pos": status["end_of_arm"]["wrist_pitch"]["pos"],
            "wrist_roll.pos": status["end_of_arm"]["wrist_roll"]["pos"],
            "wrist_yaw.pos": status["end_of_arm"]["wrist_yaw"]["pos"],
            "gripper.pos": status["end_of_arm"]["stretch_gripper"]["pos"],
            "base_x.pos": status["base"]["x"],
            "base_y.pos": status["base"]["y"],
            "base_theta.pos": status["base"]["theta"],
        }
        velocity = {
            "head_pan.vel": status["head"]["head_pan"]["vel"],
            "head_tilt.vel": status["head"]["head_tilt"]["vel"],
            "lift.vel": status["lift"]["vel"],
            "arm.vel": status["arm"]["vel"],
            "wrist_pitch.vel": status["end_of_arm"]["wrist_pitch"]["vel"],
            "wrist_roll.vel": status["end_of_arm"]["wrist_roll"]["vel"],
            "wrist_yaw.vel": status["end_of_arm"]["wrist_yaw"]["vel"],
            "gripper.vel": status["end_of_arm"]["stretch_gripper"]["vel"],
            "base_x.vel": status["base"]["x_vel"],
            "base_y.vel": status["base"]["y_vel"],
            "base_theta.vel": status["base"]["theta_vel"],
        }
        return state, velocity

    def reset_to_home(self) -> None:
        """
        手动将机器人复位到默认位置，而不是调用home()。时间上有所提升，但准确度可能不如home()。
        """
        time.sleep(0.5)
        if self.fast_reset_count >= 10:
            # 每10次强制调用一次home()进行校准归位，避免累积误差
            self.api.home()
            self.fast_reset_count = 0
            return

        HOME_POS = {'head_pan.pos': -1.57, 'head_tilt.pos': -0.787, 'lift.pos': 0.60, 'arm.pos': 0.10, 'wrist_pitch.pos': -0.628, 'wrist_roll.pos': 0.0, 'wrist_yaw.pos': 0.00, 'gripper.pos': 0.00}    # 归位时不再归位底盘，而是手动归位底盘
        self.send_action(HOME_POS, control_mode='pos')  # 使用绝对位置控制机器人归位
        state = self._get_state()
        meter_delta, rad_delta = 0.01, 0.08  # 位置和角度的容差
        force_home = False
        for key, value in state.items():
            if key not in HOME_POS:
                continue
            if 'head' in key or 'wrist' in key or 'gripper' in key:
                delta = rad_delta  # 头部和腕部关节使用弧度容差
            else:
                delta = meter_delta
            if abs(value - HOME_POS[key]) > delta:
                print(f"Warning: {key} is not at home position. Current: {value}, Expected: {HOME_POS[key]}. Use home() instead.")
                force_home = True
        if force_home:
            self.api.home()  # 如果有偏差，调用home()进行精确复位
            self.fast_reset_count = 0
            return 
        self.fast_reset_count += 1
        self.api.base.reset_odometry()  # 重置底盘位置
        self.api.base.pull_status()     # 确保底盘坐标重置为0

    def reset_base_odometry(self) -> None:
        """
        重置底盘位置信息为0。
        """
        if not self._is_connected:
            raise ConnectionError()
        self.api.base.reset_odometry()

    def get_observation(self) -> dict[str, np.ndarray]:

        # Read Stretch state
        before_read_t = time.perf_counter()
        state = self._get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        if self.state_keys is None:
            self.state_keys = list(state)

        # state = np.asarray(list(state.values()))
        # obs_dict[OBS_STATE] = state
        obs_dict = {**state}

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            before_camread_t = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            self.logs[f"async_read_camera_{cam_key}_dt_s"] = time.perf_counter() - before_camread_t

        return obs_dict
 
    def move_to_base_pos(self, target_pose: tuple, error:float=0.01) -> None:
        """
        控制机器人移动到目标位置。

        参数:
            target_pose (tuple): 机器人的目标姿态 (x, y, theta_radians)。
                                x, y 为坐标，theta_radians 为朝向（弧度）。
        """

        def normalize_angle(angle_radians):
            """将角度归一化到 [-pi, pi] 区间。"""
            # math.atan2(sin(angle), cos(angle)) 会自动处理象限并返回 [-pi, pi] 内的值
            return math.atan2(math.sin(angle_radians), math.cos(angle_radians))

        state = self._get_state()
        initial_x, initial_y, initial_theta = state['base_x.pos'], state['base_y.pos'], state['base_theta.pos']
        target_x, target_y, target_theta = target_pose

        delta_x = target_x - initial_x
        delta_y = target_y - initial_y
        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)

        orientation_after_translation = initial_theta # 默认情况下，如果没有移动，朝向不变

        # 情况0: 如果已经到达目标位置点，则仅执行最终旋转
        if distance_to_target > error:
            # 计算从当前位置指向目标位置点的绝对角度
            angle_to_target_point = math.atan2(delta_y, delta_x)

            # 计算机器人需要旋转多少才能正对目标点 (如果打算前进)
            rotation_to_face_target_directly = normalize_angle(angle_to_target_point - initial_theta)

            move_backward = False
            # 初始假定：执行为了正对目标点而进行的旋转
            rotation1_to_perform = rotation_to_face_target_directly

            # 判断目标点是否大致在机器人后方
            # 后方区域定义为: rotation_to_face_target_directly > π/2 或 < -π/2 (不包括π/2 和 -π/2)
            # 使用一个小容差 (epsilon) 进行浮点数比较
            epsilon = 1e-9
            is_target_in_rear_quadrants = (rotation_to_face_target_directly > (math.pi / 2 + epsilon)) or \
                                        (rotation_to_face_target_directly < (-math.pi / 2 - epsilon))

            if is_target_in_rear_quadrants:
                if distance_to_target <= 1.0:
                    move_backward = True
                    # 要后退到 angle_to_target_point, 机器人正面应朝向 angle_to_target_point + pi
                    facing_direction_for_backward_move = normalize_angle(angle_to_target_point + math.pi)
                    rotation1_to_perform = normalize_angle(facing_direction_for_backward_move - initial_theta)

            # --- 步骤 1: 执行第一次旋转 ---
            if abs(rotation1_to_perform) > error:
                self.api.base.rotate_by(rotation1_to_perform)
                self.api.push_command()
                self.api.wait_command()

            orientation_after_rotation1 = normalize_angle(initial_theta + rotation1_to_perform)

            # --- 步骤 2: 执行平移 ---
            if move_backward:
                self.api.base.translate_by(-distance_to_target)
                self.api.push_command()
                self.api.wait_command()
            else:
                self.api.base.translate_by(distance_to_target)
                self.api.push_command()
                self.api.wait_command()
            
            orientation_after_translation = orientation_after_rotation1 # 移动不改变自身朝向

        # --- 步骤 3: 旋转至最终的目标朝向 ---
        # 此时机器人已在目标位置 (target_x, target_y)
        # 其当前朝向为 orientation_after_translation
        rotation2 = target_theta - orientation_after_translation
        rotation2_normalized = normalize_angle(rotation2)

        if abs(rotation2_normalized) > error:
            self.api.base.rotate_by(rotation2_normalized)
            self.api.push_command()
            self.api.wait_command()

    def head_look_at_end(self) -> None:
        self.api.head.pose('tool')
        self.api.wait_command()

    def get_action(self) -> dict[str, Any]:
        """
        获得当前的action，用于在record - teleop过程中将action记录下来。\n
        尽管Stretch在控制时分为pos和vel两种模式，但是在record采数据过程中统一保存vel为action，后续如有需要则要对数据集进行处理。
        """
        if not self._is_connected:
            raise ConnectionError()
        
        _, velocity = self._get_state_and_velocity()  # 获取各关节状态和速度
        return velocity

    def send_action(self, action_args: dict[str, Any], control_mode: str = None) -> dict[str, Any]:
        if control_mode is None:
            control_mode = self.control_mode
        assert control_mode in ['pos', 'vel'], "Control mode must be either 'pos' or 'vel'."

        # 将 action_args 的键从 "head_pan.vel" 转换为 "head_pan" 等形式
        parsed_action_args = {key.split('.')[0] : value for key, value in action_args.items()}

        if control_mode == 'pos':
            self.send_action_pos(parsed_action_args)
        elif control_mode == 'vel':
            self.send_action_vel(parsed_action_args)
        return action_args

    def send_action_vel(self, velocity: dict[str, Any]) -> dict[str, Any]:
        vel_to_pos_coeff = 0.1 # TODO(yew): 针对不同关节，是否可以采用不同的系数？
        if not self._is_connected:
            raise ConnectionError()

        # ["head_pan.vel", "head_tilt.vel", "lift.vel", "arm.vel", "wrist_pitch.vel", "wrist_roll.vel", "wrist_yaw.vel", "gripper.vel", "base_x.vel", "base_y.vel", "base_theta.vel"]
        # 注意：Stretch机器人底盘只能沿着x轴移动，因此base_y.vel永远为0。

        origin_pos = self._get_state()
        print("Origin position is ", origin_pos)
        print("target velocity is ", velocity)

        if "base_theta" in velocity:
            # 使用速度控制底盘时，先旋转底盘，然后再沿着x轴平移。
            self.api.base.rotate_by(velocity["base_theta"] * vel_to_pos_coeff)
            self.api.push_command()
            self.api.wait_command()

        if "base_x" in velocity:
            self.api.base.translate_by(velocity["base_x"] * vel_to_pos_coeff)
            self.api.push_command()
            self.api.wait_command()

        if "head_pan" in velocity:
            self.api.head.move_to("head_pan", origin_pos["head_pan.pos"] + velocity["head_pan"] * vel_to_pos_coeff)

        if "head_tilt" in velocity:
            self.api.head.move_to("head_tilt", origin_pos["head_tilt.pos"] + velocity["head_tilt"] * vel_to_pos_coeff)
        
        if "wrist_pitch" in velocity:
            self.api.end_of_arm.move_to("wrist_pitch", origin_pos["wrist_pitch.pos"] + velocity["wrist_pitch"] * vel_to_pos_coeff)
        if "wrist_roll" in velocity:
            self.api.end_of_arm.move_to("wrist_roll", origin_pos["wrist_roll.pos"] + velocity["wrist_roll"] * vel_to_pos_coeff)
        if "wrist_yaw" in velocity:
            self.api.end_of_arm.move_to("wrist_yaw", origin_pos["wrist_yaw.pos"] + velocity["wrist_yaw"] * vel_to_pos_coeff)
        # 夹爪采集的数据为pos，范围在-5.5~5.5之间；夹爪控制使用的是pos_pct，范围在-100~100之间，需要归一化
        if "gripper" in velocity:
            self.api.end_of_arm.move_to("stretch_gripper", (origin_pos["gripper.pos"] + velocity["gripper"] * vel_to_pos_coeff) * 100 / 5.5)
        self.api.wait_command()

        if "lift" in velocity:
            self.api.lift.move_to(origin_pos["lift.pos"] + velocity["lift"] * vel_to_pos_coeff)
        if "arm" in velocity:
            self.api.arm.move_to(origin_pos["arm.pos"] + velocity["arm"] * vel_to_pos_coeff)
        self.api.push_command()
        self.api.wait_command()
        
        return velocity


    def send_action_pos(self, position: dict[str, Any]) -> dict[str, Any]:
        """
        使用关节绝对值控制机器人，底层调用stretch_body提供的api。
        """
        #TODO(yew): 操控各个关节的顺序是否会对结果有影响？

        if not self._is_connected:
            raise ConnectionError()

        # ["head_pan.pos", "head_tilt.pos", "lift.pos", "arm.pos", "wrist_pitch.pos", "wrist_roll.pos", "wrist_yaw.pos", "gripper.pos", "base_x.pos", "base_y.pos", "base_theta.pos", ]

        print("Origin position is ", self._get_state())
        print("target position is ", position)

        if "base_x" in position:
            if "base_y" in position and "base_theta" in position:
                self.move_to_base_pos(
                    target_pose=(position["base_x"], position["base_y"], position["base_theta"])
                )
            else:
                self.move_to_base_pos(
                    target_pose=(position["base_x"], 0.0, 0.0)
                )

        if "head_pan" in position:
            self.api.head.move_to("head_pan", position["head_pan"])
        if "head_tilt" in position:
            self.api.head.move_to("head_tilt", position["head_tilt"])

        if "wrist_pitch" in position:
            self.api.end_of_arm.move_to("wrist_pitch", position["wrist_pitch"])
        if "wrist_roll" in position:
            self.api.end_of_arm.move_to("wrist_roll", position["wrist_roll"])
        if "wrist_yaw" in position:
            self.api.end_of_arm.move_to("wrist_yaw", position["wrist_yaw"])
        # 夹爪采集的数据为pos，范围在-5.5~5.5之间；夹爪控制使用的是pos_pct，范围在-100~100之间，需要归一化
        if "gripper" in position:
            self.api.end_of_arm.move_to("stretch_gripper", position["gripper"] * 100 / 5.5)
        self.api.wait_command()

        if "lift" in position:
            self.api.lift.move_to(position["lift"])
        if "arm" in position:
            self.api.arm.move_to(position["arm"])
        self.api.push_command()
        self.api.wait_command()

        print("Final position is ", self._get_state())

        return position

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def teleop_safety_stop(self) -> None:
        if self.teleop is not None:
            self.teleop._safety_stop(robot=self)

    def disconnect(self) -> None:
        self.api.stop()
        if self.teleop is not None:
            self.teleop.gamepad_controller.stop()
            self.teleop.stop()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self._is_connected = False

    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    # 在__del__中调用self.api.stop()会出现报错，原因是此时self.api，即stretch_body.robot.Robot中依赖的类已经被销毁。改为每次手动执行disconnect()或使用with语句。
    # def __del__(self):
    #     self.disconnect()
