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

import torch
from stretch_body.gamepad_teleop import GamePadTeleop
from stretch_body.robot import Robot as StretchAPI
from stretch_body.robot_params import RobotParams

from lerobot.common.robot_devices.robots.configs import StretchRobotConfig
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs

class StretchRobot(StretchAPI):
    """Wrapper of stretch_body.robot.Robot"""

    def __init__(self, config: StretchRobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            self.config = StretchRobotConfig(**kwargs)
        else:
            # Overwrite config arguments using kwargs
            self.config = replace(config, **kwargs)

        self.robot_type = self.config.type
        self.cameras_configs = self.config.cameras

        self.cameras = make_cameras_from_configs(self.cameras_configs)

        self.is_connected = False
        self.teleop = None
        self.logs = {}

        # TODO(aliberts): test this
        RobotParams.set_logging_level("WARNING")
        RobotParams.set_logging_formatter("brief_console_formatter")

        self.state_keys = None
        self.action_keys = None



    @property
    def camera_features(self) -> dict:
        # TODO(yew)： 增加深度图像
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        observation_states = ["head_pan.pos", "head_tilt.pos", "lift.pos", "arm.pos", "wrist_pitch.pos", "wrist_roll.pos", "wrist_yaw.pos", "gripper.pos", "base_x.pos", "base_y.pos", "base_theta.pos", ]    # 11个自由度，记录关节位置
        action_spaces = ["head_pan.vel", "head_tilt.vel", "lift.vel", "arm.vel", "wrist_pitch.vel", "wrist_roll.vel", "wrist_yaw.vel", "gripper.vel", "base_x.vel", "base_y.vel", "base_theta.vel"] # 11个自由度，记录关节速度
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_spaces),),
                "names": action_spaces,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(observation_states),),
                "names": observation_states,
            },
        }

    def connect(self) -> None:
        self.is_connected = self.startup()
        if not self.is_connected:
            print("Another process is already using Stretch. Try running 'stretch_free_robot_process.py'")
            raise ConnectionError()

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.run_calibration()

    def run_calibration(self) -> None:
        if not self.is_homed():
            self.home()

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        在记录数据时，将各个关节的速度作为action存下来，而不是手柄输入。手柄输入现在仅用于控制机器人。
        """
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        if self.teleop is None:
            self.teleop = GamePadTeleop(robot_instance=False)
            self.teleop.startup(robot=self)

        before_read_t = time.perf_counter()
        state, velocity = self.get_state_and_velocity()  # 获取状态和速度
        action = self.teleop.gamepad_controller.get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        before_write_t = time.perf_counter()
        # self.teleop.do_motion(robot=self)       
        self.teleop.do_motion(state=action, robot=self)     # 使用之前读到手柄输入，而不是重新等待
        self.push_command()
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if self.state_keys is None:
            self.state_keys = list(state)

        if not record_data:
            return

        state = torch.as_tensor(list(state.values()))
        velocity = torch.as_tensor(list(velocity.values()))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = velocity    # 使用关节的速度作为action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def get_state(self) -> dict:
        status = self.get_status()
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
    
    def get_state_and_velocity(self) -> tuple[dict, dict]:
        """
        只调用一次self.get_status()，确保velocity和position是同一时刻的
        """
        status = self.get_status()
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

    def capture_observation(self) -> dict:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        before_read_t = time.perf_counter()
        state = self.get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        if self.state_keys is None:
            self.state_keys = list(state)

        state = torch.as_tensor(list(state.values()))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

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

        state = self.get_state()
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
                self.base.rotate_by(rotation1_to_perform)
                self.push_command()
                self.wait_command()

            orientation_after_rotation1 = normalize_angle(initial_theta + rotation1_to_perform)

            # --- 步骤 2: 执行平移 ---
            if move_backward:
                self.base.translate_by(-distance_to_target)
                self.push_command()
                self.wait_command()
            else:
                self.base.translate_by(distance_to_target)
                self.push_command()
                self.wait_command()
            
            orientation_after_translation = orientation_after_rotation1 # 移动不改变自身朝向

        # --- 步骤 3: 旋转至最终的目标朝向 ---
        # 此时机器人已在目标位置 (target_x, target_y)
        # 其当前朝向为 orientation_after_translation
        rotation2 = target_theta - orientation_after_translation
        rotation2_normalized = normalize_angle(rotation2)

        if abs(rotation2_normalized) > error:
            self.base.rotate_by(rotation2_normalized)
            self.push_command()
            self.wait_command()

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # TODO(yew): 传输的action为手柄的输入，长为21的向量，后续需要按照我们规定的动作格式修改该函数，包括修改控制机器人移动的逻辑

        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        if self.teleop is None:
            self.teleop = GamePadTeleop(robot_instance=False)
            self.teleop.startup(robot=self)

        if self.action_keys is None:
            dummy_action = self.teleop.gamepad_controller.get_state()
            self.action_keys = list(dummy_action.keys())

        action_dict = dict(zip(self.action_keys, action.tolist(), strict=True))

        before_write_t = time.perf_counter()
        self.teleop.do_motion(state=action_dict, robot=self)
        self.push_command()
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        # TODO(aliberts): return action_sent when motion is limited
        return action

    def send_pos_action(self, position: torch.Tensor) -> torch.Tensor:
        """
        使用关节绝对值控制机器人，底层调用stretch_body提供的api。
        """
        #TODO(yew): 操控各个关节的顺序是否会对结果有影响？

        if not self.is_connected:
            raise ConnectionError()
        
        # ["head_pan.pos", "head_tilt.pos", "lift.pos", "arm.pos", "wrist_pitch.pos", "wrist_roll.pos", "wrist_yaw.pos", "gripper.pos", "base_x.pos", "base_y.pos", "base_theta.pos", ]

        print("Origin position is ", self.get_state())
        print("target position is ", position)

        before_base_t = time.perf_counter()
        self.move_to_base_pos(
            target_pose=(position[8].item(), position[9].item(), position[10].item())
        )
        self.logs["move_to_base_pos_dt_s"] = time.perf_counter() - before_base_t

        before_head_t = time.perf_counter()
        self.head.move_to("head_pan", position[0].item())
        self.head.move_to("head_tilt", position[1].item())
        self.logs["move_to_head_dt_s"] = time.perf_counter() - before_head_t

        before_wrist_t = time.perf_counter()
        self.end_of_arm.move_to("wrist_pitch", position[4].item())
        self.end_of_arm.move_to("wrist_roll", position[5].item())
        self.end_of_arm.move_to("wrist_yaw", position[6].item())
        # 夹爪采集的数据为pos，范围在-5.5~5.5之间；夹爪控制使用的是pos_pct，范围在-100~100之间，需要归一化
        self.end_of_arm.move_to("stretch_gripper", position[7].item() * 100 / 5.5)
        self.wait_command()
        self.logs["move_to_wrist_dt_s"] = time.perf_counter() - before_wrist_t

        before_arm_lift_t = time.perf_counter()
        self.lift.move_to(position[2].item())
        self.arm.move_to(position[3].item())
        self.push_command()
        self.wait_command()
        self.logs["move_to_lift_arm_dt_s"] = time.perf_counter() - before_arm_lift_t

        print("Final position is ", self.get_state())

        return position

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def teleop_safety_stop(self) -> None:
        if self.teleop is not None:
            self.teleop._safety_stop(robot=self)

    def disconnect(self) -> None:
        self.stop()
        if self.teleop is not None:
            self.teleop.gamepad_controller.stop()
            self.teleop.stop()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def __del__(self):
        self.disconnect()
