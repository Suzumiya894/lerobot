import socket

import time
import numpy as np

from ..robot import Robot
from .configuration_stretch3 import Stretch3RobotConfig
from lerobot.common.utils.remote_utils import recv_msg, send_msg

class MockCamera:
    fps = 30

class StretchRobotServer(Robot):
    """
    Substitute for StretchRobot class, used for remote control of the Stretch Robot.
    Should run scripts/stretch_client_control.py on the real Stretch Robot to connect to this server.
    """
    STRETCH_STATE = ["head_pan", "head_tilt", "lift", "arm", "wrist_pitch", "wrist_roll", "wrist_yaw", "gripper", "base_x", "base_y", "base_theta"]
    def __init__(self, config: Stretch3RobotConfig):

        print("Warning: This is the implementation of Stretch robot server, used for controlling the Stretch Robot remotely.\nIf this is not what you want, check lerobot/common/robot_devices/robots/configs.py and set is_remote_server to False.")

        self.is_connected = False
        self.host = "10.176.44.2"  # 本地地址
        self.port = config.server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置地址重用，这样即使程序异常退出，端口也能立即被重新使用
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print(f"Stretch robot server started at {self.host}:{self.port}")
        self.conn, self.addr = None, None

        self.cameras = {'head': MockCamera(), 'wrist': MockCamera()}  # 模拟摄像头
        self.robot_type = config.type  
        self.logs = {}

        self.config = config

        self.control_mode = config.control_mode
        self.control_action_use_head = config.control_action_use_head
        self.control_action_base_only_x = config.control_action_base_only_x

    def connect(self):
        print("Waiting for connection from Stretch robot...")
        self.conn, self.addr = self.socket.accept()
        if not self.conn:
            raise ConnectionError("Failed to accept connection from Stretch robot.")
        print(f"Connected to Stretch robot at {self.addr}")
        self.is_connected = True

    def disconnect(self):
        """Disconnect from the Stretch robot server."""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Disconnected from Stretch robot.")
        self.is_connected = False

    def __del__(self):
        self.disconnect()

    def capture_observation(self) -> dict:
        print("等待接收来自Stretch机器人的观察数据...")
        before_read_t = time.perf_counter()
        observation_data = recv_msg(self.conn)
        if observation_data is None:
            raise ConnectionError("Failed to receive data.")

        if not self.control_action_use_head:
            observation_data['observation.state'] = observation_data['observation.state'][2:]
        if self.control_action_base_only_x:
            observation_data['observation.state'] = observation_data['observation.state'][:-2]
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        print(f"接收到来自机器人的数据。Observation.state: {observation_data.get('observation.state', None)}")
        return observation_data
    
    def send_action(self, action_args: np.ndarray) -> np.ndarray:
        print(f"准备发送动作指令: {action_args}")
        send_msg(self.conn, action_args)
        print("动作指令已发送。")
        return action_args
    
    @property
    def camera_features(self) -> dict:
        # 相机只包含训练的模型使用的特征，默认navigation相机不包含在内
        return {'observation.images.head': {'shape': (640, 480, 3), 'names': ['height', 'width', 'channels'], 'info': None}, 'observation.images.wrist': {'shape': (480, 640, 3), 'names': ['height', 'width', 'channels'], 'info': None}}
    
    @property
    def motor_features(self) -> dict:
        observation_states = [i + ".pos" for i in self.STRETCH_STATE]
        action_spaces = [i + ".next_pos" if self.control_mode == "pos" else i + ".vel" for i in self.STRETCH_STATE]
        if not self.control_action_use_head:
            observation_states = observation_states[2:]
            action_spaces = action_spaces[2:]
        if self.control_action_base_only_x:
            observation_states = observation_states[:-2]
            action_spaces = action_spaces[:-2]
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
    
    @property
    def observation_features(self) -> dict:
        return self.motor_features['observation.state']
    
    @property
    def action_features(self) -> dict:
        return self.motor_features['action']
    
    def is_homed(self):
        return True
    
    def home(self):
        # TODO
        return
    
    def head_look_at_end(self):
        # TODO
        return 
    
