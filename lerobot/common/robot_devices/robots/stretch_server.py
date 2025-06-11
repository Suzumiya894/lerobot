import socket

import time
import torch

from lerobot.common.robot_devices.robots.configs import StretchRobotConfig
from lerobot.common.utils.remote_utils import recv_msg, send_msg

class MockCamera:
    fps = 30

class StretchRobotServer:
    """
    Substitute for StretchRobot class, used for remote control of the Stretch Robot.
    Should run scripts/stretch_client_control.py on the real Stretch Robot to connect to this server.
    """
    def __init__(self, config: StretchRobotConfig):

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

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        print(f"接收到来自机器人的数据。Observation.state: {observation_data.get('observation.state', None)}")
        return observation_data
    
    def send_action(self, position: torch.Tensor) -> torch.Tensor:
        print(f"准备发送动作指令: {position}")
        send_msg(self.conn, position)
        print("动作指令已发送。")
        return position
    
    @property
    def camera_features(self) -> dict:
        return {'observation.images.head': {'shape': (640, 480, 3), 'names': ['height', 'width', 'channels'], 'info': None}, 'observation.images.wrist': {'shape': (480, 640, 3), 'names': ['height', 'width', 'channels'], 'info': None}}
    
    @property
    def motor_features(self) -> dict:
        observation_states = ["lift.pos", "arm.pos", "wrist_pitch.pos", "wrist_roll.pos", "wrist_yaw.pos", "gripper.pos", "base_x.pos", "base_y.pos", "base_theta.pos", ]    # 11个自由度，记录关节位置
        action_spaces = ["lift.next_pos", "arm.next_pos", "wrist_pitch.next_pos", "wrist_roll.next_pos", "wrist_yaw.next_pos", "gripper.next_pos", "base_x.next_pos", "base_y.next_pos", "base_theta.next_pos",] # 11个自由度，记录关节速度
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
    
    def is_homed(self):
        return True
    
    def home(self):
        # TODO
        return
    
    def head_look_at_end(self):
        # TODO
        return 
    
