import asyncio
import time
import numpy as np
from concurrent.futures import Future
from functools import cached_property
import threading

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
    config_class = Stretch3RobotConfig
    name = "stretch3"

    STRETCH_STATE = ["head_pan", "head_tilt", "lift", "arm", "wrist_pitch", "wrist_roll", "wrist_yaw", "gripper", "base_x", "base_y", "base_theta"]
    def __init__(self, config: Stretch3RobotConfig):

        print("Warning: This is the implementation of Stretch robot server, used for controlling the Stretch Robot remotely.\nIf this is not what you want, check lerobot/common/robot_devices/robots/configs.py and set is_remote_server to False.")

        print("Initializing Async Stretch Robot Server...")
        self.host = "0.0.0.0"  # 监听所有网络接口
        self.port = config.server_port
        
        self._is_connected = False
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        
        self.server_task = None
        self.client_handler_task = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.server_thread: threading.Thread | None = None
        
        # 使用线程安全的 asyncio.Event
        self.connection_event = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.server_ready_event = threading.Event()

        self.cameras = {'head': MockCamera(), 'wrist': MockCamera()}  # 模拟摄像头
        self.robot_type = config.type 
        self.config = config
        self.logs = {}

        self.control_mode = config.control_mode
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

        self.api = None # Mock API, no need to set it for the server

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        这是为每个客户端连接调用的回调处理函数。
        """
        if self.is_connected:
            print("已经有一个客户端连接，拒绝新的连接。")
            writer.close()
            await writer.wait_closed()
            return

        self.reader = reader
        self.writer = writer
        self._is_connected = True
        addr = writer.get_extra_info('peername')
        print(f"已连接到机器人客户端: {addr}")
        
        self.connection_event.set()

        try:
            await self.stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            print(f"与 {addr} 的连接正在关闭。")
            self._is_connected = False
            self.connection_event.clear()
            writer.close()
            await writer.wait_closed()

    async def start(self):
        """
        启动服务器并开始监听连接。
        """
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        addr = server.sockets[0].getsockname()
        print(f"服务器已在 {addr} 上启动并监听...")

        async with server:
            await server.serve_forever()


    def wait_for_connection_sync(self, timeout: float = 60.0) -> bool:
        async def wait_for_connection(timeout) -> bool:
            try:
                await asyncio.wait_for(self.connection_event.wait(), timeout)
                return True
            except asyncio.TimeoutError:
                print(f"等待连接超时，超过 {timeout} 秒。")
                return False
        
        if self.loop is None or not self.loop.is_running():
            raise RuntimeError("事件循环未启动，无法等待连接。请先调用 connect() 启动服务器。")
        future = asyncio.run_coroutine_threadsafe(wait_for_connection(timeout), self.loop)
        return future.result()  # 阻塞直到协程完成

    def run_server_forever(self):
        """
        在一个单独的线程中运行服务器事件循环。
        这个方法可以在主线程中调用来启动服务器。
        """
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.server_ready_event.set()
            self.loop.run_until_complete(self.start())
        except asyncio.CancelledError:
            print("服务器事件循环被停止。")
        except Exception as e:
            print(f"服务器发生错误: {e}")
        finally:
            if self.loop is not None:
                self.loop.close()
    
    def connect(self):
        self.server_thread = threading.Thread(target=self.run_server_forever, daemon=True)
        self.server_thread.start()

        server_is_ready = self.server_ready_event.wait(timeout=10.0)
        if not server_is_ready:
            self.disconnect()
            raise RuntimeError("服务器未能在10秒内启动。请检查配置或网络连接。")
        
        print("等待客户端连接...")
        connection_is_ready = self.wait_for_connection_sync(timeout=60)
        if not connection_is_ready:
            self.disconnect()
            raise RuntimeError("未能在60秒内连接到客户端。请检查客户端是否已启动并尝试重新连接。")

    def disconnect(self):
        """
        从外部线程安全地停止服务器。
        """
        if self.loop:
            self.loop.call_soon_threadsafe(self.stop_event.set)
            self.server_thread.join(timeout=2.0)

    # ------------------ 同步接口 ------------------

    def get_observation(self) -> dict | None:
        """
        同步地从客户端获取观察数据。
        这个方法是线程安全的，可以从外部线程调用。
        """
        if not self.loop or not self.is_connected:
            raise ConnectionError("服务器未运行或机器人未连接。")
        
        # 将异步函数的调用提交到事件循环，并等待结果
        future: Future = asyncio.run_coroutine_threadsafe(self._get_observation_async(), self.loop)
        return future.result() # .result() 会阻塞直到协程完成

    def send_action(self, action_args: np.ndarray) -> np.ndarray:
        """
        同步地向客户端发送动作指令。
        这个方法是线程安全的，可以从外部线程调用。
        """
        if not self.loop or not self.is_connected:
            raise ConnectionError("服务器未运行或机器人未连接。")
            
        future: Future = asyncio.run_coroutine_threadsafe(self._send_action_async(action_args), self.loop)
        return future.result()

    # ------------------ 异步核心实现 ------------------

    async def _get_observation_async(self) -> dict | None:
        if not self.reader:
            raise ConnectionError("StreamReader 不可用。")
            
        print("等待接收来自Stretch机器人的观察数据...")
        before_read_t = time.perf_counter()
        
        observation_data = await recv_msg(self.reader)
        
        if observation_data is None:
            self._is_connected = False
            raise ConnectionError("接收数据失败，连接可能已断开。")

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        print("接收到来自机器人的数据。")
        return observation_data
    
    async def _send_action_async(self, action_args: np.ndarray) -> np.ndarray:
        if not self.writer:
            raise ConnectionError("StreamWriter 不可用。")

        print("准备发送动作指令...")
        await send_msg(self.writer, action_args)
        print("动作指令已发送。")
        return action_args
    
    @cached_property 
    def _cameras_ft(self)  -> dict[str, tuple[int, int, int]]:
        # 相机只包含训练的模型使用的特征，默认navigation相机不包含在内
        return {'head' : (640, 480, 3), 'wrist' : (480, 640, 3)}
    
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
    
    def is_homed(self):
        return True
    
    def home(self):
        # TODO
        pass
    
    def calibrate(self):
        # TODO
        pass
    
    def is_calibrated(self) -> bool:
        # TODO
        return True
    
    def configure(self):
        pass

    def head_look_at_end(self):
        # TODO
        pass 

    def reset_to_home(self):
        pass
    
