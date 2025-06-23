import socket
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import asyncio

from lerobot.common.robots.stretch3.mystretch import MyStretchRobot, Stretch3RobotConfig
from lerobot.common.teleoperators.stretch3_gamepad import Stretch3GamePad, Stretch3GamePadConfig

from lerobot.common.utils.remote_utils import recv_msg, send_msg

async def run_robot(robot, teleop, host, port=65432, fps=30):
    """
    运行机器人客户端，连接服务器并进行通信。
    """
    step = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            print(f"成功连接到服务器 {host}:{port}")
        except socket.error as e:
            print(f"连接服务器失败: {e}")
            return

        while True:
            capture_observation_start_time = time.time()
            observation_data = robot.get_observation()
            teleop.send_action(teleop.get_action()) # 发送控制器的动作指令
            action = robot.get_action()  # 获取机器人的动作指令
            
            print(f"捕获环境观测耗时: {time.time() - capture_observation_start_time:.2f}秒")
            print("-" * 30 + f"Step {step}" + "-" * 30)
            step += 1
            # 使用新的函数发送完整的消息
            print(f"发送环境观测。")
            await send_msg(s, {"observation" : observation_data, "action" : action})
            
            
            time.sleep(1 / fps)

if __name__ == "__main__":
    FPS = 30
    SERVER_HOST = '10.176.44.2'

    config = Stretch3RobotConfig()
    robot = MyStretchRobot(config)
    robot.connect()

    robot.api.head.pose('tool')
    robot.api.wait_command()

    teleop_config = Stretch3GamePadConfig()
    teleop = Stretch3GamePad(teleop_config)
    teleop.set_robot(robot.api)
    teleop.connect()
    
    try:
        asyncio.run(run_robot(robot, teleop, host=SERVER_HOST, port=65432, fps=FPS))
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
