import socket
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import asyncio

from lerobot.common.robots.stretch3.mystretch import MyStretchRobot
from lerobot.common.robots.stretch3.configuration_stretch3 import Stretch3RobotConfig

from lerobot.common.utils.remote_utils import recv_msg, send_msg

async def run_robot(robot, host, port):
    step = 0

    try:
        reader, writer = await asyncio.open_connection(host, port)
        print(f"成功连接到服务器 {host}:{port}")
    except (socket.gaierror, ConnectionRefusedError) as e:
        print(f"连接服务器失败: {e}")
        return
    
    while True:
        capture_observation_start_time = time.time()
        observation_data = robot.get_observation()
        
        observation_data.pop("observation.images.navigation", None)
        print(f"捕获环境观测耗时: {time.time() - capture_observation_start_time:.2f}秒")
        print("-" * 30 + f"Step {step}" + "-" * 30)
        step += 1
        # 使用新的函数发送完整的消息
        print(f"发送环境观测。")
        await send_msg(writer, observation_data)
        
        # 使用新的函数接收完整的消息
        print("等待服务器发送动作指令...")
        action_params = await recv_msg(reader)

        if action_params is None:
            print("与服务器的连接已断开。")
            break
        
        print(f"从服务器接收到动作指令: {action_params}")
        
        action_start_time = time.time()
        robot.send_action(action_params)
        print(f"执行动作指令耗时: {time.time() - action_start_time:.2f}秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='stretch机器人端控制脚本')
    parser.add_argument('--port', 
                        type=int, 
                        default=65432, 
                        help='服务器端口号，默认为65432')
    parser.add_argument('--host', 
                        type=str, 
                        default='10.176.44.2', 
                        help='服务器IP地址')
    
    args = parser.parse_args()

    config = Stretch3RobotConfig()
    robot = MyStretchRobot(config)
    robot.connect()

    robot.api.head.pose('tool')
    robot.api.wait_command()
    try:
        asyncio.run(run_robot(robot, **vars(args)))
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
        print("开始绘制动作历史图表...")
