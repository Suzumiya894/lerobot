import socket
import json
import time
import struct # 用于处理固定长度的报头
import numpy as np
import matplotlib.pyplot as plt
import asyncio

from lerobot.common.robots.stretch3.mystretch import MyStretchRobot
from lerobot.common.robots.stretch3.configuration_stretch3 import Stretch3RobotConfig

from lerobot.common.utils.remote_utils import recv_msg, send_msg

config = Stretch3RobotConfig()
robot = MyStretchRobot(config)
robot.connect()

robot.api.head.pose('tool')
robot.api.wait_command()

def plot_actions(action_history):
    joint_names = [
        "Lift Position",
        "Arm Position",
        "Wrist Pitch",
        "Wrist Roll",
        "Wrist Yaw",
        "Gripper Position",
        "Base X Position",
        "Base Y Position",
        "Base Theta"
    ]

    # 创建3x3子图画布
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Joint Position History', fontsize=16)

    # 将数据转置为按关节分组的列表
    transposed_data = np.array(action_history).T if action_history else []
    # 遍历每个关节绘制折线图
    for i, ax in enumerate(axes.flat):
        if i < len(joint_names):
            # 提取当前关节的所有历史值
            joint_data = transposed_data[i]
            steps = list(range(len(joint_data)))
            
            # 绘制折线图
            ax.plot(steps, joint_data, 'b-', linewidth=1.5)
            ax.set_title(joint_names[i])
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Position Value')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 设置纵轴范围自适应数据
            y_min, y_max = min(joint_data), max(joint_data)
            y_padding = (y_max - y_min) * 0.1  # 添加10%的空白边距
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        else:
            ax.axis('off')  # 隐藏多余的子图

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    plt.show()
    plt.imsave("action_history_plot.png", fig.canvas.buffer_rgba(), format='png')

async def run_robot(host, port=65432, action_history = []):
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
            
            observation_data.pop("observation.images.navigation", None)
            print(f"捕获环境观测耗时: {time.time() - capture_observation_start_time:.2f}秒")
            print("-" * 30 + f"Step {step}" + "-" * 30)
            step += 1
            # 使用新的函数发送完整的消息
            print(f"发送环境观测。")
            await send_msg(s, observation_data)
            
            # 使用新的函数接收完整的消息
            print("等待服务器发送动作指令...")
            action_params = await recv_msg(s)
            
            if action_params is None:
                print("与服务器的连接已断开。")
                break
            
            action_history.append(action_params)
            print(f"从服务器接收到动作指令: {action_params}")
            
            action_start_time = time.time()
            robot.send_action(action_params)
            print(f"执行动作指令耗时: {time.time() - action_start_time:.2f}秒")

if __name__ == "__main__":
    SERVER_HOST = '10.176.44.2'
    action_history = []
    try:
        asyncio.run(run_robot(SERVER_HOST, port=65432, action_history=action_history))
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
        print("开始绘制动作历史图表...")
        plot_actions(action_history)