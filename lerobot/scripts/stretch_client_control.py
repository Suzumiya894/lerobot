import socket
import json
import time
import struct # 用于处理固定长度的报头
import numpy as np
import torch

from lerobot.common.robot_devices.robots.stretch import StretchRobot

robot = StretchRobot()
robot.connect()

robot.head.pose('tool')
robot.wait_command()


def send_msg(sock, msg_dict):
    """
    为消息添加固定长度的报头，然后发送。
    """
    # 将消息字典编码为字节
    msg_bytes = json.dumps(msg_dict).encode('utf-8')
    # 计算消息长度，并打包成一个4字节的整数
    msg_len_header = struct.pack('!I', len(msg_bytes))
    # 发送报头
    sock.sendall(msg_len_header)
    # 发送实际消息
    sock.sendall(msg_bytes)

def recv_msg(sock):
    """
    接收固定长度的报头以确定消息大小，然后接收完整的消息。
    """
    # 首先接收4字节的报头
    raw_msg_len = recv_all(sock, 4)
    if not raw_msg_len:
        return None
    
    # 解包报头以获取消息长度
    msg_len = struct.unpack('!I', raw_msg_len)[0]
    
    # 根据获取的长度接收完整的消息
    return recv_all(sock, msg_len)

def recv_all(sock, n):
    """
    一个辅助函数，确保从套接字接收到n个字节的数据。
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def run_robot(host, port=65432):
    """
    运行机器人客户端，连接服务器并进行通信。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            print(f"成功连接到服务器 {host}:{port}")
        except socket.error as e:
            print(f"连接服务器失败: {e}")
            return

        while True:
            observation_data = robot.capture_observation()
            fixed_observation_data = {}
            for key, value in observation_data.items():
                if key == "observation.images.navigation":
                    continue
                elif key == "observation.state":
                    fixed_observation_data[key] = value.tolist()[2:]
                elif isinstance(value, torch.Tensor):
                    fixed_observation_data[key] = value.tolist()
            
            # 使用新的函数发送完整的消息
            print(f"发送环境观测。Observation.state: {fixed_observation_data['observation.state']}")
            print(f"Observation keys: {list(fixed_observation_data.keys())}")
            send_msg(s, fixed_observation_data)
            
            # 使用新的函数接收完整的消息
            print("等待服务器发送动作指令...")
            response_bytes = recv_msg(s)
            
            if not response_bytes:
                print("与服务器的连接已断开。")
                break
                
            action_params = json.loads(response_bytes.decode('utf-8'))
            assert isinstance(action_params, list), "接收到的动作参数应为列表类型"
            action_params = torch.tensor(action_params, dtype=torch.float32)
            print(f"从服务器接收到动作指令: {action_params}")
            
            robot.send_pos_action(action_params)

if __name__ == "__main__":
    SERVER_HOST = '10.176.44.2'

    try:
        run_robot(SERVER_HOST, port=65432)
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        robot.disconnect()