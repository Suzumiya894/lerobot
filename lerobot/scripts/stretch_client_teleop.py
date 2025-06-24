import socket
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import asyncio

from lerobot.common.robots.stretch3.mystretch import MyStretchRobot, Stretch3RobotConfig
from lerobot.common.teleoperators.stretch3_gamepad import Stretch3GamePad, Stretch3GamePadConfig

from lerobot.common.utils.remote_utils import recv_msg, send_msg
from lerobot.common.utils.utils import log_say
import argparse


async def my_record(robot, teleop, host, port, fps=30, warmup_time=20, episode_time_s=30, reset_time_s=10, num_episodes=50, play_sounds=True):
    """
    运行机器人客户端，连接服务器并进行通信。
    """
    step = 0
    print(f"args: \n\t{host=}, \n\t{port=}, \n\t{fps=}, \n\t{warmup_time=}, \n\t{episode_time_s=}, \n\t{reset_time_s=}, \n\t{num_episodes=}")

    try:
        reader, writer = await asyncio.open_connection(host, port)
        print(f"成功连接到服务器 {host}:{port}")
    except (socket.gaierror, ConnectionRefusedError) as e:
        print(f"连接服务器失败: {e}")
        return
    
    log_say(f"Warming Up. Waiting for {warmup_time} seconds.", play_sounds)
    await asyncio.sleep(warmup_time)  # 预热时间

    try:
        for recorded_episodes in range(num_episodes):
            log_say(f"Recording episode {recorded_episodes}", play_sounds)

            robot.reset_base_odometry() # 对于Stretch机器人，在每次采数据前确保底盘位置信息重置为0。
            timestamp = 0
            start_episode_time = time.perf_counter()
            while timestamp < episode_time_s:
                start_loop_t = time.perf_counter()

                capture_observation_start_time = time.time()
                observation_data = robot.get_observation()
                teleop.send_action(teleop.get_action()) # 发送控制器的动作指令
                action = robot.get_action()  # 获取机器人的动作指令
                
                print(f"捕获环境观测耗时: {time.time() - capture_observation_start_time:.2f}秒")
                print("-" * 30 + f"Step {step}" + "-" * 30)
                step += 1

                await send_msg(writer, {"observation" : observation_data, "action" : action})
                dt_s = time.perf_counter() - start_loop_t
                print(f"已发送环境观测。dt_s: {dt_s:.4f}秒")
                await asyncio.sleep(1 / fps - dt_s)

                timestamp = time.perf_counter() - start_episode_time
            
            await send_msg(writer, {"episode_finished": True})
            log_say("Reset the environment.", play_sounds)
            reset_start_time = time.perf_counter()
            teleop.safety_stop()
            robot.reset_to_home()
            robot.head_look_at_end()
            left_reset_time = max(time.perf_counter() - reset_start_time - reset_time_s, 0)
            await asyncio.sleep(left_reset_time)
            
    except (ConnectionResetError, asyncio.IncompleteReadError) as e:
        print(f"与服务器的连接发生错误: {e}")
    finally:
        print("正在关闭连接...")
        writer.close()
        await writer.wait_closed()

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='stretch机器人端采数据脚本')
    parser.add_argument('--port', 
                        type=int, 
                        default=65432, 
                        help='服务器端口号，默认为65432')
    parser.add_argument('--fps', 
                        type=int, 
                        default=30, 
                        help='采集数据的帧率，默认为30 FPS')
    parser.add_argument('--server_host',
                        '--server-host', 
                        type=str, 
                        default='10.176.44.2', 
                        help='服务器IP地址')
    parser.add_argument('--warmup_time', 
                        '--warmup-time',
                        type=int, 
                        default=20, 
                        help='预热时间，单位为秒，默认为20秒')
    parser.add_argument('--episode_time_s', 
                        type=int, 
                        default=30, 
                        help='每个episode的持续时间，单位为秒，默认为30秒')
    parser.add_argument('--reset_time_s', 
                        type=int, 
                        default=10, 
                        help='每个episode之间的重置时间，单位为秒，默认为10秒')
    parser.add_argument('--num_episodes', 
                        type=int, 
                        default=50, 
                        help='数据集包含的episode数量，默认为50个')
    parser.add_argument('--play_sounds', 
                        type=bool, 
                        default=True, 
                        help='是否播放声音提示，默认为True')

    args = parser.parse_args()

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
        asyncio.run(my_record(robot, teleop, **vars(args)))
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
