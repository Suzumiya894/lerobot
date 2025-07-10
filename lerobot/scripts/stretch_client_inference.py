"""
Run command:
python lerobot/scripts/stretch_client_inference.py --single_task="Take the water bottle from the bottom-left of the box and place it inside the top tier of the cabinet." --repo_id="Suzumiya894/eval_BOF_LH_long_description" --root="./data/Suzumiya894/eval_BOF_LH_long_description" --episode_time_s=1000
"""
import socket
import asyncio
from collections import deque
import time
import argparse

from lerobot.common.robots.stretch3.mystretch import MyStretchRobot, Stretch3RobotConfig
from lerobot.common.teleoperators.stretch3_gamepad import Stretch3GamePad, Stretch3GamePadConfig

from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.remote_utils import recv_msg, send_msg
from lerobot.common.utils.utils import log_say
from lerobot.common.utils.robot_utils import busy_wait

@safe_stop_image_writer
async def my_record(robot: MyStretchRobot, dataset: LeRobotDataset, host: str, port: int, fps=30, warmup_time=20, episode_time_s=360, reset_time_s=10, num_episodes=1, play_sounds=True, single_task=None):
    print(f"args: \n\t{host=}, \n\t{port=}, \n\t{fps=}, \n\t{warmup_time=}, \n\t{episode_time_s=}, \n\t{reset_time_s=}, \n\t{num_episodes=}")

    try:
        reader, writer = await asyncio.open_connection(host, port)
        print(f"成功连接到服务器 {host}:{port}")
    except (socket.gaierror, ConnectionRefusedError) as e:
        print(f"连接服务器失败: {e}")
        return

    steps, cur_step = 0, 0
    window_len = 1
    todo_steps = deque(maxlen=window_len)

    timestamp = 0
    start_episode_t = time.perf_counter()

    try:
    
        # TODO: add record loop

        while timestamp < episode_time_s:
            start_loop_t = time.perf_counter()
            # TODO: add keyboard listener
            # TODO: add recovery window

            obs_start_t = time.perf_counter()
            observation = robot.get_observation()
            print(f"捕获环境观测耗时: {time.perf_counter() - obs_start_t:.2f}秒")
            if len(todo_steps) > 0:
                action_values = todo_steps.popleft()
            else:
                print("Todo steps is empty, waiting for server to send actions...")
                await send_msg(writer, observation)
                predicted_actions = await recv_msg(reader)
                todo_steps.extend(predicted_actions)
                if len(todo_steps) == 0:
                    print("与服务器的连接已断开。")
                    break
                action_values = todo_steps.popleft()
                
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}

            print(f"从服务器接收到动作指令: {action}")
            action_start_t = time.perf_counter()
            sent_action = robot.send_action(action)
            print(f"执行动作指令耗时: {time.perf_counter() - action_start_t:.2f}秒")

            dataset_start_t = time.perf_counter()
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)
            print(f"构建数据集帧耗时: {time.perf_counter() - dataset_start_t:.2f}秒")

            dt_s = time.perf_counter() - start_loop_t
            print(f"总计用时: {dt_s:.2f}秒, 采集帧率: {1 / dt_s:.2f} FPS")
            busy_wait(1 / fps - dt_s)

            timestamp = time.perf_counter() - start_episode_t

        dataset.save_episode()

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
    parser.add_argument('--host', 
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
                        default=1, 
                        help='数据集包含的episode数量，默认为1个')
    parser.add_argument('--play_sounds', 
                        type=bool, 
                        default=True, 
                        help='是否播放声音提示，默认为True')
    parser.add_argument('--repo_id', 
                        type=str, 
                        required=True,
                        help='数据集仓库名（必填）')
    parser.add_argument('--root', 
                        type=str, 
                        default=None,
                        help='数据集仓库本地路径，如果未指定则从huggingface下载')
    parser.add_argument('--single_task', 
                        type=str, 
                        required=True,
                        help='任务自然语言描述（必填）')

    args = parser.parse_args()

    config = Stretch3RobotConfig()
    robot = MyStretchRobot(config)
    robot.connect()

    robot.api.head.pose('tool')
    robot.api.wait_command()

    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        args.repo_id,
        args.fps,
        root=args.root,
        robot_type="stretch3",
        features=dataset_features,
        use_videos=True,
    )

    # teleop_config = Stretch3GamePadConfig()
    # teleop = Stretch3GamePad(teleop_config)
    # teleop.set_robot(robot.api)
    # teleop.connect()
    
    robot.calibrate()

    try:
        # Prepare arguments for my_record, excluding repo_id and root
        record_args = {
            "host": args.host,
            "port": args.port,
            "fps": args.fps,
            "warmup_time": args.warmup_time,
            "episode_time_s": args.episode_time_s,
            "reset_time_s": args.reset_time_s,
            "num_episodes": args.num_episodes,
            "play_sounds": args.play_sounds,
            "single_task": args.single_task,
        }
        asyncio.run(my_record(robot, dataset, **record_args))
    except Exception as e:
        dataset.save_episode()
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()
    