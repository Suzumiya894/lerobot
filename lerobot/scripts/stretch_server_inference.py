"""
run command:

CUDA_VISIBLE_DEVICES=0 python -m lerobot.scripts.stretch_server_inference \
    --robot.type=stretch3 \
    --dataset.video=true \
    --dataset.repo_id="None" \
    --dataset.single_task="None" \
    --policy.path=./outputs/train/box_of_water_long_horizon_fixed/checkpoints/080000/pretrained_model  \
    --policy.device=cuda 
"""

import time

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.record import DatasetRecordConfig, RecordConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.robots import make_robot_from_config
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.remote_utils import recv_msg, send_msg
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.utils import get_safe_torch_device


class LeRobotDatasetMetadataMock():
    """
    make_policy() requires a LeRobotDatasetMetadata object.
    This mock object is used to pass the dataset features to make_policy().
    """
    def __init__(self, dataset_features):
        self.features = dataset_features
        self.stats = {}

@parser.wrap()
def inference(cfg: RecordConfig):
    robot = make_robot_from_config(cfg.robot)

    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    dataset_meta = LeRobotDatasetMetadataMock(dataset_features)

    policy = make_policy(cfg.policy, ds_meta=dataset_meta)
    single_task = cfg.dataset.single_task
    # 服务器端网络通信的api被封装在了StretchRobotServer中，通过robot.get_observation()接受数据，robot.send_action(action)发送数据。

    robot.connect()

    while True:
        try:
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset_features, observation, prefix="observation")

            predicted_actions = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
                predict_actions=True,
            )   # return shape: (n_action_steps, action_dim)

            robot.send_action(predicted_actions)
        except ConnectionError as e:
            print(f"主程序：检测到连接断开 ({e})。准备接受新连接...")
            time.sleep(5)
        except RuntimeError  as e:
            print(f"主程序：发生运行时错误 ({e})。")
            break
        except KeyboardInterrupt:
            print("主程序：已接收到键盘中断。正在关闭...")
            break
    
    robot.disconnect()

if __name__ == "__main__":
    inference()