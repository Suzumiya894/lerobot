
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.record import DatasetRecordConfig, RecordConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robots import make_robot_from_config
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.remote_utils import recv_msg, send_msg
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.utils import get_safe_torch_device


@parser.wrap()
def inference(cfg: RecordConfig):
    robot = make_robot_from_config(cfg.robot)

    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
    )

    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    single_task = cfg.dataset.single_task
    # 服务器端网络通信的api被封装在了StretchRobotServer中，通过robot.get_observation()接受数据，robot.send_action(action)发送数据。

    robot.connect()

    try:
        while True:
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

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

            # TODO: add recovery window
    except Exception as e:
        print(f"发生异常: {e}")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    inference()