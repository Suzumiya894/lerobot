## 

This repository contains the implementation of Stretch Robot for Lerobot.

## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

When using `miniconda`, install `ffmpeg` in your environment:
```bash
conda install ffmpeg -c conda-forge
```

> **NOTE:** This usually installs `ffmpeg 7.X` for your platform compiled with the `libsvtav1` encoder. If `libsvtav1` is not supported (check supported encoders with `ffmpeg -encoders`), you can:
>  - _[On any platform]_ Explicitly install `ffmpeg 7.X` using:
>  ```bash
>  conda install ffmpeg=7.1.1 -c conda-forge
>  ```
>  - _[On Linux only]_ Install [ffmpeg build dependencies](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) and [compile ffmpeg from source with libsvtav1](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), and make sure you use the corresponding ffmpeg binary to your install with `which ffmpeg`.

Install 🤗 LeRobot:
```bash
pip install -e ".[dev, pi0, smolvla, stretch]"
```

## Run Commands

Before you run any commands, check the local configuration file `./local_config.json`. If it does not exist, you can copy this:

```json
{
    "Stretch3RobotConfig" : {
        "is_remote_server" : true,
        "server_port" : 65432,
        "control_mode" : "vel",
        "control_action_use_head" : false,
        "control_action_base_only_x" : true
    },
    "Stretch3GamePadConfig" : {
        "mock" : true,
        "control_action_base_only_x" : true
    }
}
```

This configuration will override the default configuration for the robot and gamepad. Here is a simple description of the parameters:
- "Stretch3RobotConfig":
    - `is_remote_server`: Set to `true` if the robot is running on a remote server.
    - `server_port`: The port to connect to the robot.
    - `control_mode`: The control mode for the robot,  "vel" for velocity control, and "pos" for position control.
    - `control_action_use_head`: If set to `true`, Action will contain two joints for the head: head_pan and head_tilt.
    - `control_action_base_only_x`: If set to `true`, only the x-axis of the base will be controlled, in other words, the robot will only move in the x-direction and will not rotate.
- "Stretch3GamePadConfig":
    - `mock`: Set to `true` if this repository is running on a remote server.
    - `control_action_base_only_x`: If set to `true`, only the x-axis of the base will be controlled, in other words, the robot will only move in the x-direction and will not rotate.

### Record a dataset

Before you record a dataset, check the local configuration file and make sure 'Stretch3RobotConfig' - 'control_mode' is set to 'vel'. This is because the configuration will also affect the dataset features.

The default dataset features are:
- observation:
    - observation.state: the state of Stretch3 robot joints, which is a 11-dimensional vector: `["head_pan.pos", "head_tilt.pos", "lift.pos", "arm.pos", "wrist_pitch.pos", "wrist_roll.pos", "wrist_yaw.pos", "gripper.pos", "base_x.pos", "base_y.pos", "base_theta.pos", ]`
    - observation.image.head: the image from the head camera, which is a 640x480 RGB image.
    - observation.image.wrist: the image from the wrist camera, which is a 480x640 RGB image.
    - observation.image.navigation: the image from the navigation camera, which is a 1280x720 RGB image.
- action:
    - action: the velocity of Stretch3 robot joints, which is a 11-dimensional vector: `["head_pan.vel", "head_tilt.vel", "lift.vel", "arm.vel", "wrist_pitch.vel", "wrist_roll.vel", "wrist_yaw.vel", "gripper.vel", "base_x.vel", "base_y.vel", "base_theta.vel", ]`

Since this type of dataset contains the most information, you have no need to modify it before recording. You can easily change it after recording, using some offered scripts, see next section.

To record a dataset, run this command on the robot:
```bash
python -m lerobot.record     \
    --robot.type=stretch3     \
    --dataset.repo_id=${HF_USER}/dataset_name     \
    --dataset.num_episodes=50     \
    --dataset.single_task="Grab a plastic bottle and put it in the box."     \
    --teleop.type=stretch3     \
    --dataset.video=true     \
    --dataset.root=./data/${HF_USER}/dataset_name     \
    --dataset.reset_time_s=5     \
    --dataset.episode_time_s=30     \
    --dataset.push_to_hub=false     \
    --dataset.fps=30     \
    --dataset.private=true
```

modify the parameters if necessary.

### Post Processing dataset

Two scripts are provided to post-process the dataset after recording:
- `lerobot.scripts.process_dataset_angle.py`: This script will transform the angle in the dataset(range from 0 to 2*pi) to the range from -pi to pi, which is more suitable for training.
- `lerobot.scripts.stretch_dataset_transfer.py`: This script offer two methods to omit some dimensions in the dataset.
    - `process_dataset_keep_vel(dataset_path: str, to_pop_names:list[str] = ["head_pan.pos", "head_tilt.pos"])`: This method will only omit the target dimensions from the dataset.
    - `process_dataset_vel_to_pos(dataset_path: str, to_pop_names:list[str] = ["head_pan.pos", "head_tilt.pos"])`: This method will transform the form of action from velocity to joints position, and omit the target dimensions from the dataset.

see `lerobot.scripts.stretch_dataset_transfer.py` for more details.

### Training

To use wandb for logging training and evaluation curves, make sure you've run `wandb login` as a one-time setup step. Then, when running the training command above, enable WandB in the configuration by adding `--wandb.enable=true`.

A link to the wandb logs for the run will also show up in yellow in your terminal. Here is an example of what they look like in your browser. Please also check [here](./examples/4_train_policy_with_script.md#typical-logs-and-metrics) for the explanation of some commonly used metrics in logs.

#### train pi0

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=${HF_USER}/pi0_dataset \
    --policy.type=pi0 \
    --output_dir=outputs/train/pi0_dataset \
    --job_name=pi0_dataset \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --steps=100000 \
    --save_freq=25000
```

#### train smolvla

```bash
python lerobot/scripts/train.py \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=${HF_USER}/smolvla_dataset \
    --batch_size=32 \
    --steps=60000 \
    --save_freq=10000 \
    --output_dir=outputs/train/smolvla_dataset \
    --job_name=smolvla_dataset \
    --policy.device=cuda \
    --wandb.enable=true
```

### Inference

To run inference with the trained policy, you should first run this command on the server:

```bash
python -m lerobot.record \
    --robot.type=stretch3 \
    --dataset.single_task="Grab a plastic bottle and put it in the box." \
    --dataset.repo_id=${HF_USER}/eval_smolvla_dataset \
    --dataset.episode_time_s=400 \
    --dataset.num_episodes=1 \
    --policy.path=/data/yew/lerobot/outputs/train/smolvla_dataset/checkpoints/060000/pretrained_model  \
    --play_sounds=false \
    --dataset.push_to_hub=false \
    --policy.device=cuda
```

Make sure the task is alined with the description in the dataset, and the repo_id is started with 'eval_'. Modify the '--policy.path' to the path of the trained policy you want to use.

Then, run this command on the robot:

```bash
python lerobot/scripts/stretch_client_control.py
```

Modify the SERVER_HOST and port in the script to the address of the server, which is running the inference command above.

### Other useful commands

#### visualize a local dataset

```bash
python lerobot/scripts/visualize_dataset_html.py   \
    --repo-id ${HF_USER}/smolvla_dataset \
    --root /home/fdse/yew/huggingface/lerobot/${HF_USER}/smolvla_dataset \
    --port 8421
```

## Dataset

A dataset of 50 episodes is available at[huggingface](https://huggingface.co/datasets/Suzumiya894/stretch_smolvla_dataset). The task is to "Grab a plastic bottle and put it in the box.", and it contains 5 different start point for the bottle.The dataset is recorded with "control_action_use_head" set to 'False', and "control_action_base_only_x" set to 'True', then use the post processing script to omit the head joints, base_y and base_theta dimensions.