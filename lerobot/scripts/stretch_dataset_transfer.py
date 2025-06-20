"""
该脚本用于后处理STRETCH采集的数据集，原始数据集包括三个相机navigation, head, wrist，包含11个自由度，observation.state为关节的绝对位置，action为关节的速度。
一般训练时使用只使用head和wrist的图像数据，并且也用不到全部11维自由度，因此需要后处理。（navigation图像数据需要手动删除）
该脚本可以删除指定的列（例如head_pan.pos, head_tilt.pos等），并且提供两种处理action的方式：
1. 将action从速度（vel）转换为位置（pos），即将下一帧的observation.state作为当前帧的action：process_dataset_vel_to_pos()
2. 保留action的速度信息（vel）：process_dataset_keep_vel()
"""
import os
import json
import pandas as pd

FEATURES = ['head_pan.pos', 'head_tilt.pos', 'lift.pos', 'arm.pos',
            'wrist_pitch.pos', 'wrist_roll.pos', 'wrist_yaw.pos', 
            'gripper.pos', 'base_x.pos', 'base_y.pos', 'base_theta.pos']
FEATURE_IDX_NAMES = {name: idx for idx, name in enumerate(FEATURES)}
def process_dataset_vel_to_pos(dataset_path: str, to_pop_names:list[str] = ["head_pan.pos", "head_tilt.pos"]) -> None:
    """
    一个用于修改Lerobot数据集的函数。该函数会删除to_pop_names中指定的列名，并且将原本数据时vel的action改为pos的action。

    该函数会执行以下操作：
    2. 处理episode_stats.jsonl文件，删除指定的列名，并修改action的统计数据。
    3. 迭代处理data/chunk-000/中的所有Parquet文件。

    Args:
        dataset_path (str): Lerobot数据集的根目录路径。
    """
    print(f"正在处理数据集: {os.path.abspath(dataset_path)}\n")

    
    to_pop_idxs = [FEATURE_IDX_NAMES[name] for name in to_pop_names]
    new_len = len(FEATURES) - len(to_pop_names) 

    # --- 1. 删除 `observation.images.navigation` 目录 ---
    # 手动删除即可

    # --- 2. 读取并显示 meta 文件夹中元数据的结构 ---
    meta_path = os.path.join(dataset_path, 'meta')
    print("\n--- 正在读取元数据 (meta) ---")

    # 首先处理 info.json
    try:
        info_json_path = os.path.join(meta_path, 'info.json')
        with open(info_json_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
            print("\n📄 `info.json` 的结构:")
            print(info_data)
            for key, value in info_data.items():
                print(f"  {key}: {value}")
                print("-" * 40)

            info_data['total_frames'] # ？这个需要修改吗
            info_data['total_videos'] = info_data['total_episodes'] * 2 # 两个摄像头，数量乘2
            info_data['features']['action']['shape'] = [new_len]
            info_data['features']['observation.state']['shape'] = [new_len]
            # 原本的'names': ['head_pan.vel', 'head_tilt.vel', 'lift.vel', 'arm.vel', 'wrist_pitch.vel', 'wrist_roll.vel', 'wrist_yaw.vel', 'gripper.vel', 'base_x.vel', 'base_y.vel', 'base_theta.vel']

            kept_features = [name for name in FEATURE_IDX_NAMES if name not in to_pop_names]
            info_data['features']['observation.state']['names'] = kept_features
            info_data['features']['action']['names'] = [f"{name.replace('.pos', '.next_pos')}" for name in kept_features]

            # info_data['features']['action']['names'] = ['lift.next_pos', 'arm.next_pos', 'wrist_pitch.next_pos', 'wrist_roll.next_pos', 'wrist_yaw.next_pos', 'gripper.next_pos', 'base_x.next_pos', 'base_y.next_pos', 'base_theta.next_pos']
            # info_data['features']['observation.state']['names'] = ['lift.pos', 'arm.pos', 'wrist_pitch.pos', 'wrist_roll.pos', 'wrist_yaw.pos', 'gripper.pos', 'base_x.pos', 'base_y.pos', 'base_theta.pos']

            info_data['features'].pop("observation.images.navigation", None)
        
        with open(info_json_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=4)
    
    except FileNotFoundError:
        print(f"⚠️ 未找到 `info.json`。")


    # 接着处理 .jsonl 文件 (episodes.jsonl, episodes_stats.jsonl, tasks.jsonl)
    # for filename in ['episodes.jsonl', 'episodes_stats.jsonl', 'tasks.jsonl']:
    filename = 'episodes_stats.jsonl'   # 只需要处理episodes_stats.jsonl
    modified_episodes_stats = []
    jsonl_path = os.path.join(meta_path, filename)
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            first_line = True
            for line in f:
                data = json.loads(line)
                if first_line:
                    print(f"\n📄 `{filename}` 的第一行数据:")
                    print(data)
                    first_line = False

                for key, value in data['stats']['observation.state'].items():
                    if key == 'count':
                        continue
                    if len(value) != len(FEATURES):
                        # 避免重复执行该脚本出现错误
                        continue

                    data['stats']['observation.state'][key] = [value[i] for i in range(len(value)) if i not in to_pop_idxs]
                    # data['stats']['observation.state'][key] = value[2:] # 删除前两个元素
                
                data['stats']['action'] = data['stats']['observation.state']  # 修改 action 的统计数据
                data['stats'].pop("observation.images.navigation", None)  # 删除 navigation 图像的统计数据

                modified_episodes_stats.append(data)
            
        # 将修改后的数据写回到同一个文件
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in modified_episodes_stats:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
    except FileNotFoundError:
        print(f"⚠️ 未找到 `{filename}`。")

    # --- 3. 迭代处理 data/chunk-000/ 中的所有 Parquet 文件 ---
    data_chunk_path = os.path.join(dataset_path, 'data', 'chunk-000')
    print(f"\n--- 正在迭代处理 Parquet 文件于: {data_chunk_path} ---")

    if not os.path.exists(data_chunk_path):
        print(f"❌ 错误: 数据目录不存在 {data_chunk_path}")
        return

    # 获取所有.parquet文件的列表
    parquet_files = [f for f in os.listdir(data_chunk_path) if f.endswith('.parquet')]
    parquet_files.sort()

    for i, filename in enumerate(parquet_files):
        file_path = os.path.join(data_chunk_path, filename)
        print(f"  ({i+1}/{len(parquet_files)}) 正在处理: {filename}")

        # 将 Parquet 文件读入 pandas DataFrame
        df = pd.read_parquet(file_path)

        # ===============================================================
        # 1. 修改 'observation.state' 列
        #    对该列的每一行应用一个lambda函数，裁掉列表的前两个元素
        #    我们先将结果存储在一个新的Series中，因为它在下一步中需要被使用
        if len(df['observation.state'][0]) != len(FEATURES):
            print(f"⚠️ 跳过文件 {filename}，因为 'observation.state' 列的长度不是 {len(FEATURES)}。")
            continue
        modified_state = df['observation.state'].apply(lambda state_list: [state_list[i] for i in range(len(state_list)) if i not in to_pop_idxs])
        # modified_state = df['observation.state'].apply(lambda state_list: state_list[2:])

        # 2. 修改 'action' 列
        #    核心思想是使用 pandas 的 shift(-1) 方法，它能将列的数据向上移动一行
        #    这样第 idx 行就能取到原始的第 idx+1 行的数据
        new_actions = modified_state.shift(-1)

        #    处理最后一行：shift(-1) 会在最后一行产生一个空值 (NaN)
        #    根据你的规则，最后一行action应被赋予同行的observation.state
        #    我们用 modified_state 的最后一行值来填充这个空值
        last_row_index = df.index[-1]
        last_state = modified_state.loc[last_row_index]
        
        # 当Series中包含列表等对象时，直接用fillna可能行为不一致
        # 更稳妥的方式是直接定位并赋值
        # 将new_actions转换为object类型以接受列表赋值
        new_actions = new_actions.astype(object)
        new_actions.loc[last_row_index] = last_state

        # 3. 将修改后的列赋值回 DataFrame
        df['observation.state'] = modified_state
        df['action'] = new_actions
        
        print(f"    -> 列 'observation.state' 和 'action' 已成功修改。")
        # ===============================================================

        # 将修改后的 DataFrame 写回原来的 Parquet 文件，覆盖旧文件
        df.to_parquet(file_path)
        

    print("\n✅ 所有 Parquet 文件处理完毕！")


def process_dataset_keep_vel(dataset_path: str, to_pop_names:list[str] = ["head_pan.pos", "head_tilt.pos"]):
    """
    action保留为速度信息，只将head_pan， head_tilt删除
    """
    print(f"正在处理数据集: {os.path.abspath(dataset_path)}\n")


    to_pop_idxs = [FEATURE_IDX_NAMES[name] for name in to_pop_names]
    new_len = len(FEATURES) - len(to_pop_names) 

    # --- 1. 删除 `observation.images.navigation` 目录 ---
    # 手动删除即可

    # --- 2. 读取并显示 meta 文件夹中元数据的结构 ---
    meta_path = os.path.join(dataset_path, 'meta')
    print("\n--- 正在读取元数据 (meta) ---")

    # 首先处理 info.json
    try:
        info_json_path = os.path.join(meta_path, 'info.json')
        with open(info_json_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
            print("\n📄 `info.json` 的结构:")
            print(info_data)
            for key, value in info_data.items():
                print(f"  {key}: {value}")
                print("-" * 40)

            info_data['total_frames'] # ？这个需要修改吗
            info_data['total_videos'] = info_data['total_episodes'] * 2 # 两个摄像头，数量乘2
            info_data['features']['action']['shape'] = [new_len]
            info_data['features']['observation.state']['shape'] = [new_len]
            # 原本的'names': ['head_pan.vel', 'head_tilt.vel', 'lift.vel', 'arm.vel', 'wrist_pitch.vel', 'wrist_roll.vel', 'wrist_yaw.vel', 'gripper.vel', 'base_x.vel', 'base_y.vel', 'base_theta.vel']

            kept_features = [name for name in FEATURE_IDX_NAMES if name not in to_pop_names]
            info_data['features']['observation.state']['names'] = kept_features
            info_data['features']['action']['names'] = [f"{name.replace('.pos', '.vel')}" for name in kept_features]

            # info_data['features']['action']['names'] = ['lift.next_pos', 'arm.next_pos', 'wrist_pitch.next_pos', 'wrist_roll.next_pos', 'wrist_yaw.next_pos', 'gripper.next_pos', 'base_x.next_pos', 'base_y.next_pos', 'base_theta.next_pos']
            # info_data['features']['observation.state']['names'] = ['lift.pos', 'arm.pos', 'wrist_pitch.pos', 'wrist_roll.pos', 'wrist_yaw.pos', 'gripper.pos', 'base_x.pos', 'base_y.pos', 'base_theta.pos']

            info_data['features'].pop("observation.images.navigation", None)
        
        with open(info_json_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=4)
    
    except FileNotFoundError:
        print(f"⚠️ 未找到 `info.json`。")


    # 接着处理 .jsonl 文件 (episodes.jsonl, episodes_stats.jsonl, tasks.jsonl)
    # for filename in ['episodes.jsonl', 'episodes_stats.jsonl', 'tasks.jsonl']:
    filename = 'episodes_stats.jsonl'   # 只需要处理episodes_stats.jsonl
    modified_episodes_stats = []
    jsonl_path = os.path.join(meta_path, filename)
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            first_line = True
            for line in f:
                data = json.loads(line)
                if first_line:
                    print(f"\n📄 `{filename}` 的第一行数据:")
                    print(data)
                    first_line = False

                for tp in ['observation.state', 'action']:
                    for key, value in data['stats'][tp].items():
                        if key == 'count':
                            continue
                        if len(value) != len(FEATURES):
                            # 避免重复执行该脚本出现错误
                            continue

                        data['stats'][tp][key] = [value[i] for i in range(len(value)) if i not in to_pop_idxs]
                
                data['stats'].pop("observation.images.navigation", None)  # 删除 navigation 图像的统计数据

                modified_episodes_stats.append(data)
            
        # 将修改后的数据写回到同一个文件
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in modified_episodes_stats:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
    except FileNotFoundError:
        print(f"⚠️ 未找到 `{filename}`。")

    # --- 3. 迭代处理 data/chunk-000/ 中的所有 Parquet 文件 ---
    data_chunk_path = os.path.join(dataset_path, 'data', 'chunk-000')
    print(f"\n--- 正在迭代处理 Parquet 文件于: {data_chunk_path} ---")

    if not os.path.exists(data_chunk_path):
        print(f"❌ 错误: 数据目录不存在 {data_chunk_path}")
        return

    # 获取所有.parquet文件的列表
    parquet_files = [f for f in os.listdir(data_chunk_path) if f.endswith('.parquet')]
    parquet_files.sort()
    for i, filename in enumerate(parquet_files):
        file_path = os.path.join(data_chunk_path, filename)
        print(f"  ({i+1}/{len(parquet_files)}) 正在处理: {filename}")

        # 将 Parquet 文件读入 pandas DataFrame
        df = pd.read_parquet(file_path)

        # ===============================================================
        # 1. 修改 'observation.state' 列
        #    对该列的每一行应用一个lambda函数，裁掉列表的前两个元素
        #    我们先将结果存储在一个新的Series中，因为它在下一步中需要被使用
        if len(df['observation.state'][0]) != len(FEATURES):
            print(f"⚠️ 跳过文件 {filename}，因为 'observation.state' 列的长度不是 11。")
            continue
        modified_state = df['observation.state'].apply(lambda state_list: [state_list[i] for i in range(len(state_list)) if i not in to_pop_idxs])
        modified_action = df['action'].apply(lambda action_list: [action_list[i] for i in range(len(action_list)) if i not in to_pop_idxs])

        df['observation.state'] = modified_state
        df['action'] = modified_action
        
        print(f"    -> 列 'observation.state' 和 'action' 已成功修改。")
        # ===============================================================

        # 将修改后的 DataFrame 写回原来的 Parquet 文件，覆盖旧文件
        df.to_parquet(file_path)
        

    print("\n✅ 所有 Parquet 文件处理完毕！")
    


if __name__ == "__main__":
    YOUR_DATASET_ROOT = "/data/yew/huggingface/lerobot/Suzumiya894/smolvla_dataset2_vel/"

    to_pop_features = ["head_pan.pos", "head_tilt.pos", "base_y.pos", "base_theta.pos"]  # 需要删除的特征名
    KEEP_VEL = True  # 是否将速度作为ACTION
    if KEEP_VEL:
        process_dataset_keep_vel(YOUR_DATASET_ROOT, to_pop_names=to_pop_features)
    else:
        # 将next_pos作为ACTION
        process_dataset_vel_to_pos(YOUR_DATASET_ROOT, to_pop_names=to_pop_features)
