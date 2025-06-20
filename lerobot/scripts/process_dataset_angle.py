"""
对于STRETCH数据集，默认采到的角度值在0~2pi之间，这就会导致如果机器人角度在0点附近震荡，在数据集中则是从0跃变到2pi，这样会导致模型学习到错误的角度变化，因此需要使用该脚本将角度值转换到[-pi, pi]区间。
"""
import os
import pandas as pd
import numpy as np
import json

def process_dataset_angle(dataset_path: str, BASE_THETA_INDEX: int = 10, ACTION_LENGTH: int = 11):
    stats = {}
    # 处理.parquet文件
    data_chunk_path = os.path.join(dataset_path, 'data', 'chunk-000')
    parquet_files = [f for f in os.listdir(data_chunk_path) if f.endswith('.parquet')]

    if not os.path.exists(data_chunk_path):
        print(f"❌ 错误: 数据目录不存在 {data_chunk_path}")
        return

    parquet_files.sort()  # 确保文件按字母顺序处理

    for i, filename in enumerate(parquet_files):
        file_path = os.path.join(data_chunk_path, filename)
        print(f"  ({i}/{len(parquet_files)}) 正在处理: {filename}")

        # 将 Parquet 文件读入 pandas DataFrame
        df = pd.read_parquet(file_path)
        stats[i] = {}
        for col in ['action', 'observation.state']:
            print(f"\t处理列: {col}")
            assert len(df[col][0]) == ACTION_LENGTH
            assert isinstance(df[col][0], np.ndarray)
            angles = df[col].apply(lambda lst: lst[BASE_THETA_INDEX])

            # 步骤2：将角度转换到[-π, π]区间
            converted_angles = np.where(angles > np.pi, angles - 2 * np.pi, angles)

            # 步骤3：计算统计指标
            min_val = float(np.min(converted_angles))
            max_val = float(np.max(converted_angles))
            mean_val = float(np.mean(converted_angles))
            std_val = float(np.std(converted_angles))

            df[col] = [
                np.concatenate([lst[:BASE_THETA_INDEX], [angle], lst[BASE_THETA_INDEX + 1:]])
                for lst, angle in zip(df[col], angles)
            ]

            assert len(df[col][0]) == ACTION_LENGTH

            stats[i][col] = {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val
            }
        
        df.to_parquet(file_path)

    # 处理episodes_stats.jsonl文件
    filename = 'episodes_stats.jsonl'   # 只需要处理episodes_stats.jsonl
    modified_episodes_stats = []
    meta_path = os.path.join(dataset_path, 'meta')
    jsonl_path = os.path.join(meta_path, filename)
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                cur_idx = data['episode_index']
                print(f"正在处理 episodes_stats.jsonl 中的第 {cur_idx} 行数据...")
                for col in ['action', 'observation.state']:
                    print(f"\t处理列: {col}")
                    for key, value in stats[cur_idx][col].items():
                        print(f"\t\t原始角度统计数据: {key} = {data['stats'][col][key][BASE_THETA_INDEX]}")
                        data['stats'][col][key][BASE_THETA_INDEX] = value  # 更新角度统计数据
                        print(f"\t\t更新后的角度统计数据: {key} = {value}")

                modified_episodes_stats.append(data)
            
        # 将修改后的数据写回到同一个文件
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in modified_episodes_stats:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except FileNotFoundError:
        print(f"⚠️ 未找到 `{filename}`。")



if __name__ == "__main__":
    YOUR_DATASET_ROOT = "/home/fdse/yew/huggingface/lerobot/Suzumiya894/stretch_test2"

    process_dataset_angle(YOUR_DATASET_ROOT, BASE_THETA_INDEX=10, ACTION_LENGTH=11)  # 这里的BASE_THETA_INDEX可以根据需要调整，对于11维的Stretch数据集，base_theta通常是第11个元素（索引为10）