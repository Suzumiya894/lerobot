"""
该脚本用于确保STRETCH数据集中每个episode中base_x的值从0开始，并重新计算统计指标。
注意，该脚本目前只适用于control_action_base_only_x=True的情况，即机器人仅沿x轴移动。对于其他情况，需要进行坐标变换才能修正，TODO。
"""
import os
import pandas as pd
import numpy as np
import json

def process_dataset_base_x(dataset_path: str, BASE_X_INDEX: int = 8, ACTION_LENGTH: int = 11):
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
            print(f"\t处理列: {col}", end=' ')
            assert len(df[col][0]) == ACTION_LENGTH
            assert isinstance(df[col][0], np.ndarray)
            base_xs = df[col].apply(lambda lst: lst[BASE_X_INDEX])
            print("base x 起始位置：", base_xs[0])
            # 步骤2：将第一个值作为基准，确保base_xs从0开始
            base_xs = base_xs - base_xs[0] 

            # 步骤3：计算统计指标
            min_val = float(np.min(base_xs))
            max_val = float(np.max(base_xs))
            mean_val = float(np.mean(base_xs))
            std_val = float(np.std(base_xs))

            df[col] = [
                np.concatenate([lst[:BASE_X_INDEX], [base_x], lst[BASE_X_INDEX + 1:]])
                for lst, base_x in zip(df[col], base_xs)
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
                        print(f"\t\t原始base_x统计数据: {key} = {data['stats'][col][key][BASE_X_INDEX]}")
                        data['stats'][col][key][BASE_X_INDEX] = value  # 更新角度统计数据
                        print(f"\t\t更新后的角度统计数据: {key} = {value}")

                modified_episodes_stats.append(data)
            
        # 将修改后的数据写回到同一个文件
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in modified_episodes_stats:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except FileNotFoundError:
        print(f"⚠️ 未找到 `{filename}`。")



if __name__ == "__main__":
    YOUR_DATASET_ROOT = "/home/fdse/yew/huggingface/lerobot/Suzumiya894/smolvla_dataset2"

    process_dataset_base_x(YOUR_DATASET_ROOT, BASE_X_INDEX=8, ACTION_LENGTH=11)  # 这里的BASE_X_INDEX可以根据需要调整，对于11维的Stretch数据集，base_x通常是第9个元素（索引为8）