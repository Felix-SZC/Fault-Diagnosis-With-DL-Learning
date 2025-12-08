import os
import sys
import argparse
from scipy.io import loadmat
import pandas as pd

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config

def find_data_key(mat_data):
    for key in mat_data.keys():
        if key.endswith('_DE_time'):
            return key

# 使用“滑动窗口”技术，将一维时间序列数据切分成带有重叠的多个样本。
def split_data_with_overlap(data, time_steps, label, overlap_ratio):
    stride = int(time_steps * (1 - overlap_ratio)) # 计算每次窗口滑动的步长。
    
    samples_count = (len(data) - time_steps) // stride + 1 # 计算可以生成的总样本数
    
    data_list = [] # 用于存放所有切分好的样本
    for i in range(samples_count):
        start_idx = i * stride
        end_idx = start_idx + time_steps

        sample = data[start_idx:end_idx].tolist()# 从原始数据中切片，得到一个样本
        
        sample.append(label)# 在样本数据的末尾，追加上它的类别标签
        data_list.append(sample)
    
    # 定义 DataFrame 的列名。
    # 前 N 列是数据点 (0, 1, ..., N-1)，最后一列是 'label'。
    columns = list(range(time_steps)) + ['label']

    return pd.DataFrame(data_list, columns=columns)


# 制作样本数据集
def make_samples(data_path, file_names, truncated_value, time_steps, overlap_ratio):
    all_samples_dfs = [] # 创建一个空列表，用于汇集从每个 .mat 文件中处理出来的样本 DataFrame
    
    print("\n开始处理 .mat 文件...")

    for label, file_name in enumerate(file_names):
        # 拼接出 .mat 文件的完整路径
        file_path = os.path.join(data_path, file_name)

        print(f"  正在处理文件: {file_name} (分配标签: {label})")

        mat_data = loadmat(file_path)
        
        data_key = find_data_key(mat_data) # 查找包含驱动端数据的键名
        
        data_series = mat_data[data_key].reshape(-1) # 重塑数据
        data_series = data_series[:truncated_value] # 从头截取指定长度的数据
        
        split_samples_df = split_data_with_overlap(data_series, time_steps, label, overlap_ratio) # 使用滑动窗口切分数据
        all_samples_dfs.append(split_samples_df)

    final_samples_df = pd.concat(all_samples_dfs, ignore_index=True)

    final_samples_df = final_samples_df.sample(frac=1).reset_index(drop=True) # 随机打乱样本顺序
    
    return final_samples_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MAT files to CSV dataset')
    # 默认使用 Handcraft_ResNet 的配置作为参考，因为所有项目共享同一份 mat 数据配置
    parser.add_argument('--config', type=str, default='projects/Handcraft_ResNet/config.yaml', 
                        help='Path to config file containing data settings')
    args = parser.parse_args()

    # 1. 加载配置文件
    config = load_config(args.config)
    data_config = config.get('data', {})

    # 2. 从配置中提取参数
    data_path = data_config['data_path']
    file_names = data_config['file_names']
    truncated_value = data_config['truncated_value']
    time_steps = data_config['time_steps']
    overlap_ratio = data_config['overlap_ratio']
    csv_output_path = data_config['csv_output_path']

    # 3. 创建样本数据
    samples_data = make_samples(data_path, file_names, truncated_value, time_steps, overlap_ratio)

    # 4. 保存数据到 CSV
    print(f"\n将数据保存到: {csv_output_path}")
    
    output_dir = os.path.dirname(csv_output_path)
    
    # 如果目录不存在，则创建它
    if output_dir and not os.path.exists(output_dir):
        print(f"输出目录 '{output_dir}' 不存在，正在创建...")
        os.makedirs(output_dir)
        
    # 将最终的 DataFrame 保存为 CSV 文件
    samples_data.to_csv(csv_output_path, index=False)

    print(" 数据预处理完成！")
    print(f"总共生成样本数: {len(samples_data)}")
    print(f"数据形状 (行, 列): {samples_data.shape}")
    print(f"CSV 文件已成功保存到: {csv_output_path}")
