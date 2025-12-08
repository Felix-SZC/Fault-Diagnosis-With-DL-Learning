import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import MatplotlibDeprecationWarning

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config

# 忽略特定的 Matplotlib 弃用警告
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def split_datasets_by_ratio(X, y, split_rate, random_state=42):
    """
    按比例划分训练/验证/测试集。
    """
    assert abs(sum(split_rate) - 1) < 1e-9, "split_rate 的总和必须为 1"

    train_ratio = split_rate[0]
    val_ratio = split_rate[1]
    test_ratio = split_rate[2]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
    )
    
    test_size_in_temp = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size_in_temp, random_state=random_state, stratify=y_temp
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def ensure_dir(path: str):
    """若目录不存在则创建（幂等）。"""
    if not os.path.exists(path):
        os.makedirs(path)


def save_raw_signal_arrays(data: np.ndarray, labels: np.ndarray, out_dir: str):
    """
    将一批 1D 信号保存为 .npy 文件
    """
    ensure_dir(out_dir)
    index_rows = []
    for i, signal in enumerate(data):
        signal_array = signal.astype(np.float32)
        fname = f"{i}_{labels[i]}.npy"
        fpath = os.path.join(out_dir, fname)
        np.save(fpath, signal_array)
        index_rows.append({"file": fname, "label": int(labels[i])})
        if (i + 1) % 200 == 0:
            print(f"  已处理 {i + 1}/{len(data)}")
    
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(os.path.join(out_dir, "index.csv"), index=False, encoding='utf-8')
    print(f"已保存索引: {os.path.join(out_dir, 'index.csv')}")


def main():
    parser = argparse.ArgumentParser(description='Generate Raw Signal Dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # 1) 读取配置
    config = load_config(args.config)
    csv_path = config['data']['csv_file_path']
    split_ratios = config['data']['split_ratios']
    raw_signal_output_dir = config['data'].get('raw_signal_output_dir', 'data/RawSignals')

    # 2) 加载 CSV
    print("加载 CSV 数据...")
    samples_df = pd.read_csv(csv_path, header=0)
    X = samples_df.iloc[:, :-1].values.astype(np.float32)
    y = samples_df.iloc[:, -1].values.astype(int)
    print(f"总样本数: {X.shape[0]}, 每样本长度: {X.shape[1]}")

    # 3) 划分数据集
    X_train, y_train, X_val, y_val, X_test, y_test = split_datasets_by_ratio(X, y, split_ratios)

    # 4) 生成输出
    train_dir = os.path.join(raw_signal_output_dir, 'train')
    val_dir = os.path.join(raw_signal_output_dir, 'val')
    test_dir = os.path.join(raw_signal_output_dir, 'test')

    print("\n生成训练集原始信号数组...")
    save_raw_signal_arrays(X_train, y_train, train_dir)
    print("生成验证集原始信号数组...")
    save_raw_signal_arrays(X_val, y_val, val_dir)
    print("生成测试集原始信号数组...")
    save_raw_signal_arrays(X_test, y_test, test_dir)

    print("\n原始信号预处理完成！")
    print(f"输出目录: {os.path.abspath(raw_signal_output_dir)}")


if __name__ == '__main__':
    main()
