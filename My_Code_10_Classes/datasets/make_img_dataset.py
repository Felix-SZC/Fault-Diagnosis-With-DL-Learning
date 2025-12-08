import os
import sys
import warnings
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from matplotlib import MatplotlibDeprecationWarning

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config

# 设置警告过滤器，忽略掉Matplotlib的弃用警告，保持输出整洁
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def split_datasets(data_file_csv, split_rate, random_state=42):
    """
    本函数负责将数据集划分为训练集、验证集和测试集。
    """
    # 确保 split_rate 的和为 1
    assert abs(sum(split_rate) - 1) < 1e-9, "split_rate 的总和必须为 1"

    # 1.读取数据
    samples_data = pd.read_csv(data_file_csv)
    # 2.将数据和标签分开
    X = samples_data.iloc[:, :-1].values
    y = samples_data.iloc[:, -1].values

    # 3.首先划分 训练集 和 临时集（验证集 + 测试集）
    train_ratio = split_rate[0]
    val_test_ratio = 1 - train_ratio

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, random_state=random_state, stratify=y
    )

    # 4.然后从临时集中划分验证集和测试集
    val_ratio = split_rate[1]
    test_ratio = split_rate[2]
    test_size_in_temp = test_ratio / (val_ratio + test_ratio)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size_in_temp, random_state=random_state, stratify=y_temp
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_time_frequency_images(data, labels, folder, config):
    """
    本函数负责将信号数据通过短时傅里叶变换转换成时频图并保存。
    """
    # 1.从配置文件中读取相关参数
    img_size = config['data']['image_size']              # 图像尺寸
    fs = config['data']['sampling_frequency']            # 采样频率
    window_size = config['data']['stft_window_size']     # STFT窗口大小
    overlap = config['data']['stft_overlap']             # STFT窗口重叠率

    # 2.检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 3.计算STFT窗口重叠的样本点数
    overlap_samples = int(window_size * overlap)

    # 4.遍历数据集中的每一个信号样本
    for i, signal in enumerate(data):
        # 执行短时傅里叶变换
        frequencies, times, magnitude = stft(signal, fs, nperseg=window_size, noverlap=overlap_samples)
        
        # 5.绘制时频图
        plt.pcolormesh(times, frequencies, np.abs(magnitude), shading='gouraud')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gcf().set_size_inches(img_size / 100, img_size / 100)
        
        # 6.保存图像
        image_path = os.path.join(folder, f'{i}_{labels[i]}.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        
        # 7.清理和关闭图像
        plt.clf()
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate STFT Image Dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # 1. 加载配置文件 
    config = load_config(args.config)

    # 2. 划分数据集 
    csv_path = config['data']['csv_file_path']
    split_ratios = config['data']['split_ratios']
    X_train, y_train, X_val, y_val, X_test, y_test = split_datasets(csv_path, split_ratios)

    # 3. 准备图像输出目录 
    img_output_dir = config['data']['img_output_dir']
    train_path = os.path.join(img_output_dir, 'train')
    val_path = os.path.join(img_output_dir, 'val')
    test_path = os.path.join(img_output_dir, 'test')

    # 4. 循环生成各个数据集的时频图 
    print("开始生成训练集图像...")
    make_time_frequency_images(X_train, y_train, train_path, config)
    
    print("开始生成验证集图像...")
    make_time_frequency_images(X_val, y_val, val_path, config)
    
    print("开始生成测试集图像...")
    make_time_frequency_images(X_test, y_test, test_path, config)

    # 5. 任务完成提示 
    print("时频图像生成完毕！")
