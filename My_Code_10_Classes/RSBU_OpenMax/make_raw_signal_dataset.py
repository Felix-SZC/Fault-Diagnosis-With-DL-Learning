# 预处理原始1D振动信号数据集，并保存为 .npy，便于后续直接读取
# 使用方式（典型三步）：
#   1) 先用 mat_to_csv.py 生成 samples_data.csv（每行=一个样本窗口，最后一列=label）
#   2) cd ResNet && python make_raw_signal_dataset.py
#      本脚本会：读取 CSV → 按比例划分 → 保存为 .npy + 生成 index.csv
#   3) python train.py / python test.py 即可直接读取预处理结果
#
# 读取配置：configs/config.yaml
#   依赖键：
#     data.csv_file_path        # CSV路径
#     data.split_ratios         # 数据集划分比例 [train, val, test]
#     data.raw_signal_output_dir # 预处理输出目录（默认 ../data/RawSignals）
#
# 产出目录结构（示例）：
#   ../data/RawSignals/
#     train/
#       index.csv            # 两列：file,label
#       0_3.npy              # 一个样本：形状 (time_steps,) = (1024,)
#       1_7.npy
#       ...
#     val/
#       index.csv
#       ...
#     test/
#       index.csv
#       ...

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.helpers import load_config
from matplotlib import MatplotlibDeprecationWarning

# 忽略特定的 Matplotlib 弃用警告，保证控制台整洁
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def split_datasets_by_ratio(X, y, split_rate, random_state=42):
	"""
	按比例划分训练/验证/测试集。
	
	输入：
	- X: (N, time_steps) 的二维数组，每行一个样本窗口
	- y: (N,) 的标签数组
	- split_rate: [train_ratio, val_ratio, test_ratio]，总和必须为 1
	
	输出：
	- X_train, y_train, X_val, y_val, X_test, y_test
	"""
	assert abs(sum(split_rate) - 1) < 1e-9, "split_rate 的总和必须为 1"

	train_ratio = split_rate[0]
	val_ratio = split_rate[1]
	test_ratio = split_rate[2]

	# 第一步：先切出训练集
	X_train, X_temp, y_train, y_temp = train_test_split(
		X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
	)
	# 第二步：在剩余部分里按 val:test 比例划分
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
	将一批 1D 信号保存为 .npy 文件：
	- 每个样本生成一个 (time_steps,) 的 .npy 文件，文件名格式：{行索引}_{label}.npy
	- 同时生成 index.csv（两列：file,label）便于后续 DataLoader 读取
	
	输入：
	- data: (N_samples, time_steps) 的二维数组
	- labels: (N_samples,) 的标签数组
	- out_dir: 输出目录，例如 ../data/RawSignals/train
	"""
	ensure_dir(out_dir)
	index_rows = []
	for i, signal in enumerate(data):
		# 确保信号是 float32 类型
		signal_array = signal.astype(np.float32)
		# 构建文件名并保存 .npy
		fname = f"{i}_{labels[i]}.npy"
		fpath = os.path.join(out_dir, fname)
		np.save(fpath, signal_array)
		# 记录索引（后续 DataLoader 读取）
		index_rows.append({"file": fname, "label": int(labels[i])})
		# 打印进度（每 200 条）
		if (i + 1) % 200 == 0:
			print(f"  已处理 {i + 1}/{len(data)}")
	# 写出索引 CSV（UTF-8）
	index_df = pd.DataFrame(index_rows)
	index_df.to_csv(os.path.join(out_dir, "index.csv"), index=False, encoding='utf-8')
	print(f"已保存索引: {os.path.join(out_dir, 'index.csv')}")


def main():
	"""
	主流程：
	1) 读取配置（CSV路径、划分比例、输出目录）
	2) 加载 CSV：X 为信号，y 为标签
	3) 按比例划分 train/val/test
	4) 分别生成对应目录下的 .npy 与 index.csv
	"""
	# 1) 读取配置
	config = load_config("configs/config.yaml")
	csv_path = config['data']['csv_file_path']
	split_ratios = config['data']['split_ratios']
	raw_signal_output_dir = config['data'].get('raw_signal_output_dir', '../data/RawSignals')

	# 2) 加载 CSV
	print("加载 CSV 数据...")
	samples_df = pd.read_csv(csv_path, header=0)  # 第一行是列名
	# 前 N-1 列是时间点，最后一列是 label（与 mat_to_csv.py 生成格式一致）
	X = samples_df.iloc[:, :-1].values.astype(np.float32)
	y = samples_df.iloc[:, -1].values.astype(int)
	print(f"总样本数: {X.shape[0]}, 每样本长度: {X.shape[1]}")

	# 3) 划分数据集（复现实验：固定随机种子）
	X_train, y_train, X_val, y_val, X_test, y_test = split_datasets_by_ratio(X, y, split_ratios)

	# 4) 分别生成 train/val/test 的输出
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
	print("结构：train/val/test 目录下为 .npy 样本与 index.csv")
	print("下次训练可直接从这些 .npy 读取，跳过CSV加载和划分。")


if __name__ == '__main__':
	main()

