import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pywt
from matplotlib import MatplotlibDeprecationWarning

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config

# 忽略特定的 Matplotlib 弃用警告
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def wavelet_packet_decomposition(signal: np.ndarray, wavelet: str = 'db4', max_level: int = 4) -> np.ndarray:
	"""
	将一维信号做小波包分解，返回 (频带数, 时间点数) 的系数矩阵。
	"""
	wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
	leaf_nodes = [node.path for node in wp.get_level(max_level, order='freq')]
	coeffs_list = [wp[path].data for path in leaf_nodes]
	max_len = max(len(c) for c in coeffs_list)
	coeff_matrix = np.zeros((len(coeffs_list), max_len), dtype=np.float32)
	for i, c in enumerate(coeffs_list):
		l = min(len(c), max_len)
		coeff_matrix[i, :l] = c[:l]
	return coeff_matrix


def split_datasets_by_ratio(X, y, split_rate, random_state=42):
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
	if not os.path.exists(path):
		os.makedirs(path)


def save_wpt_arrays(data: np.ndarray, labels: np.ndarray, out_dir: str, wavelet: str, level: int):
	ensure_dir(out_dir)
	index_rows = []
	for i, signal in enumerate(data):
		wpt_matrix = wavelet_packet_decomposition(signal.astype(np.float32), wavelet=wavelet, max_level=level)
		fname = f"{i}_{labels[i]}.npy"
		fpath = os.path.join(out_dir, fname)
		np.save(fpath, wpt_matrix)
		index_rows.append({"file": fname, "label": int(labels[i])})
		if (i + 1) % 200 == 0:
			print(f"  已处理 {i + 1}/{len(data)}")
            
	index_df = pd.DataFrame(index_rows)
	index_df.to_csv(os.path.join(out_dir, "index.csv"), index=False, encoding='utf-8')
	print(f"已保存索引: {os.path.join(out_dir, 'index.csv')}")


def main():
    parser = argparse.ArgumentParser(description='Generate WPT Dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

	# 1) 读取配置
    config = load_config(args.config)
	csv_path = config['data']['csv_file_path']
	split_ratios = config['data']['split_ratios']
	wavelet = config['wavelet']['wavelet_type']
	level = int(config['wavelet']['max_level'])
    wpt_output_dir = config['data'].get('wpt_output_dir', 'data/WPTarrays')

	# 2) 加载 CSV
	print("加载 CSV 数据...")
	samples_df = pd.read_csv(csv_path)
	X = samples_df.iloc[:, :-1].values.astype(np.float32)
	y = samples_df.iloc[:, -1].values.astype(int)
	print(f"总样本数: {X.shape[0]}, 每样本长度: {X.shape[1]}")

    # 3) 划分数据集
	X_train, y_train, X_val, y_val, X_test, y_test = split_datasets_by_ratio(X, y, split_ratios)

    # 4) 生成输出
	train_dir = os.path.join(wpt_output_dir, 'train')
	val_dir = os.path.join(wpt_output_dir, 'val')
	test_dir = os.path.join(wpt_output_dir, 'test')

	print("\n生成训练集 WPT 数组...")
	save_wpt_arrays(X_train, y_train, train_dir, wavelet, level)
	print("生成验证集 WPT 数组...")
	save_wpt_arrays(X_val, y_val, val_dir, wavelet, level)
	print("生成测试集 WPT 数组...")
	save_wpt_arrays(X_test, y_test, test_dir, wavelet, level)

	print("\nWPT 预计算完成！")
	print(f"输出目录: {os.path.abspath(wpt_output_dir)}")


if __name__ == '__main__':
	main()
