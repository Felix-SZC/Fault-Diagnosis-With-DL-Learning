import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from utils.helpers import load_config
from scipy import signal as sig

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def plot_graph(x_data, y_data, title, xlabel, ylabel, save_name, xlim=None, annotations=None):
    """一个用于绘制通用图形的函数"""
    plt.figure(figsize=(15, 5))
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    plt.grid(True)
    # 可选：在单图上叠加竖线标注（如特征频率及倍频）
    if annotations:
        for ann in annotations:
            plt.axvline(x=ann['x'], color=ann['color'], linestyle=ann['style'], label=ann['label'], alpha=0.4)
        plt.legend()
    plt.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
    plt.show()

def plot_subplot(ax, x_data, y_data, title, xlabel, ylabel, xlim=None, annotations=None):
    """在一个指定的子画板(ax)上绘制图形的通用函数"""
    ax.plot(x_data, y_data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if xlim:
        ax.set_xlim(xlim)
    if annotations:
        for ann in annotations:
            ax.axvline(x=ann['x'], color=ann['color'], linestyle=ann['style'], label=ann['label'], alpha=0.4)
        ax.legend()

# 一、加载数据

# 加载文件
config = load_config('configs/config.yaml')
data_config = config.get('data', {})
analysis_config = config.get('analysis', {})
processing_config = config.get('processing', {})
data_path = data_config['data_path']  # 原始文件路径
save_path = data_config['save_path'] # 文件保存地址

file_name = data_config['file_name'] # 目标信号名
file_path = os.path.join(data_path,file_name) # 故障信号地址
file_stem, _ = os.path.splitext(file_name)
save_path = os.path.join(save_path, file_stem)
os.makedirs(save_path, exist_ok=True)
mat_data = scipy.io.loadmat(file_path) # 加载故障文件

# 备份当前配置到结果目录，便于追踪本次运行参数
shutil.copy('configs/config.yaml', os.path.join(save_path, 'config.yaml'))

# 带通滤波参数加载
fft_order = processing_config['fft_order']
low_cut = processing_config['low_cut']
high_cut = processing_config['high_cut']


# 自动提取转速（rpm）。若 .mat 未包含，则回退到配置（若仍无，则采用常见工况 1797 rpm）
rpm = None
for key_rpm in mat_data.keys():
    if key_rpm.endswith('RPM'):
        rpm = float(np.squeeze(mat_data[key_rpm]))
        break
fr = rpm / 60.0  # 转频 (Hz)


# 寻找目标信号
signal = None # 信号数据
key = ''  # 信号键值
for key_ in mat_data.keys():
    if 'DE_time' in key_:
        key = key_
        signal = mat_data[key_].flatten()

if signal is not None:
    # print('驱动端信号键名：',key)
    # print('原始信号信号类型：',type(signal))
    print('原始信号形状：',signal.shape)
else:
    print('未找到原始信号')


# 二、可视化信号

# 获取时域信号
fs = data_config['sampling_frequency'] #采样频率
time = np.linspace(0,len(signal)/fs,len(signal))

# 验证
# print(time)
# print(len(time))

# 绘制时域信号
plot_graph(time, signal, '原始时域图', '时间', '幅度', '1. 原始时域图.png', xlim=(0,0.1))


# 三、初始频谱图分析

# 计算振幅
fft_amplitudes = (abs(np.fft.fft(signal))/len(signal))[:len(signal)//2]
# 计算频率轴
fft_frequencies = (np.fft.fftfreq(len(signal),1/fs))[:len(signal)//2]
# 绘制频域信号
plot_graph(fft_frequencies, fft_amplitudes, '原始频域图', '频率', '振幅', '2. 原始频域图.png')


# 四、带通滤波
b, a = sig.butter(fft_order, [low_cut, high_cut], btype = 'bandpass', fs = fs)
signal_filtered = sig.filtfilt(b, a, signal)
print("带通滤波信号的形状:", signal_filtered.shape)

# 绘制时域信号
plot_graph(time, signal_filtered, '带通滤波时域图', '时间', '幅度', '3. 带通时域图.png', xlim=(0,0.1))

# 计算振幅
fft_amplitudes_filtered = (abs(np.fft.fft(signal_filtered))/len(signal_filtered))[:len(signal_filtered)//2]
# 计算频率轴
fft_frequencies_filtered = (np.fft.fftfreq(len(signal_filtered),1/fs))[:len(signal_filtered)//2]
plot_graph(fft_frequencies_filtered, fft_amplitudes_filtered, '带通滤波频域图', '频率', '振幅', '4. 带通频域图.png')


# 五、希尔伯特变换+包络谱分析
signal_analytic  = sig.hilbert(signal_filtered) # 希尔伯特变换
signal_envelope = np.abs(signal_analytic) # 包络谱分析
envelope_no_dc = signal_envelope - np.mean(signal_envelope)
print("包络信号的形状:", signal_envelope.shape)

plt.figure(figsize=(15,5))
plt.plot(time, signal_filtered, alpha=0.6, label='滤波信号')
plt.plot(time,signal_envelope, color='red', label='包络信号')
plt.title('带通滤波信号与包络')
plt.xlabel('时间')
plt.ylabel('振幅')
plt.xlim(0, 0.1)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_path, '5. 包络时域图.png'), dpi=300, bbox_inches='tight')
plt.show()

# 计算振幅
fft_amplitudes_envelope = (abs(np.fft.fft(envelope_no_dc))/len(envelope_no_dc))[:len(envelope_no_dc)//2]
# 计算频率轴
fft_frequencies_envelope = (np.fft.fftfreq(len(envelope_no_dc),1/fs))[:len(envelope_no_dc)//2]

# CWRU 缺陷频率倍数，计算特征频率
bpfi = 5.4152 * fr
bpfo = 3.5848 * fr
ftf  = 0.39828 * fr
bsf  = 4.7135 * fr

print(f"转速信息: {rpm} RPM -> 转频 fr = {fr:.2f} Hz")
print(f"特征频率: BPFI = {bpfi:.2f} Hz, BPFO = {bpfo:.2f} Hz, BSF = {bsf:.2f} Hz, FTF = {ftf:.2f} Hz")

## 根据配置选择要标注的特征频率族（BPFI/BPFO/BSF/FTF/NONE）
max_view = int(analysis_config.get('max_view'))  # 与图像 x 轴范围配合
base_feature = str(analysis_config.get('base_feature', 'AUTO')).upper()

# 自动根据文件名判断故障类型
if base_feature == 'AUTO':
    if 'IR' in file_name or 'IR' in file_stem:
        base_feature = 'BPFI'  # 内圈故障
        print(f"检测到内圈故障文件 '{file_name}'，自动选择 BPFI")
    elif 'B' in file_name or 'B' in file_stem:
        base_feature = 'BSF'   # 滚动体故障
        print(f"检测到滚动体故障文件 '{file_name}'，自动选择 BSF")
    elif 'OR' in file_name or 'OR' in file_stem:
        base_feature = 'BPFO'  # 外圈故障
        print(f"检测到外圈故障文件 '{file_name}'，自动选择 BPFO")
    else:
        base_feature = 'BPFI'  # 默认
        print(f"无法自动判断文件类型，默认使用 BPFI")

if base_feature == 'BPFO':
    base_freq = bpfo
    base_label = 'BPFO'
elif base_feature == 'BSF':
    base_freq = bsf
    base_label = 'BSF'
elif base_feature == 'FTF':
    base_freq = ftf
    base_label = 'FTF'
elif base_feature in ['NONE', 'NULL', 'OFF']:
    base_freq = None
    base_label = ''
else:
    base_freq = bpfi
    base_label = 'BPFI'

annotations = []

# 转频及其前5阶谐波（不超过可视频段）
max_k_fr = min(5, int(max_view // max(1e-9, fr)))
fr_harmonics = [k * fr for k in range(1, max_k_fr + 1)]
annotations += [{'x': f, 'color': 'green', 'style': ':', 'label': f'fr x{idx+1}'} for idx, f in enumerate(fr_harmonics)]

# 选定特征频率族的前5阶倍频（不超过可视频段）
if base_freq is not None:
    max_k_feat = min(5, int(max_view // max(1e-9, base_freq)))
    feat_harmonics = [k * base_freq for k in range(1, max_k_feat + 1)]
    annotations = (
        [{'x': f, 'color': 'red', 'style': '--', 'label': f'{base_label} x{idx+1}'} for idx, f in enumerate(feat_harmonics)]
        + annotations
    )

plot_graph(fft_frequencies_envelope, fft_amplitudes_envelope, '包络谱', '频率', '振幅', '6. 包络谱.png', xlim=(0,max_view), annotations=annotations)


# 六、整合分析图
fig, axs = plt.subplots(2, 3, figsize=(20, 8))
fig.suptitle(f'轴承故障信号分析全流程 - {file_name}', fontsize=16)

# 1 原始时域
plot_subplot(axs[0, 0], time, signal, '1. 原始时域信号', '时间', '幅度', xlim=(0,0.1))
# 2 原始频谱
plot_subplot(axs[0, 1], fft_frequencies, fft_amplitudes, '2. 原始频谱图', '频率', '振幅')
# 3 带通时域
plot_subplot(axs[0, 2], time, signal_filtered, '3. 带通滤波时域图', '时间', '幅度', xlim=(0,0.1))
# 4 带通频谱
plot_subplot(axs[1, 0], fft_frequencies_filtered, fft_amplitudes_filtered, '4. 带通滤波频谱图', '频率', '振幅')

# 5 滤波与包络
axs[1, 1].plot(time, signal_filtered, alpha=0.6, label='滤波信号')
axs[1, 1].plot(time, signal_envelope, color='red', label='包络')
axs[1, 1].set_title('5. 滤波信号与包络')
axs[1, 1].set_xlabel('时间')
axs[1, 1].set_ylabel('幅度')
axs[1, 1].set_xlim(0, 0.1)
axs[1, 1].legend()
axs[1, 1].grid(True)

# 6 包络谱与特征频率标注
plot_subplot(axs[1, 2], fft_frequencies_envelope, fft_amplitudes_envelope, '6. 最终包络谱分析', '频率', '振幅', xlim=(0,max_view), annotations=annotations)

# 最终调整与显示
plt.tight_layout(rect=[0, 0, 1, 1])  # 为总标题留出空间
plt.savefig(os.path.join(save_path, f'0. 分析全流程图 - {file_stem}.png'), dpi=300)
plt.show()

