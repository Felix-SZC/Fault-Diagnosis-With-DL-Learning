import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据，使用您指定的字典格式
snr_db = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

ResNet = {
    -5:0.9364,
    -4:0.9543,
    -3:0.9671,
    -2:0.9800,
    -1:0.9807,
    0: 0.9814,
    1: 0.9907,
    2: 0.9979,
    3: 0.9936,
    4: 0.9971,
    5: 0.9936
}

DRSN_CW = {
    -5: 0.9571,
    -4: 0.9743,
    -3: 0.9878,
    -2: 0.9685,
    -1: 0.9814,
    0: 0.9907,
    1: 0.9936,
    2: 0.9971,
    3: 0.9986,
    4: 0.9993,
    5: 0.9979
}

# 2. 绘图设置
bar_width = 0.3
x = np.arange(len(snr_db))

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 3. 绘制柱状图
# 关键：在绘图前，使用 .values() 从字典中提取出所有的值
rects1 = ax.bar(x - bar_width/2, list(ResNet.values()), bar_width, label='ResNet')
rects2 = ax.bar(x + bar_width/2, list(DRSN_CW.values()), bar_width, label='DRSN-CW')

# 4. 美化和设置标签 (这部分完全不变)
ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Train Accuracy', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(snr_db)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 1.3)
ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=7)
ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=7)
fig.tight_layout()

# 保存图片
save_path = 'My_Code_10_Classes/RSBU/train_accuracy_comparison.png'
fig.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'图片已保存至: {save_path}')

plt.show()
