import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_loader import RawSignalDataset
from utils.helpers import load_config
from models import get_model

config = load_config('configs/config.yaml')

data_config = config['data']
train_config = config['train']
model_config = config['model']

# 使用原始信号数据目录
raw_signal_output_dir = data_config.get('raw_signal_output_dir', '../data/RawSignals')
test_dir = os.path.join(raw_signal_output_dir, 'test')

batch_size = train_config['batch_size']
checkpoint_dir = train_config['checkpoint_dir']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"从预处理目录读取: {raw_signal_output_dir}")
test_dataset = RawSignalDataset(split_dir=test_dir)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"测试集大小：{len(test_dataset)}")
print(f"使用设备：{device}")

# 根据配置自动选择和实例化模型
model = get_model(model_config).to(device)
model_type = model_config.get('type', 'Unknown')
print(f'使用模型: {model_type}')
model_path = os.path.join(checkpoint_dir, "best_model.pth")

model.load_state_dict(torch.load(model_path, weights_only=True))

model.eval()

print(f"已加载模型: {model_path}")

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='测试中'):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("测试完成!")

report = classification_report(all_labels, all_preds)
print("\n详细分类报告:")
print(report)

report_path = os.path.join(checkpoint_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"\n分类报告已保存至: {report_path}")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
cm_path = f"{config['train']['checkpoint_dir']}/confusion_matrix.png"
plt.savefig(cm_path)
print(f"\n混淆矩阵已保存至 {cm_path}")

plt.show()