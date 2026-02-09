'''
分析模型在测试集上的不确定性和证据量，并生成可视化图表和CSV报告。

该脚本会执行以下操作：
1. 加载指定的模型和测试数据。
2. 遍历测试集，计算每个样本的预测类别、不确定性、总证据量。
3. 将每个样本的详细分析结果保存到 a CSV 文件中。
4. 按真实类别分组，生成不确定性和总证据量的盒须图 (Box Plot)，以对比已知类和未知类的分布差异。
5. 所有输出保存在模型检查点目录下的 'test' 子目录中。

使用方法:
    python common/analyze_uncertainty.py --config projects/ResNet2d_EDL/config.yaml
'''

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 matplotlib 中文字体，解决中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config
from common.utils.data_loader import RawSignalDataset, NpyIndexDataset
from common.edl_losses import relu_evidence
from models import get_model

def get_test_dataset(data_config, split='test'):
    """根据配置加载测试数据集，不进行类别过滤 (analyze_uncertainty 版本)"""
    data_type = data_config.get('type')
    if data_type is None:
        raise ValueError("配置文件中必须指定 data.type")

    if data_type == 'raw_signal':
        base_dir = data_config.get('raw_signal_output_dir')
        split_dir = os.path.join(base_dir, split)
        # 加载测试集时，不传递 filter_classes，加载所有数据
        return RawSignalDataset(split_dir=split_dir, filter_classes=None)
    elif data_type == 'wpt':
        base_dir = data_config.get('wpt_output_dir')
        split_dir = os.path.join(base_dir, split)
        return NpyIndexDataset(split_dir=split_dir, filter_classes=None)
    else:
        raise ValueError(f"脚本目前仅支持 'raw_signal' 和 'wpt' 数据类型，当前为: {data_type}")

def analyze_model(model, dataloader, num_classes, device):
    """遍历数据集，计算不确定性和证据"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            logits, features = model(inputs, return_features=True)
            
            evidence = relu_evidence(logits)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1)
            total_evidence = torch.sum(evidence, dim=1)
            _, preds = torch.max(logits, 1)

            # 逐个样本保存结果
            for i in range(len(labels)):
                sample_result = {
                    'true_label': labels[i].item(),
                    'predicted_label': preds[i].item(),
                    'uncertainty': uncertainty[i].item(),
                    'total_evidence': total_evidence[i].item(),
                }
                # 记录每个类别的证据值
                for j in range(num_classes):
                    sample_result[f'evidence_{j}'] = evidence[i][j].item()
                
                results.append(sample_result)
    
    return pd.DataFrame(results)

def plot_distributions(df, known_classes, label_names, output_dir):
    """绘制不确定性和证据量的柱状图"""
    # 增加一个'type'列来区分已知/未知类
    df["type"] = df["true_label"].apply(lambda x: "Known" if x in known_classes else "Unknown")
    df["label_name"] = df["true_label"].map(label_names)

    # --- 1. 绘制不确定性分布图 (柱状图) ---
    plt.figure(figsize=(14, 8))
    # 使用 barplot 来显示每个类别的平均不确定性
    sns.barplot(x="label_name", y="uncertainty", data=df, hue="type", dodge=False, 
                palette={"Known": "skyblue", "Unknown": "salmon"})
    plt.title("各类别样本的平均不确定性", fontsize=18)
    plt.xlabel("真实类别", fontsize=12)
    plt.ylabel("平均不确定性 (Mean Uncertainty)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.tight_layout()
    uncertainty_plot_path = os.path.join(output_dir, "uncertainty_distribution_bar.png")
    plt.savefig(uncertainty_plot_path, dpi=300)
    print(f"不确定性柱状图已保存至: {uncertainty_plot_path}")
    plt.close()

    # --- 2. 绘制总证据量分布图 (柱状图) ---
    plt.figure(figsize=(14, 8))
    # 使用 barplot 来显示每个类别的平均总证据量
    sns.barplot(x="label_name", y="total_evidence", data=df, hue="type", dodge=False, 
                palette={"Known": "skyblue", "Unknown": "salmon"})
    plt.title("各类别样本的平均总证据量", fontsize=18)
    plt.xlabel("真实类别", fontsize=12)
    plt.ylabel("平均总证据量 (Mean Total Evidence)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.tight_layout()
    evidence_plot_path = os.path.join(output_dir, "evidence_distribution_bar.png")
    plt.savefig(evidence_plot_path, dpi=300)
    print(f"总证据量柱状图已保存至: {evidence_plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='模型不确定性与证据量分析脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 1. 加载配置
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']
    
    known_classes = data_config['openset']['known_classes']
    
    # 类别名称映射 (从 visualize_features.py 复制过来，确保一致)
    label_names = {
        0: 'Normal',
        1: 'IR007',
        2: 'B007',
        3: 'OR007@6',
        4: 'IR014',
        5: 'B014',
        6: 'OR014@6',
        7: 'IR021',
        8: 'B021',
        9: 'OR021@6'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 加载模型
    checkpoint_dir = train_config['checkpoint_dir']
    num_classes = model_config.get('num_classes')
    
    model = get_model(model_config.get('type'), num_classes=num_classes).to(device)
    model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"错误: 在 {model_path} 未找到模型文件 'best_model.pth'")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"从 {model_path} 加载模型成功。")
    
    # 3. 准备数据
    batch_size = train_config['batch_size']
    test_dataset = get_test_dataset(data_config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"测试集大小: {len(test_dataset)}")

    # 4. 分析模型
    print("\n正在分析模型在测试集上的表现...")
    results_df = analyze_model(model, test_loader, num_classes, device)
    print("分析完成。")

    # 5. 设置输出目录并保存结果
    output_dir = os.path.join(checkpoint_dir, 'test')
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'uncertainty_analysis.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"详细分析结果已保存至: {csv_path}")

    # 6. 绘图
    print("\n正在生成可视化图表...")
    # 获取测试数据中的所有标签，以确保 label_names 映射完整
    all_test_labels = results_df['true_label'].unique()
    mapped_label_names = {k: v for k, v in label_names.items() if k in all_test_labels}

    plot_distributions(results_df, known_classes, mapped_label_names, output_dir)
    print("所有任务完成！")

if __name__ == '__main__':
    main()
