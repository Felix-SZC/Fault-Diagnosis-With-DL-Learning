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

def _analyze_standard_edl(model, inputs, labels, num_classes, device):
    """标准 K 类 EDL 模型：单输出 logits [B, K]"""
    try:
        output = model(inputs, return_features=True)
    except TypeError:
        output = model(inputs)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    evidence = relu_evidence(logits)
    alpha = evidence + 1
    uncertainty = num_classes / torch.sum(alpha, dim=1)
    total_evidence = torch.sum(evidence, dim=1)
    _, preds = torch.max(logits, 1)

    batch_results = []
    for i in range(len(labels)):
        sample_result = {
            'true_label': labels[i].item(),
            'predicted_label': preds[i].item(),
            'uncertainty': uncertainty[i].item(),
            'total_evidence': total_evidence[i].item(),
        }
        for j in range(num_classes):
            sample_result[f'evidence_{j}'] = evidence[i][j].item()
        batch_results.append(sample_result)
    return batch_results


def _analyze_binary_edl_with_main(model, inputs, labels, num_classes, device):
    """多二分类头 EDL 且启用主头：用主头 logits 做不确定性与证据分析（与标准 EDL 一致）"""
    output = model(inputs)
    logits = output[0]  # main_logits [B, K]
    evidence = relu_evidence(logits)
    alpha = evidence + 1
    uncertainty = num_classes / torch.sum(alpha, dim=1)
    total_evidence = torch.sum(evidence, dim=1)
    _, preds = torch.max(logits, 1)
    batch_results = []
    for i in range(len(labels)):
        sample_result = {
            'true_label': labels[i].item(),
            'predicted_label': preds[i].item(),
            'uncertainty': uncertainty[i].item(),
            'total_evidence': total_evidence[i].item(),
        }
        for j in range(num_classes):
            sample_result[f'evidence_{j}'] = evidence[i][j].item()
        batch_results.append(sample_result)
    return batch_results


def _analyze_binary_edl(model, inputs, labels, num_classes, device):
    """多二分类头 EDL 模型（无主头）：输出 binary_logits_list，K 个 [B, 2]"""
    output = model(inputs)
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], list):
        binary_logits_list = output[1]
    else:
        binary_logits_list = output

    # 每个头的 evidence、alpha、以及 “Yes” 概率
    yes_probs_list = []   # K 个 [B]
    evidence_yes_list = []  # K 个 [B]，用于 evidence_{j} 列
    total_evidence_per_sample = None  # [B]

    for k, logits_k in enumerate(binary_logits_list):
        evidence_k = relu_evidence(logits_k)  # [B, 2]
        alpha_k = evidence_k + 1
        S_k = torch.sum(alpha_k, dim=1, keepdim=True)
        probs_k = alpha_k / S_k
        yes_probs_list.append(probs_k[:, 1])
        evidence_yes_list.append(evidence_k[:, 1])
        if total_evidence_per_sample is None:
            total_evidence_per_sample = torch.sum(evidence_k, dim=1)
        else:
            total_evidence_per_sample = total_evidence_per_sample + torch.sum(evidence_k, dim=1)

    # 预测：哪个头的 “Yes” 概率最大
    yes_probs = torch.stack(yes_probs_list, dim=1)  # [B, K]
    max_yes_prob, preds = torch.max(yes_probs, dim=1)
    # 不确定性：1 - 最大 “Yes” 概率（无头置信时高，适合开放集）
    uncertainty = (1.0 - max_yes_prob).cpu().numpy()
    total_evidence = total_evidence_per_sample.cpu().numpy()
    preds = preds.cpu().numpy()

    batch_results = []
    for i in range(len(labels)):
        sample_result = {
            'true_label': labels[i].item(),
            'predicted_label': int(preds[i]),
            'uncertainty': float(uncertainty[i]),
            'total_evidence': float(total_evidence[i]),
        }
        for j in range(num_classes):
            sample_result[f'evidence_{j}'] = evidence_yes_list[j][i].item()
        batch_results.append(sample_result)
    return batch_results


def _analyze_ensemble_binary_edl(models, inputs, labels, K, device):
    """K 个独立二分类 EDL 模型：每个输出 [B, 2]，聚合 p_yes 与 u_k=2/S_k，与 test_edl_ensemble_binary 一致。"""
    p_yes_list = []
    u_k_list = []
    S_k_list = []
    for k in range(K):
        logits_k = models[k](inputs)
        evidence_k = relu_evidence(logits_k)
        alpha_k = evidence_k + 1
        S_k = torch.sum(alpha_k, dim=1)
        probs_k = alpha_k / S_k.unsqueeze(1)
        p_yes_list.append(probs_k[:, 1])
        u_k_list.append(2.0 / S_k)
        S_k_list.append(S_k)
    p_yes = torch.stack(p_yes_list, dim=1)
    u_k_stack = torch.stack(u_k_list, dim=1)
    S_k_stack = torch.stack(S_k_list, dim=1)
    uncertainty = u_k_stack.mean(dim=1).cpu().numpy()
    total_evidence = S_k_stack.sum(dim=1).cpu().numpy()
    preds = p_yes.argmax(dim=1).cpu().numpy()
    batch_results = []
    for i in range(len(labels)):
        sample_result = {
            'true_label': labels[i].item(),
            'predicted_label': int(preds[i]),
            'uncertainty': float(uncertainty[i]),
            'total_evidence': float(total_evidence[i]),
        }
        for j in range(K):
            sample_result[f'evidence_{j}'] = S_k_stack[i][j].item()
        batch_results.append(sample_result)
    return batch_results


def analyze_ensemble_binary_models(models, dataloader, K, device):
    """遍历数据集，用 K 个二分类模型计算不确定性与证据（与 test_edl_ensemble_binary 一致）。"""
    for m in models:
        m.eval()
    results = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            batch_results = _analyze_ensemble_binary_edl(models, inputs, labels, K, device)
            results.extend(batch_results)
    return pd.DataFrame(results)


def analyze_model(model, dataloader, num_classes, device, is_binary_model=False, use_main_head=False):
    """遍历数据集，计算不确定性和证据。支持标准 EDL 与多二分类头 EDL（可选主头）。"""
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            if is_binary_model and use_main_head:
                batch_results = _analyze_binary_edl_with_main(model, inputs, labels, num_classes, device)
            elif is_binary_model:
                batch_results = _analyze_binary_edl(model, inputs, labels, num_classes, device)
            else:
                batch_results = _analyze_standard_edl(model, inputs, labels, num_classes, device)
            results.extend(batch_results)

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


def plot_uncertainty_kde_id_ood(df, known_classes, output_dir, uncertainty_threshold=None):
    """
    绘制 ID（已知/可见类）与 OOD（未知/不可见类）测试样本的不确定性分布 KDE 图（不画阈值线）。
    """
    # 按真实标签划分为 ID 与 OOD
    id_mask = df['true_label'].isin(known_classes)
    id_uncertainties = df.loc[id_mask, 'uncertainty'].values
    ood_uncertainties = df.loc[~id_mask, 'uncertainty'].values

    if len(id_uncertainties) == 0:
        print("警告: 无 ID 样本，跳过 ID/OOD 不确定性 KDE 图")
        return
    if len(ood_uncertainties) == 0:
        print("警告: 无 OOD 样本，仅绘制 ID 分布")

    fig, ax = plt.subplots(figsize=(8, 5))
    # ID 与 OOD 的 KDE
    if len(id_uncertainties) > 0:
        sns.kdeplot(id_uncertainties, ax=ax, color='#1f77b4', fill=True, label='ID data / Seen condition', linewidth=2)
    if len(ood_uncertainties) > 0:
        sns.kdeplot(ood_uncertainties, ax=ax, color='#ff7f0e', fill=True, label='OOD data / Unseen fault', linewidth=2)
    ax.set_xlabel('Uncertainty', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('ID vs OOD 测试样本的不确定性分布', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(left=max(0, ax.get_xlim()[0]))
    plt.tight_layout()
    kde_path = os.path.join(output_dir, 'uncertainty_distribution_kde.png')
    plt.savefig(kde_path, dpi=300)
    plt.close()
    print(f"ID/OOD 不确定性 KDE 图已保存至: {kde_path}")


def main():
    parser = argparse.ArgumentParser(description='模型不确定性与证据量分析脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型权重文件路径（.pth文件）。如果未指定，则从配置文件的checkpoint_dir读取')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录。如果未指定，则自动从checkpoint路径推断')
    parser.add_argument('--uncertainty_threshold', type=float, default=None,
                        help='KDE 图中绘制的不确定性阈值；未指定时由 ID 样本 IQR 自动计算')
    parser.add_argument('--load_csv', type=str, default=None,
                        help='若提供，则从该 CSV 加载结果（如 test_edl_ensemble_binary 生成的 uncertainty_analysis.csv），仅绘制 KDE 图，不加载模型')
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

    # 支持从 CSV 仅绘图：不加载模型，直接使用已有 uncertainty 结果（如 test_edl_ensemble_binary 生成的 uncertainty_analysis.csv）
    if args.load_csv:
        if not os.path.exists(args.load_csv):
            print(f"错误: 未找到 CSV 文件 {args.load_csv}")
            return
        results_df = pd.read_csv(args.load_csv)
        if 'uncertainty' not in results_df.columns or 'true_label' not in results_df.columns:
            print("错误: CSV 需包含列 true_label, uncertainty")
            return
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.load_csv))
        os.makedirs(output_dir, exist_ok=True)
        # 仅绘制 ID vs OOD 不确定性 KDE 图（不绘制 evidence 等需模型前向的图）
        plot_uncertainty_kde_id_ood(
            results_df, known_classes, output_dir,
            uncertainty_threshold=args.uncertainty_threshold
        )
        print(f"KDE 图已生成（从 CSV）: {output_dir}/uncertainty_distribution_kde.png")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 确定 checkpoint 目录与模型路径
    num_classes = model_config.get('num_classes')
    if args.checkpoint:
        model_path = args.checkpoint
        checkpoint_dir = os.path.dirname(os.path.abspath(model_path))
    else:
        checkpoint_dir = train_config['checkpoint_dir']
        model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # 3. 准备数据（分析阶段统一需要）
    batch_size = train_config['batch_size']
    test_dataset = get_test_dataset(data_config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"测试集大小: {len(test_dataset)}")

    # 4. 自动选择加载方式：单模型 best_model.pth 或 集成二分类 model_0.pth..model_{K-1}.pth
    use_ensemble_binary = False
    if not os.path.exists(model_path):
        # 未找到 best_model.pth 时，检查是否为 EDL 集成二分类目录（含 model_0.pth）
        path_0 = os.path.join(checkpoint_dir, 'model_0.pth')
        if os.path.exists(path_0) and num_classes is not None and num_classes >= 1:
            use_ensemble_binary = True
            for k in range(num_classes):
                if not os.path.exists(os.path.join(checkpoint_dir, f'model_{k}.pth')):
                    use_ensemble_binary = False
                    break
            if use_ensemble_binary:
                print(f"未找到 {model_path}，检测到集成二分类目录（model_0.pth..model_{num_classes-1}.pth），将自动加载。")

    if use_ensemble_binary:
        # 加载 K 个二分类模型（与 test_edl_ensemble_binary 一致）
        K = num_classes
        backbone_type = model_config.get('type', 'ResNet18_2d')
        if 'Binary' in str(backbone_type):
            backbone_type = 'ResNet18_2d'
        models = []
        for k in range(K):
            path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
            model_k = get_model(backbone_type, num_classes=2).to(device)
            ckpt = torch.load(path_k, map_location=device, weights_only=True)
            state = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
            model_k.load_state_dict(state, strict=True)
            model_k.eval()
            models.append(model_k)
        print(f"已加载 K={K} 个二分类模型（backbone={backbone_type}）。")
        print("\n正在分析模型在测试集上的表现...")
        results_df = analyze_ensemble_binary_models(models, test_loader, K, device)
        print("分析完成。")
    else:
        # 单模型：best_model.pth 或多二分类头
        if not os.path.exists(model_path):
            print(f"错误: 在 {model_path} 未找到模型文件；且未在 {checkpoint_dir} 下找到完整 model_0.pth..model_{num_classes-1}.pth")
            return
        model_type_name = model_config.get('type', '')
        is_binary_model = 'Binary' in model_type_name or 'binary' in model_type_name
        use_main_head = model_config.get('use_main_head', True) if is_binary_model else False
        model_kwargs = {'num_classes': num_classes}
        if is_binary_model:
            model_kwargs['use_main_head'] = use_main_head
        model = get_model(model_config.get('type'), **model_kwargs).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"从 {model_path} 加载模型成功。")
        print(f"模型类型: {'多二分类头 EDL' if is_binary_model else '标准 K 类 EDL'}")
        if is_binary_model:
            print(f"主分类头: {'启用（不确定性基于主头）' if use_main_head else '关闭（不确定性基于二分类头）'}")
        print("\n正在分析模型在测试集上的表现...")
        results_df = analyze_model(
            model, test_loader, num_classes, device,
            is_binary_model=is_binary_model,
            use_main_head=use_main_head
        )
        print("分析完成。")

    # 5. 设置输出目录并保存结果
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 自动推断：在checkpoint目录下创建test子目录
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
    # ID vs OOD 不确定性密度图（论文图 11 风格）
    plot_uncertainty_kde_id_ood(
        results_df, known_classes, output_dir,
        uncertainty_threshold=args.uncertainty_threshold
    )
    print("所有任务完成！")

if __name__ == '__main__':
    main()
