import argparse
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# 设置 matplotlib 中文字体，解决中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config
from common.utils.data_loader import RawSignalDataset, NpyIndexDataset
from models import get_model

def get_dataset(data_config, split='train', filter_classes=None):
    """根据配置动态加载数据集 (visualize_features 版本)"""
    data_type = data_config.get('type')
    if data_type is None:
        raise ValueError("配置文件中必须指定 data.type")

    if data_type == 'raw_signal':
        base_dir = data_config.get('raw_signal_output_dir')
        if base_dir is None:
            raise ValueError("配置文件中缺少 'raw_signal_output_dir'")
        split_dir = os.path.join(base_dir, split)
        return RawSignalDataset(split_dir=split_dir, filter_classes=filter_classes)
    elif data_type == 'wpt':
        base_dir = data_config.get('wpt_output_dir')
        if base_dir is None:
            raise ValueError("配置文件中缺少 'wpt_output_dir'")
        split_dir = os.path.join(base_dir, split)
        return NpyIndexDataset(split_dir=split_dir, filter_classes=filter_classes)
    else:
        raise ValueError(f"visualize_features.py 目前仅支持 'raw_signal' 和 'wpt' 数据类型，当前为: {data_type}")

def extract_features(model, dataloader, device, return_logits=True):
    """
    从数据集中提取特征和 logits
    
    参数:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        return_logits: 是否同时返回 logits
    
    返回:
        features: (N, feature_dim) 的特征矩阵
        logits: (N, num_classes) 的 logits 矩阵（如果 return_logits=True）
        labels: (N,) 的标签数组
        original_labels: (N,) 的原始标签数组（如果数据集有原始标签映射）
    """
    model.eval()
    all_features = []
    all_logits = []
    all_labels = []
    all_original_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            logits, features = model(inputs, return_features=True)
            
            all_features.append(features.cpu().numpy())
            if return_logits:
                all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if return_logits:
        logits = np.concatenate(all_logits, axis=0)
        return features, logits, labels
    else:
        return features, labels

def visualize_tsne(features, labels, known_classes, unknown_classes, 
                   title="t-SNE Visualization", save_path=None, 
                   label_names=None, original_labels=None, perplexity=30, max_iter=1000):
    """
    使用 t-SNE 可视化特征分布
    
    参数:
        features: (N, feature_dim) 特征矩阵
        labels: (N,) 标签数组（可能是重映射后的）
        known_classes: 已知类别列表（原始标签）
        unknown_classes: 未知类别列表（原始标签）
        title: 图表标题
        save_path: 保存路径
        label_names: 类别名称字典 {label: name}
        original_labels: 原始标签数组（如果提供了，使用原始标签进行可视化）
    """
    # 使用原始标签（如果提供）或重映射后的标签
    if original_labels is not None:
        vis_labels = original_labels
    else:
        vis_labels = labels
    
    # 判断每个样本是已知类还是未知类
    is_known = np.array([label in known_classes for label in vis_labels])
    is_unknown = np.array([label in unknown_classes for label in vis_labels])
    
    print(f"已知类样本数: {np.sum(is_known)}")
    print(f"未知类样本数: {np.sum(is_unknown)}")
    
    # 执行 t-SNE 降维
    print("正在执行 t-SNE 降维...")
    # 兼容新旧版本的 scikit-learn: n_iter 在新版本中改为 max_iter
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=max_iter)
    except TypeError:
        # 旧版本使用 n_iter
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=max_iter)
    features_2d = tsne.fit_transform(features)
    
    # 创建图表
    plt.figure(figsize=(14, 6))
    
    # 左图：区分已知类和未知类
    ax1 = plt.subplot(1, 2, 1)
    
    # 绘制已知类
    known_mask = is_known
    if np.any(known_mask):
        scatter1 = ax1.scatter(features_2d[known_mask, 0], features_2d[known_mask, 1], 
                              c='blue', alpha=0.6, s=20, label='已知类', edgecolors='none')
    
    # 绘制未知类
    unknown_mask = is_unknown
    if np.any(unknown_mask):
        scatter2 = ax1.scatter(features_2d[unknown_mask, 0], features_2d[unknown_mask, 1], 
                              c='red', alpha=0.6, s=20, label='未知类', edgecolors='none')
    
    ax1.set_xlabel('t-SNE 维度 1', fontsize=12)
    ax1.set_ylabel('t-SNE 维度 2', fontsize=12)
    ax1.set_title(f'{title}\n(已知类 vs 未知类)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：按具体类别着色
    ax2 = plt.subplot(1, 2, 2)
    
    # 为每个类别分配颜色
    all_classes = sorted(set(vis_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_classes)))
    class_color_map = {cls: colors[i] for i, cls in enumerate(all_classes)}
    
    # 绘制每个类别
    for cls in all_classes:
        mask = vis_labels == cls
        if np.any(mask):
            label_name = label_names.get(cls, f'类别 {cls}') if label_names else f'类别 {cls}'
            # 如果是未知类，添加标记
            if cls in unknown_classes:
                label_name += ' (未知)'
            ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[class_color_map[cls]], alpha=0.6, s=20, 
                       label=label_name, edgecolors='none')
    
    ax2.set_xlabel('t-SNE 维度 1', fontsize=12)
    ax2.set_ylabel('t-SNE 维度 2', fontsize=12)
    ax2.set_title(f'{title}\n(按类别着色)', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    
    plt.show()

def visualize_logits_distribution(logits, labels, known_classes, unknown_classes,
                                  title="Logits 分布", save_path=None, original_labels=None):
    """
    可视化 Logits 的分布情况
    
    参数:
        logits: (N, num_classes) logits 矩阵
        labels: (N,) 标签数组
        known_classes: 已知类别列表
        unknown_classes: 未知类别列表
        title: 图表标题
        save_path: 保存路径
        original_labels: 原始标签数组
    """
    if original_labels is not None:
        vis_labels = original_labels
    else:
        vis_labels = labels
    
    is_known = np.array([label in known_classes for label in vis_labels])
    is_unknown = np.array([label in unknown_classes for label in vis_labels])
    
    # 计算每个样本的最大 logit 值
    max_logits = np.max(logits, axis=1)
    # 计算每个样本的 logit 熵（不确定性）
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：最大 logit 值分布
    ax1 = axes[0]
    if np.any(is_known):
        ax1.hist(max_logits[is_known], bins=50, alpha=0.6, label='已知类', color='blue', density=True)
    if np.any(is_unknown):
        ax1.hist(max_logits[is_unknown], bins=50, alpha=0.6, label='未知类', color='red', density=True)
    ax1.set_xlabel('最大 Logit 值', fontsize=12)
    ax1.set_ylabel('密度', fontsize=12)
    ax1.set_title('最大 Logit 值分布', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：熵分布
    ax2 = axes[1]
    if np.any(is_known):
        ax2.hist(entropy[is_known], bins=50, alpha=0.6, label='已知类', color='blue', density=True)
    if np.any(is_unknown):
        ax2.hist(entropy[is_unknown], bins=50, alpha=0.6, label='未知类', color='red', density=True)
    ax2.set_xlabel('预测熵 (Entropy)', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.set_title('预测不确定性分布', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Logits 分布图已保存至: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='特征可视化脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='输出目录（默认使用 checkpoint_dir）')
    parser.add_argument('--perplexity', type=int, default=30, 
                       help='t-SNE 的 perplexity 参数')
    parser.add_argument('--max_iter', type=int, default=1000, 
                       help='t-SNE 的迭代次数（新版本 scikit-learn 使用 max_iter）')
    args = parser.parse_args()
    
    # 1. 加载配置
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']
    
    known_classes = data_config['openset']['known_classes']
    unknown_classes = data_config['openset']['unknown_classes']
    
    # 类别名称映射（根据你的数据文件命名）
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
    model = get_model(model_config.get('type'), num_classes=model_config.get('num_classes')).to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), weights_only=True))
    model.eval()
    print("模型加载成功。")
    
    # 3. 准备数据
    batch_size = train_config['batch_size']
    
    # 训练集（仅已知类）
    train_dataset = get_dataset(data_config, split='train', filter_classes=known_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # 测试集（包含所有类别，不筛选）
    test_dataset = get_dataset(data_config, split='test', filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 4. 提取特征
    print("\n正在提取训练集特征...")
    train_features, train_logits, train_labels = extract_features(model, train_loader, device)
    print(f"训练集特征形状: {train_features.shape}")
    print(f"训练集 Logits 形状: {train_logits.shape}")
    
    print("\n正在提取测试集特征...")
    test_features, test_logits, test_labels = extract_features(model, test_loader, device)
    print(f"测试集特征形状: {test_features.shape}")
    print(f"测试集 Logits 形状: {test_logits.shape}")
    
    # 获取测试集的原始标签（如果数据集有 label_map）
    test_original_labels = None
    if hasattr(test_dataset, 'label_map') and test_dataset.label_map:
        # 反向映射：从重映射后的标签恢复到原始标签
        reverse_map = {v: k for k, v in test_dataset.label_map.items()}
        test_original_labels = np.array([reverse_map.get(label, label) for label in test_labels])
    else:
        test_original_labels = test_labels
    
    # 5. 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(checkpoint_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. 可视化训练集特征（仅已知类）
    print("\n正在可视化训练集特征分布...")
    train_original_labels = None
    if hasattr(train_dataset, 'label_map') and train_dataset.label_map:
        reverse_map = {v: k for k, v in train_dataset.label_map.items()}
        train_original_labels = np.array([reverse_map.get(label, label) for label in train_labels])
    else:
        train_original_labels = train_labels
    
    visualize_tsne(
        train_features, train_labels, known_classes, unknown_classes,
        title="训练集特征分布 (Features Space)",
        save_path=os.path.join(output_dir, 'train_features_tsne.png'),
        label_names=label_names,
        original_labels=train_original_labels,
        perplexity=args.perplexity,
        max_iter=args.max_iter
    )
    
    # 7. 可视化测试集特征（包含已知类和未知类）
    print("\n正在可视化测试集特征分布...")
    visualize_tsne(
        test_features, test_labels, known_classes, unknown_classes,
        title="测试集特征分布 (Features Space)",
        save_path=os.path.join(output_dir, 'test_features_tsne.png'),
        label_names=label_names,
        original_labels=test_original_labels,
        perplexity=args.perplexity,
        max_iter=args.max_iter
    )
    
    # 8. 可视化 Logits 空间（如果维度不太高）
    if test_logits.shape[1] <= 20:  # 只对类别数不太多的情况做 t-SNE
        print("\n正在可视化测试集 Logits 分布...")
        visualize_tsne(
            test_logits, test_labels, known_classes, unknown_classes,
            title="测试集 Logits 分布 (Logits Space)",
            save_path=os.path.join(output_dir, 'test_logits_tsne.png'),
            label_names=label_names,
            original_labels=test_original_labels,
            perplexity=args.perplexity,
            max_iter=args.max_iter
        )
    
    # 9. 可视化 Logits 统计分布
    print("\n正在可视化 Logits 统计分布...")
    visualize_logits_distribution(
        test_logits, test_labels, known_classes, unknown_classes,
        title="测试集 Logits 统计分布",
        save_path=os.path.join(output_dir, 'test_logits_distribution.png'),
        original_labels=test_original_labels
    )
    
    # 10. 合并训练集和测试集进行对比可视化
    print("\n正在生成合并可视化（训练集+测试集）...")
    all_features = np.vstack([train_features, test_features])
    all_labels_combined = np.concatenate([train_original_labels, test_original_labels])
    all_is_known = np.array([label in known_classes for label in all_labels_combined])
    all_is_unknown = np.array([label in unknown_classes for label in all_labels_combined])
    
    # 创建标记数组：0=训练集已知类, 1=测试集已知类, 2=测试集未知类
    split_markers = np.concatenate([
        np.zeros(len(train_original_labels)),  # 训练集
        np.ones(len(test_original_labels)) * (1 + all_is_unknown[len(train_original_labels):].astype(int))
    ])
    
    visualize_tsne(
        all_features, all_labels_combined, known_classes, unknown_classes,
        title="合并特征分布 (训练集+测试集)",
        save_path=os.path.join(output_dir, 'combined_features_tsne.png'),
        label_names=label_names,
        original_labels=all_labels_combined,
        perplexity=args.perplexity,
        max_iter=args.max_iter
    )
    
    print(f"\n所有可视化结果已保存至: {output_dir}")

if __name__ == '__main__':
    main()

