# train_nvf_fi_triple.py 代码详尽解析指南

该脚本是 **Fi-only 架构的阶段一（Stage 1）训练程序**。它的核心任务是为每一个已知的故障类别训练一个“二分类专家模型”，每个专家只负责区分“正常”和“它自己对应的那个故障”。

---

## 一、 整体逻辑概述

1.  **多专家循环**：脚本不会一次性训练一个多分类模型，而是遍历 $k=1 \dots K-1$（$K$ 为已知类总数），依次训练 $K-1$ 个子模型。
2.  **数据子集化**：对于第 $k$ 个专家，从完整数据集中只提取出“类别 0（正常）”和“类别 k（该专家负责的故障）”的数据。
3.  **EDL 二分类**：每个专家模型输出 2 维证据（Evidence），使用证据深度学习（EDL）损失函数进行训练，旨在输出高证据（低不确定度）的判断。
4.  **模型保存**：保存每个专家在验证集上表现最好的权重（`model_k.pth`）和最后一轮的权重（`model_last_k.pth`）。

---

## 二、 逐段代码深度解析

### 1. 数据集加载与过滤 (`get_dataset`)
```python
def get_dataset(data_config, split="train"):
    # ... 略过路径读取 ...
    return NpyPackDataset1D(
        # ... 传入 known_classes, unknown_classes 等 ...
    )
```
*   **解析**：当前 LaoDA-only 流程会固定走 1D 数据集类（`NpyPackDataset1D`）。它首先加载完整的已知类数据。

### 2. 专家模型大循环 (`run_fi_only_training`)
```python
for k in range(1, K):
    print(f"训练二分类模型 k = {k} / {K - 1}（正常 vs 第 {k} 类故障）")
```
*   **关键点**：这是脚本最重要的外层循环。`k` 从 1 开始，因为 `0` 永远代表正常。每个 `k` 都会启动一次完整的训练流程。

### 3. 数据子集构建（核心逻辑）
```python
train_indices = np.where((train_dataset.y == 0) | (train_dataset.y == k))[0]
train_subset = Subset(train_dataset, train_indices)
```
*   **深度解析**：
    *   `train_dataset.y == 0`：选中正常样本。
    *   `train_dataset.y == k`：选中当前专家 $k$ 负责的特定故障样本。
    *   使用 `Subset` 包装后，训练加载器（DataLoader）只会读取这两类数据。这保证了模型 $k$ 是一个**局部专家**。

### 4. 损失函数选择
```python
edl_loss_type = train_config.get("edl_loss_type", "mse")
if edl_loss_type == "mse":
    criterion = edl_mse_loss
# ...
```
*   **解析**：默认使用 `edl_mse_loss`。EDL 损失与交叉熵不同，它最小化的是 Dirichlet 分布与目标 one-hot 之间的平方误差，并包含一个 KL 散度项。

### 5. 训练内循环与二分类标签转换
```python
for inputs, labels in train_loader:
    # labels 原本是原始类别索引 (如 0, 1, 2... k ... K-1)
    binary_labels = (labels == k).long() 
    # 转换后：
    #   原标签为 k 的样本 -> binary_labels = 1 (正类)
    #   原标签为 0 的样本 -> binary_labels = 0 (负类)
```
*   **解析**：这是实现二分类的关键转换。在专家 $k$ 的视角下，只有“我的故障（正类）”和“正常（负类）”。

### 6. 类别不平衡处理 (Sample Weighting)
```python
n_pos = (binary_labels == 1).sum().item()
n_neg = (binary_labels == 0).sum().item()
pos_w = n_neg / max(n_pos, 1) # 计算正类权重
sample_w = torch.where(binary_labels == 1, torch.tensor(pos_w...), torch.ones(...))
```
*   **深度解析**：由于正常样本通常远多于单一故障样本，代码动态计算 `pos_w`。如果正常样本是故障样本的 10 倍，则故障样本的损失权重会被设为 10，防止模型倾向于预测“正常”。

### 7. EDL 损失计算与退火
```python
loss = criterion(
    logits, y_binary, epoch, 2, annealing_step, device, sample_weight=sample_w, kl_weight=kl_weight
)
```
*   **解析**：
    *   `annealing_step`：退火步数。EDL 训练初期通常会抑制 KL 散度项，让模型先学会分类，后期再增强不确定性惩罚。
    *   `y_binary`：是长度为 2 的 one-hot 向量。

### 8. 不确定度与预测计算
```python
with torch.no_grad():
    evidence = relu_evidence(logits) # 将 logits 映射为非负证据
    alpha = evidence + 1             # Dirichlet 参数
    S = torch.sum(alpha, dim=1, keepdim=True) # 总强度
    probs = alpha / S                # 类别概率
    preds = (probs[:, 1] > 0.5).long() # 简单的阈值判定
```
*   **解析**：这是典型的证据推理过程。`alpha` 越大，证据越充足。概率 `probs` 是各分量占总强度的比例。

### 9. 模型保存逻辑
```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    path_k = os.path.join(checkpoint_dir, f"model_{k}.pth")
    torch.save({"state_dict": model.state_dict(), ...}, path_k)

# 始终保存最后一轮权重
path_last_k = os.path.join(checkpoint_dir, f"model_last_{k}.pth")
torch.save({"state_dict": model.state_dict(), ...}, path_last_k)
```
*   **解析**：
    *   `model_k.pth`：代表在该专家擅长的领域内（正常 vs 故障k），验证准确率最高的时刻。
    *   `model_last_k.pth`：代表训练完全结束后的状态。

---

## 三、 总结：为什么这样写？

1.  **解耦训练**：如果新增一种故障，只需要运行脚本训练对应的 `model_new.pth`，不需要重训所有老模型。
2.  **局部专注**：专家 $k$ 只需要解决最简单的二分类问题，这比一次性处理十几个类别的多分类要容易得多。
3.  **不确定性基础**：这种独立训练模式为阶段二（Stage 2）的“三态监督”打下了基础，因为每个模型在阶段一已经明确了自己的职责范围。
