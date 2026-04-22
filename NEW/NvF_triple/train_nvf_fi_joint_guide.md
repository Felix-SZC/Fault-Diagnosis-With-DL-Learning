# train_nvf_fi_joint.py 代码详尽解析指南

该脚本是 **Fi-only 架构的阶段二（Stage 2）联合微调程序**。它的任务是将阶段一训练好的多个独立专家模型聚合在一起，通过“三态监督”逻辑进行联合训练，使系统具备完整的故障分类与 OOD（离域）检测能力。

---

## 一、 整体逻辑概述

1.  **加载专家库**：从阶段一的目录中加载 $K-1$ 个预训练好的 `model_k.pth`。
2.  **锁定/开启层**：通常锁定卷积特征提取层，只微调全连接层（FC 层），以保留专家在阶段一学到的局部特征。
3.  **三态监督训练**：这是核心。对于每一个专家头 $k$，在训练时不仅要看它自己的故障，还要看“别人的故障”。
    *   **正常**样本 -> 目标 `[1,0]`
    *   **故障 k** 样本 -> 目标 `[0,1]`
    *   **其他故障**（或噪声）样本 -> 目标 `[0,0]`（关键：强制产生高不确定性）
4.  **白噪声注入**：动态生成各种类型的随机噪声，模拟未知工况，强迫模型对这些样本输出 `[0,0]`。

---

## 二、 逐段代码深度解析

### 1. 三态目标构造逻辑 (`triple_target_batch` - 位于 common 库)
虽然这个函数在 `nvf_fi_triple_common.py` 中，但它是本脚本的灵魂。
```python
# 逻辑伪代码：
if label == 0: target = [1,0]      # 正常
elif label == k: target = [0,1]    # 我的故障
else: target = [0,0]               # 别人的故障/噪声 -> 产生 Vacuity 不确定性
```
*   **解析**：这种设计让每个专家在看到不属于自己管辖范围的故障时，学会“保持沉默”（即输出 `[0,0]` 这种低证据状态），而不是误报。

### 2. 多头 Loss 聚合 (`calc_fi_triple_base_loss`)
```python
def calc_fi_triple_base_loss(...):
    loss_total = 0.0
    for idx, k in enumerate(range(1, K)):
        logits = logits_list[idx]
        y_triple = triple_target_batch(labels, k, ...) # 构造当前头 k 的三态目标
        sw = triple_sample_weights(...)               # 构造当前头 k 的样本权重
        loss_k = fi_triple_edl_mse_loss(...)           # 计算单头 EDL Loss
        loss_total += loss_k
    return loss_total / (K - 1) # 取所有专家头的平均损失
```
*   **解析**：
    *   每个 Batch 数据会同时流过所有 $K-1$ 个专家模型。
    *   每个专家模型根据自己和当前样本标签的关系，计算独立的三态 Loss。
    *   最终通过反向传播同时更新所有专家的权重。

### 3. 可训练参数设置 (`set_trainable_for_joint`)
```python
if head_only:
    for _, p in model.named_parameters():
        p.requires_grad = False # 先冻结全部
    for n, p in model.named_parameters():
        if "fc" in n: # 只开启 FC 层（或特定层如 fc2）
            p.requires_grad = True
```
*   **解析**：阶段二通常不需要从头训练特征提取器。保持特征层不动，只调整分类决策头，可以有效防止模型过拟合到特定的小样本集上。

### 4. 融合准确率监控 (`fi_fusion_correct_in_batch`)
```python
def fi_fusion_correct_in_batch(...):
    # ... 收集所有头的 p0, p1, u ...
    preds, _ = fi_triple_preds_and_ood(p0_stack, p1_stack, u_stack, tau_n, tau_f)
    return int((preds == labels).sum().item())
```
*   **解析**：在训练阶段，脚本不仅仅看 Loss，还通过模拟真实的“多专家投票/融合”过程来计算准确率。这能让你直接看到联合微调对最终诊断效果的影响。

### 5. 白噪声与多样化 OOD 注入（关键增强）
```python
if wn_multi_type:
    noise_types = ["white", "uniform", "impulse", "sine", "pink", "zero", "const", "mixed"]
    # ... 生成各种奇异信号 ...
    noise_labels = torch.full((n_noise,), K) # 标签设为 K（代表未知）
    inputs = torch.cat([inputs, noise])
```
*   **深度解析**：
    *   **为什么要注入？** 如果只用已知故障训练，模型可能在遇到完全没见过的信号（如正弦波、脉冲）时产生误判。
    *   **效果**：通过强迫专家对这些奇奇怪怪的信号输出 `[0,0]`，模型学会了在遇到“不像任何已知故障”的信号时，大幅提高不确定度（Uncertainty），从而被 `τ_OOD` 阈值拦截。

---

## 三、 总结：阶段二的价值

1.  **解决“越权”问题**：阶段一训练的专家只见过正常和自己的故障，它们在见到“别人的故障”时会乱猜。阶段二通过 `[0,0]` 监督，教会专家保持谦虚。
2.  **构建不确定性边界**：通过噪声注入，显式地在已知类别之外划定了一道“不确定性长城”。
3.  **系统集成**：这是将多个孤立模型转化为一个协同工作的“信任系统”的必经之路。
