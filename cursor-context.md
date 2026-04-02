# Cursor Context（NvF-EDL 故障诊断与开集）

本文档供新对话快速对齐：**代码在做什么、思路演变、已知坑、当前矛盾**。路径以仓库内 `NEW/ResNet2d_EDL_EnsembleBinary/` 为主。

---

## 1) 当前主线：NvF

- **结构**：`model_0`（正常 vs 全体已知故障）+ `model_1..K-1`（正常 vs 故障 i）。
- **训练**：`train_nvf.py` 预训练各子模型 → `train_nvf_joint_uncertainty.py` 联合微调。
- **测试**：`test_NvF.py`；闭集决策：`nvf_fusion` / `min_u_gated` / `min_u_gated_nf`（见 `train.test_infer`）。
- **EDL**：`evidence = relu(logits)`，`α = evidence + 1`，`S = Σα`，常用 **vacuity 式** `u = 2/S`（证据越多、S 越大、u 越小）。
- **联合微调**：`L_total = L_base + λ_unc·L_unc + (可选) λ_neg·L_neg`；`L_base` 对每个子模型各有一套二分类标签与 EDL 损失。

---

## 2) 关键抉择与思路演变（按主题）

- **专利/路线**：先不做导师侧「距离/度量」微调，优先「错专家不确定度」与测试端策略验证；距离类 loss 可与 `L_base` 叠加，属另一大类目标。
- **测试端**：已支持多种 `ood_score_mode`（含 `winner_pos_only`）、样本级 `id/ood_uncertainty_analysis_nvf.csv`、逐 epoch 测试与汇总 CSV。
- **NvF + `L_unc` 与 model0（重要）**  
  - **改前**：`y=j>0`（已知故障）时，`L_unc` 的「错专家要高 u」包含 **model0**（除真类列外所有列）。  
  - **后果**：与 `L_base` 要求 model0 在故障图上**明确判「非正常」**相冲突，易把 model0 在 ID 故障上拉「犹豫」，**OOD 更易被 model0 高置信判正常**。  
  - **拟议改法（与代码计划一致）**：`y>0` 时 **错专家 u 只作用于故障专家 `1..K-1` 中非真类**，**不要把 model0 列纳入 wrong_u**；配置开关默认关以保持旧实验可复现。
- **`L_base` vs `L_unc`**：`L_base` 已能学「各头在 y=0 / y=i 上该如何分配证据」；`L_unc` 额外塑造**同一 batch 上各头 u 的相对关系**以利融合/拒识。仅 `L_base` 联合微调 ≈ **多专家有监督继续训练**，不保证 OOD 一定变好。
- **OOD 指标解读（实测经验）**：可出现 **AUROC 高但 F1/MAR 很差**——排序尚可，但**固定阈值（如 0.5）与当前 `ood_score` 尺度不匹配**，或 `winner_only` 下 OOD 被**误收入闭集**时胜者 **u 仍很低**。优先 **阈值校准**（`--threshold_from_val` 或扫描）、再试 **`ood_score_mode`**（`mean_all` / `key_models` 等）。

---

## 3) 已完成代码变更（摘要）

### 3.1 `test_NvF.py`

- `winner_pos_only`、`compute_batch_ood_score`、`evaluate_and_save_outputs`、样本级 CSV 字段等（见 §7）。
- **逐 epoch**：`test_infer.epoch_save_sample_level_csv`（默认 `false`）控制 `epochs/<e>/` 是否写 `id_*.csv` / `ood_*.csv`；**根目录最终一次测试**仍默认写全量。
- `evaluate_and_save_outputs(..., save_sample_level_csv=True)` 可关闭上述两大 CSV。

### 3.2 `train_nvf_joint_uncertainty.py`

- `calc_wrong_pos_logit_loss`、`L_neg` 与 `joint_history` 中 `train_neg`/`val_neg` 等（历史已完成）。

### 3.3 配置

- `configs/bench_NvF_LaoDA_joint_uncertainty.yaml`：`test_infer` 含 `test_all_epochs`、`epoch_save_sample_level_csv` 等。

---

## 4) 逐 epoch 测试为什么慢（可关闭什么）

- **慢的主要原因**：`epoch 数 ×（整集前向 + 大量 matplotlib/seaborn 图）`。最耗：**K 子图二分类混淆矩阵**、**多张类×模型热力图（dpi 高 + 逐格标注）**、大张混淆矩阵。  
- **其次**：`id/ood` **逐样本 CSV** 循环写盘。  
- **关闭方式**：`test_infer.epoch_save_sample_level_csv: false`。汇总 `test_results_all_epochs_nvf.csv` 仍会写。

---

## 5) 当前遇到的问题（给新对话的 checklist）

1. **NvF 联合微调后 OOD**：部分实验 **model0 对 OOD 高概率正常**；根因分析指向 **`L_unc` 在 y=故障 时把 model0 当错专家拉高 u**，与 `L_base` 冲突 → **待实现「y>0 时 wrong_u 不含 model0」**（见 §2）。  
2. **OOD 检测**：**AUROC 与 MAR/F1 不一致**时，先查 **阈值** 与 **`ood_score_mode`**，勿只改训练。  
3. **测试耗时**：逐轮关大 CSV + 必要时减图或降 dpi。

---

## 6) 建议的后续实验顺序（简版）

1. 固定模型与 `ood_score_mode`，做 **阈值扫描**（或 `--threshold_from_val`）。  
2. NvF：实现并 A/B **`wrong_unc_exclude_model0_on_fault`**（命名以代码为准）。  
3. A/B：`L_unc` only vs `L_neg` only vs `L_unc + L_neg`（小 λ 起步）。  
4. 用样本级 CSV 看 **OOD 被谁 winner、pos_logit 尾部**。  
5. 再考虑 **损失归一化（EMA）** 与导师侧 **距离/原型** 项。

---

## 7) 历史记录（2026-03-27 对话整理，仍有效部分）

- 仅看类均值热力图会误导；**样本级 CSV** 才能定位「错专家高置信接收」。  
- `winner_pos_only` 提升可解释性，不保证天然完全分离。  
- `L_unc`（`u=2/S`）与 `L_neg`（压错专家 `pos_logit`）**互补**。  
- 早期仅 `L_base` 闭集好属正常；加联合项后需看 **尾部分布**。

---

更新时间：2026-03-28（已去掉 Fi-only 支线描述）
