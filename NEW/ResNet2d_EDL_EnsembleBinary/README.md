# ResNet2d + EDL 集成二分类（NvF / OvR）

**请在 `NEW/ResNet2d_EDL_EnsembleBinary` 目录下执行命令**（相对路径 `data.data_dir`、`checkpoint_dir`、`pretrained_multiclass_ckpt` 等均按此工作目录解析）。

## 目录说明

| 路径 | 说明 |
|------|------|
| `configs/` | 全部实验 YAML（bench / turbine 等） |
| `common/` | 数据、损失、工具函数 |
| `models/` | 网络结构（如 `LaoDA`、`ResNet18_2d`） |
| `tools/` | 辅助脚本（如 OOD 分析） |
| `legacy/` | 旧版单入口 `train.py` / `test.py`（保留兼容） |
| `checkpoints/` | 训练输出（默认由 yaml 指定，可 gitignore） |

## 常用命令

**NvF（Normal vs Fault）**

```bash
python train_nvf.py --config configs/bench_NvF_LaoDA.yaml
python test_NvF.py --config configs/bench_NvF_LaoDA.yaml
```

**OvR（One-vs-Rest）**

```bash
python train_ovr.py --config configs/bench_OvR_LaoDA.yaml
python test_ovr.py --config configs/bench_OvR_LaoDA.yaml
```

**冻结 Backbone 两阶段（LaoDA 闭集预训练 → NvF 只训头）**

```bash
python train_lao_da_closed_pretrain.py --config configs/bench_LaoDA_closed_pretrain.yaml
python train_nvf_frozen_backbone.py --config configs/bench_NvF_LaoDA_frozen_backbone.yaml
python test_NvF.py --config configs/bench_NvF_LaoDA_frozen_backbone.yaml
```

**只评估闭集预训练 `pretrained_full.pth`（K 类混淆矩阵）**

```bash
python tools/test_lao_da_closed_pretrain.py --config configs/bench_LaoDA_closed_pretrain.yaml
# 或显式指定权重与输出目录
python tools/test_lao_da_closed_pretrain.py --config configs/bench_LaoDA_closed_pretrain.yaml ^
  --ckpt checkpoints/LaoDA/closed_pretrain/seed2/pretrained_full.pth ^
  --output_dir checkpoints/LaoDA/closed_pretrain/seed2/test_eval
```
（Linux/macOS 将 `^` 换为 `\`）

**辅助**

```bash
python tools/analyze_ood_per_model.py --config configs/bench_OvR_LaoDA.yaml
```

## 说明

- 主流程入口为根目录下的 `train_nvf.py`、`test_NvF.py`、`train_ovr.py`、`test_ovr.py`；`legacy/` 内为历史脚本。
- 更换数据集或超参时，优先复制 `configs/` 下 yaml 再改路径与 `train` 段。

### `test_NvF.py`：`train.test_infer.class_decision`

- **`nvf_fusion`（默认）**：闭集类别由 `build_nvf_final_probs_and_winner` 融合后的 `final_probs.argmax` 决定；OOD 分数由 `ood_score_mode`（如 `mean_all` / `key_models` / `winner_only`）等决定。
- **`min_u_gated`**：两阶段——模型 0 的 `p_yes>0.5` 判正常（类 0），`OOD=u_0`。否则在模型 `1..K-1` 中**仅当** `p_yes_k>0.5`（该子模型判故障）的索引上取 **`u_k=2/S_k` 最小** 者为故障类（`argmin` 平局取第一个最小值），`OOD` 为该子模型 `u_k`。**若无一子模型** `p_yes>0.5`：闭集类取 `1..K-1` 上 **`p_yes` 最大** 者，`OOD=1.0`（满分不确定度）。`ood_score_mode` / `ood_lambda` 不参与；`fault_fusion` / `fusion_tau` 不参与闭集类别决策。
- 可在 yaml 的 `train.test_infer.class_decision` 中设置，或命令行：`python test_NvF.py --config ... --class_decision min_u_gated`。
