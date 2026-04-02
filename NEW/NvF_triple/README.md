# NvF_triple 周报方法归档

本目录用于归档 2026-03-29 周报对应的 Fi-only 三态两阶段方案。

## 文件索引

- 训练阶段一: `train_nvf_fi_triple.py`
- 训练阶段二: `train_nvf_fi_joint.py`
- 测试脚本: `test_NvF_fi_triple.py`
- 三态公共逻辑: `nvf_fi_triple_common.py`
- 阶段一配置: `configs/bench_NvF_LaoDA_fi_triple.yaml`
- 阶段二配置: `configs/bench_NvF_LaoDA_fi_triple_joint.yaml`
- 依赖目录: `models/`, `common/`

## 约定

- 模型类型保持为 `LaoDA`（见两个 yaml 的 `model.type`）。
- 数据集仍使用 `NEW/Kong_code&&data` 体系下路径，你可在 yaml 的 `data.data_dir` 中按需切换子目录。
- 该归档是复制版本，不改算法逻辑；原项目目录保持不变。

## 运行方式

建议在本目录内执行，命令如下：

```bash
python train_nvf_fi_triple.py --config configs/bench_NvF_LaoDA_fi_triple.yaml
python train_nvf_fi_joint.py --config configs/bench_NvF_LaoDA_fi_triple_joint.yaml
python test_NvF_fi_triple.py --config configs/bench_NvF_LaoDA_fi_triple_joint.yaml
```

若需指定测试 epoch，可用：

```bash
python test_NvF_fi_triple.py --config configs/bench_NvF_LaoDA_fi_triple_joint.yaml --test_epochs 299
```

