# Fi-only 阶段二：全闭集联合微调 K-1 个头，三态监督 [1,0]/[0,1]/[0,0]，仅 L_base（Fi 三态 EDL-MSE）
# 从 train_nvf_joint_uncertainty.py 精简复制；不修改原 joint 脚本

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch

# 需在 import pyplot 之前决定后端：--live-plot 用 TkAgg 实时窗口，否则 Agg 仅写文件
_LIVE_PLOT = "--live-plot" in sys.argv
import matplotlib

matplotlib.use("TkAgg" if _LIVE_PLOT else "Agg")
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.utils.helpers import load_config, save_experiment_info
from common.utils.data_loader import NpyPackDataset
from common.utils.data_loader_1d import NpyPackDataset1D
from common.edl_losses import relu_evidence
from common.utils.lao_da_pretrained import summarize_trainable_params
from models import get_model

from nvf_fi_triple_common import (
    fi_triple_edl_mse_loss,
    fi_triple_preds_and_ood,
    triple_sample_weights,
    triple_target_batch,
)


def _configure_cjk_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save_fi_stage2_flowchart(save_path: str) -> None:
    """阶段二训练流程示意（与脚本逻辑一致），存 PNG。"""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _configure_cjk_font()
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Fi-only 阶段二：联合微调训练流程（无 model0）", fontsize=13, pad=12)

    def node(cx, cy, w, h, text, face="#e8f4fc"):
        x, y = cx - w / 2, cy - h / 2
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                linewidth=1.1,
                edgecolor="#333333",
                facecolor=face,
            )
        )
        ax.text(cx, cy, text, ha="center", va="center", fontsize=9, linespacing=1.25)

    def arr(x0, y0, x1, y1):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                color="#444444",
                lw=1.2,
            )
        )

    y = 11.15
    dy = 1.05
    node(4.5, y, 7.8, 0.72, "读取 YAML：K、数据、阶段一 input_checkpoint_dir、超参")
    y -= dy
    arr(4.5, y + dy / 2 + 0.36, 4.5, y + 0.36)
    node(4.5, y, 7.8, 0.72, "加载 model_1 … model_{K-1}，按 head_only / train_fc1 设可训练参数")
    y -= dy
    arr(4.5, y + dy / 2 + 0.36, 4.5, y + 0.36)
    node(4.5, y, 7.8, 0.72, "构建闭集 train / test DataLoader（仅 known_classes）")
    y -= dy
    arr(4.5, y + dy / 2 + 0.36, 4.5, y + 0.36)
    node(
        4.5,
        y,
        7.8,
        1.0,
        "每个 epoch：\n训练模式 → 逐 batch 各头前向 → 三态 y + sample_weight\n→ fi_triple_edl_mse_loss 对 K-1 头取平均 → backward + step",
        face="#fff3e0",
    )
    y -= dy + 0.15
    arr(4.5, y + dy / 2 + 0.5, 4.5, y + 0.36)
    node(
        4.5,
        y,
        7.8,
        0.88,
        "同步累计：训练集加权平均 loss；Fi 融合闭集准确率（与 test 相同阈值）",
        face="#e8f5e9",
    )
    y -= dy + 0.05
    arr(4.5, y + dy / 2 + 0.44, 4.5, y + 0.36)
    node(
        4.5,
        y,
        7.8,
        1.0,
        "验证集 no_grad：同上 loss + 准确率\n→ 记录 hist；若 val_loss 更优则保存各 model_k.pth",
        face="#fce4ec",
    )
    y -= dy + 0.15
    arr(4.5, y + dy / 2 + 0.5, 4.5, y + 0.36)
    node(4.5, y, 7.8, 0.72, "若 save_every_epoch：写入 epochs/<e>/model_*.pth")
    y -= dy
    arr(4.5, y + dy / 2 + 0.36, 4.5, y + 0.36)
    node(4.5, y, 7.8, 0.72, "每个 epoch 更新 training_curves_fi_joint.png 与 joint_history.csv（实时落盘）")
    y -= dy
    arr(4.5, y + dy / 2 + 0.36, 4.5, y + 0.36)
    node(4.5, y, 7.8, 0.72, "训练结束：再写最终权重与实验信息", face="#ede7f6")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fi_fusion_correct_in_batch(
    logits_list: list, labels: torch.Tensor, device: torch.device, tau_n: float, tau_f: float
) -> int:
    """Fi 融合闭集预测与 labels 一致的样本数（logits 可无 grad）。"""
    labels = labels.to(device).long()
    p0s, p1s, us = [], [], []
    for logits in logits_list:
        ev = relu_evidence(logits)
        alpha = ev + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        p0s.append(probs[:, 0])
        p1s.append(probs[:, 1])
        us.append(2.0 / torch.sum(alpha, dim=1))
    p0_stack = torch.stack(p0s, dim=1)
    p1_stack = torch.stack(p1s, dim=1)
    u_stack = torch.stack(us, dim=1)
    preds, _ = fi_triple_preds_and_ood(p0_stack, p1_stack, u_stack, tau_n, tau_f)
    return int((preds == labels).sum().item())


def save_training_curves_fi_joint(
    path: str,
    hist_train: list,
    hist_val: list,
    hist_train_acc: list,
    hist_val_acc: list,
    fig_axes=None,
) -> None:
    """双图：Loss + Accuracy(%)，风格对齐 train_nvf_fi_triple。"""
    _configure_cjk_font()
    epochs = np.arange(1, len(hist_train) + 1)
    if fig_axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, ax1, ax2 = fig_axes
        ax1.clear()
        ax2.clear()
    fig.suptitle("Fi-only 阶段二 Joint：K-1 头三态联合微调")
    ax1.plot(epochs, hist_train, label="Train")
    ax1.plot(epochs, hist_val, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, hist_train_acc, label="Train")
    ax2.plot(epochs, hist_val_acc, label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if fig_axes is not None:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    else:
        plt.close(fig)


def get_dataset(data_config, model_type, split="train"):
    data_dir = data_config.get("data_dir")
    openset_config = data_config.get("openset", {})
    known_classes = openset_config.get("known_classes")
    unknown_classes = openset_config.get("unknown_classes", [])
    augment_config = data_config.get("augmentation", {})
    dataset_cls = NpyPackDataset1D if model_type == "LaoDA" else NpyPackDataset
    return dataset_cls(
        data_dir=data_dir,
        split=split,
        filter_classes=known_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        augment_config=augment_config,
    )


def set_trainable_for_joint(model, backbone_type, head_only=True, train_fc1=False):
    if not head_only:
        for _, p in model.named_parameters():
            p.requires_grad = True
        return
    for _, p in model.named_parameters():
        p.requires_grad = False
    if backbone_type == "LaoDA":
        for n, p in model.named_parameters():
            if n.startswith("fc2."):
                p.requires_grad = True
            elif train_fc1 and n.startswith("fc1."):
                p.requires_grad = True
        return
    if hasattr(model, "fc"):
        for n, p in model.named_parameters():
            if n.startswith("fc."):
                p.requires_grad = True
        return
    for n, p in model.named_parameters():
        if "fc" in n:
            p.requires_grad = True


def load_models_fi(checkpoint_dir, K, backbone_type, device):
    models = []
    for k in range(1, K):
        path_k = os.path.join(checkpoint_dir, f"model_{k}.pth")
        if not os.path.exists(path_k):
            raise FileNotFoundError(f"未找到 Fi-only 权重: {path_k}")
        m = get_model(backbone_type, num_classes=2).to(device)
        ckpt = torch.load(path_k, map_location=device, weights_only=True)
        state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        m.load_state_dict(state, strict=True)
        models.append(m)
    return models


def calc_fi_triple_base_loss(
    labels,
    logits_list,
    device,
    epoch_idx,
    annealing_step,
    kl_weight,
    w_normal,
    w_fault,
    w_other,
    zero_kl_for_other_fault,
    K,
    ood_penalty_weight=0.0,
):
    """logits_list[i] 对应全局 head 索引 k=i+1。"""
    loss_total = torch.tensor(0.0, device=device)
    M = K - 1
    for idx, k in enumerate(range(1, K)):
        logits = logits_list[idx]
        y_triple = triple_target_batch(labels, k, device)
        sw = triple_sample_weights(labels, k, device, w_normal, w_fault, w_other)
        loss_k = fi_triple_edl_mse_loss(
            logits,
            y_triple,
            epoch_idx,
            annealing_step,
            device,
            sample_weight=sw,
            kl_weight=kl_weight,
            zero_kl_for_other_fault=zero_kl_for_other_fault,
            ood_penalty_weight=ood_penalty_weight,
        )
        loss_total = loss_total + loss_k
    return loss_total / max(M, 1)


def main():
    parser = argparse.ArgumentParser(description="Fi-only 阶段二：三态联合微调（无 model0）")
    parser.add_argument("--config", type=str, default="configs/bench_NvF_LaoDA_fi_triple_joint.yaml")
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="弹出 matplotlib 窗口，每个 epoch 刷新 Loss/Acc 曲线（需本机图形界面；无界面时勿用）",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    fj = train_config.get("fi_joint_finetune", {})

    K = len(data_config["openset"]["known_classes"])
    if K < 2:
        raise ValueError("K 至少为 2")
    backbone_type = model_config.get("type", "LaoDA")
    input_ckpt_dir = fj.get("input_checkpoint_dir")
    if not input_ckpt_dir:
        raise ValueError("请在 train.fi_joint_finetune.input_checkpoint_dir 指定阶段一权重目录")
    input_ckpt_dir = os.path.normpath(input_ckpt_dir)
    if not os.path.isabs(input_ckpt_dir):
        input_ckpt_dir = os.path.normpath(os.path.join(os.getcwd(), input_ckpt_dir))
    if not os.path.isdir(input_ckpt_dir):
        raise FileNotFoundError(input_ckpt_dir)

    checkpoint_dir = train_config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    test_infer = train_config.get("test_infer", {})
    tau_n = float(test_infer.get("fi_tau_normal", 0.5))
    tau_f = float(test_infer.get("fi_tau_fault", 0.5))

    flow_png = os.path.join(checkpoint_dir, "fi_stage2_training_flow.png")
    save_fi_stage2_flowchart(flow_png)
    print(f"已保存阶段二训练流程图: {flow_png}")

    batch_size = int(train_config.get("batch_size", 256))
    num_epochs = int(train_config.get("num_epochs", 50))
    learning_rate = float(train_config.get("learning_rate", 1e-4))
    device = torch.device(train_config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    annealing_step = int(train_config.get("edl_annealing_step", 10))
    kl_weight = float(train_config.get("kl_weight", 1.0))
    seed = int(train_config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    head_only = bool(fj.get("head_only", False))
    train_fc1 = bool(fj.get("train_fc1", False))
    w_normal = float(fj.get("w_normal", 1.0))
    w_fault = float(fj.get("w_fault", 1.0))
    w_other = float(fj.get("w_other", 1.0))
    zero_kl_for_other_fault = bool(fj.get("zero_kl_for_other_fault", True))
    ood_penalty_weight = float(fj.get("ood_penalty_weight", 0.0))
    max_batches = int(fj.get("max_batches_per_epoch", 0))

    wn_inj = fj.get("white_noise_injection", {})
    wn_enable = bool(wn_inj.get("enable", False))
    wn_ratio = float(wn_inj.get("ratio", 0.0))
    wn_scale = bool(wn_inj.get("scale_to_real", False))
    # 新增开关：是否开启多样化 OOD 注入
    wn_multi_type = bool(wn_inj.get("multi_type", False))

    train_dataset = get_dataset(data_config, backbone_type, split="train")
    val_dataset = get_dataset(data_config, backbone_type, split="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    models = load_models_fi(input_ckpt_dir, K, backbone_type, device)
    for m in models:
        set_trainable_for_joint(m, backbone_type, head_only=head_only, train_fc1=train_fc1)

    params = []
    for idx, m in enumerate(models):
        names = summarize_trainable_params(m)
        print(f"模型 head 索引 {idx + 1} 可训练参数数: {len(names)}")
        params.extend([p for p in m.parameters() if p.requires_grad])
    if not params:
        raise RuntimeError("无可训练参数")

    optimizer_type = train_config.get("optimizer", "Adam")
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    if optimizer_type == "SGD":
        optimizer = optim.SGD(
            params, lr=learning_rate, momentum=float(train_config.get("momentum", 0.9)), weight_decay=weight_decay
        )
    else:
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    train_start_time = datetime.now()
    save_experiment_info(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        start_time=train_start_time,
    )

    curve_png = os.path.join(checkpoint_dir, "training_curves_fi_joint.png")
    history_csv = os.path.join(checkpoint_dir, "joint_history.csv")

    print(
        f"Fi-only 联合微调: K={K}, M={K - 1} 个头, input={input_ckpt_dir}, out={checkpoint_dir}\n"
        f"监控 Fi 融合阈值: fi_tau_normal={tau_n}, fi_tau_fault={tau_f}（与 train.test_infer 一致）"
    )
    if _LIVE_PLOT:
        print("已启用 --live-plot：将弹出实时曲线窗口；同时仍每个 epoch 写入 training_curves_fi_joint.png")

    best_val = float("inf")
    hist_train, hist_val = [], []
    hist_train_acc, hist_val_acc = [], []
    history_rows = []

    live_bundle = None
    if _LIVE_PLOT:
        plt.ion()
        _configure_cjk_font()
        live_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        live_bundle = (live_fig, ax1, ax2)

    for epoch in range(num_epochs):
        for m in models:
            m.train()
        train_sum = 0.0
        n_samp = 0
        n_samp_real = 0 # 统计真实样本数
        train_correct = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
                
            # 记录本 batch 注入前的真实样本数
            curr_bs_real = inputs.size(0)

            if wn_enable and wn_ratio > 0:
                bs = inputs.size(0)
                n_noise = int(bs * wn_ratio)
                if n_noise > 0:
                    if not wn_multi_type:
                        # 仅白噪声
                        noise = torch.randn(n_noise, *inputs.shape[1:])
                    else:
                        # 随机选择一种噪声类型
                        noise_types = ["white", "uniform", "impulse", "sine", "pink", "zero", "const", "mixed"]
                        selected = np.random.choice(noise_types, size=n_noise)
                        noise = torch.zeros(n_noise, *inputs.shape[1:])
                        
                        # 获取单个样本的总数据量（长度）
                        flat_len = int(np.prod(inputs.shape[1:]))
                        
                        def get_noise_segment(t_type, length):
                            if t_type == "white":
                                return torch.randn(length)
                            elif t_type == "uniform":
                                return torch.rand(length) * 2 - 1
                            elif t_type == "impulse":
                                seg = torch.zeros(length)
                                n_spikes = max(1, int(length * 0.05))
                                idx = np.random.choice(length, size=n_spikes, replace=False)
                                seg[idx] = torch.randn(n_spikes) * 5
                                return seg
                            elif t_type == "sine":
                                freq = np.random.uniform(1, 5)
                                phase = np.random.uniform(0, 2*np.pi)
                                x = torch.linspace(0, freq * 2 * np.pi, length)
                                return torch.sin(x + phase)
                            elif t_type == "pink":
                                white = torch.randn(length)
                                pink = torch.cumsum(white, dim=0)
                                pink = pink - pink.mean()
                                if pink.std() > 0: pink = pink / pink.std()
                                return pink
                            elif t_type == "zero":
                                return torch.zeros(length)
                            elif t_type == "const":
                                return torch.ones(length) * np.random.uniform(-1, 1)
                            else:
                                return torch.randn(length)

                        for i, t in enumerate(selected):
                            if t == "mixed":
                                # 随机拼接：将样本分为 2-3 段，每段随机选一种噪声
                                n_segs = np.random.randint(2, 4)
                                seg_lens = np.random.multinomial(flat_len, [1/n_segs]*n_segs)
                                mixed_flat = []
                                for l in seg_lens:
                                    if l <= 0: continue
                                    sub_t = np.random.choice(noise_types[:-1]) # 不再嵌套 mixed
                                    mixed_flat.append(get_noise_segment(sub_t, l))
                                noise[i].view(-1).copy_(torch.cat(mixed_flat))
                            else:
                                noise[i].view(-1).copy_(get_noise_segment(t, flat_len))

                    if wn_scale:
                        real_std = inputs.std().item()
                        real_mean = inputs.mean().item()
                        # 对随机数进行缩放，对零或常数则主要是为了保持量级一致
                        noise = noise * real_std + real_mean
                    noise_labels = torch.full((n_noise,), K, dtype=labels.dtype)
                    inputs = torch.cat([inputs, noise], dim=0)
                    labels = torch.cat([labels, noise_labels], dim=0)
                    # Shuffle to mix noise and real data
                    perm = torch.randperm(inputs.size(0))
                    inputs = inputs[perm]
                    labels = labels[perm]

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits_list = []
            for m in models:
                out = m(inputs)
                logits_list.append(out[0] if isinstance(out, tuple) else out)
            loss = calc_fi_triple_base_loss(
                labels,
                logits_list,
                device,
                epoch,
                annealing_step,
                kl_weight,
                w_normal,
                w_fault,
                w_other,
                zero_kl_for_other_fault,
                K,
                ood_penalty_weight=ood_penalty_weight,
            )
            loss.backward()
            optimizer.step()
            bs = inputs.size(0)
            train_sum += loss.item() * bs
            n_samp += bs
            
            with torch.no_grad():
                det = [x.detach() for x in logits_list]
                # 仅针对真实样本 (label < K) 计算训练精度
                real_mask = labels < K
                n_real = real_mask.sum().item()
                if n_real > 0:
                    # 过滤出真实样本的 logits 和 labels
                    real_det = [d[real_mask] for d in det]
                    real_labels = labels[real_mask]
                    train_correct += fi_fusion_correct_in_batch(real_det, real_labels, device, tau_n, tau_f)
                    n_samp_real += n_real

        train_loss = train_sum / max(n_samp, 1)
        # 使用真实样本数作为分母，这样精度就能上到 90%+ 甚至 100%
        train_acc_pct = 100.0 * train_correct / max(n_samp_real, 1)

        for m in models:
            m.eval()
        val_sum = 0.0
        v_samp = 0
        val_correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits_list = []
                for m in models:
                    out = m(inputs)
                    logits_list.append(out[0] if isinstance(out, tuple) else out)
                loss = calc_fi_triple_base_loss(
                    labels,
                    logits_list,
                    device,
                    epoch,
                    annealing_step,
                    kl_weight,
                    w_normal,
                    w_fault,
                    w_other,
                    zero_kl_for_other_fault,
                    K,
                )
                bs = inputs.size(0)
                val_sum += loss.item() * bs
                v_samp += bs
                val_correct += fi_fusion_correct_in_batch(logits_list, labels, device, tau_n, tau_f)
        val_loss = val_sum / max(v_samp, 1)
        val_acc_pct = 100.0 * val_correct / max(v_samp, 1)
        hist_train.append(train_loss)
        hist_val.append(val_loss)
        hist_train_acc.append(train_acc_pct)
        hist_val_acc.append(val_acc_pct)

        if val_loss < best_val:
            best_val = val_loss
            for k in range(1, K):
                torch.save(
                    {"state_dict": models[k - 1].state_dict(), "k": k, "best_val_loss": best_val},
                    os.path.join(checkpoint_dir, f"model_{k}.pth"),
                )

        history_rows.append(
            [epoch + 1, train_loss, val_loss, train_acc_pct, val_acc_pct, best_val]
        )
        with open(history_csv, "w", newline="", encoding="utf-8") as hf:
            hw = csv.writer(hf)
            hw.writerow(
                ["epoch", "train_loss", "val_loss", "train_acc_pct", "val_acc_pct", "best_val_loss_so_far"]
            )
            hw.writerows(history_rows)

        save_training_curves_fi_joint(
            curve_png,
            hist_train,
            hist_val,
            hist_train_acc,
            hist_val_acc,
            fig_axes=live_bundle,
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_acc={train_acc_pct:.2f}% val_acc={val_acc_pct:.2f}% best_val={best_val:.6f}"
        )

        if train_config.get("save_every_epoch", False):
            ep_dir = os.path.join(checkpoint_dir, "epochs", str(epoch + 1))
            os.makedirs(ep_dir, exist_ok=True)
            for k in range(1, K):
                torch.save(
                    {"state_dict": models[k - 1].state_dict(), "k": k, "epoch": epoch + 1},
                    os.path.join(ep_dir, f"model_{k}.pth"),
                )

    for k in range(1, K):
        p = os.path.join(checkpoint_dir, f"model_{k}.pth")
        if not os.path.exists(p):
            torch.save({"state_dict": models[k - 1].state_dict(), "k": k}, p)

    if _LIVE_PLOT and live_bundle is not None:
        plt.ioff()
        plt.close(live_bundle[0])

    save_experiment_info(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        start_time=train_start_time,
        end_time=datetime.now(),
    )
    print(f"Fi-only 联合微调完成 best_val_loss={best_val:.6f} -> {checkpoint_dir}")


if __name__ == "__main__":
    main()
