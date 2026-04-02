import argparse
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator


def _configure_matplotlib_cjk_font():
    """避免中文标题/图例在 Agg 导出时为方框（选系统内首个可用 CJK 无衬线字体）。"""
    plt.rcParams["axes.unicode_minus"] = False
    preferred = (
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "SimSun",
        "KaiTi",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Heiti SC",
    )
    names_in_db = {f.name for f in fm.fontManager.ttflist}
    chosen = next((n for n in preferred if n in names_in_db), None)
    if chosen is None:
        for f in fm.fontManager.ttflist:
            n = f.name
            if any(k in n for k in ("YaHei", "SimHei", "Noto Sans CJK", "Han Sans", "WenQuanYi")):
                chosen = n
                break
    if chosen is not None:
        sans = [chosen] + [x for x in plt.rcParams["font.sans-serif"] if x != chosen]
        plt.rcParams["font.sans-serif"] = sans
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.utils.helpers import load_config, save_experiment_info
from common.utils.data_loader import NpyPackDataset
from common.utils.data_loader_1d import NpyPackDataset1D
from common.edl_losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from common.utils.edl_helpers import one_hot_embedding
from common.utils.lao_da_pretrained import summarize_trainable_params
from models import get_model


def get_dataset(data_config, model_type, split="train"):
    data_dir = data_config.get("data_dir")
    if data_dir is None:
        raise ValueError("配置文件中必须指定 data.data_dir")
    openset_config = data_config.get("openset", {})
    known_classes = openset_config.get("known_classes")
    unknown_classes = openset_config.get("unknown_classes", [])
    if known_classes is None:
        raise ValueError("配置文件中必须指定 data.openset.known_classes")
    dataset_cls = NpyPackDataset1D if model_type == "LaoDA" else NpyPackDataset
    return dataset_cls(
        data_dir=data_dir,
        split=split,
        filter_classes=known_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
    )


def resolve_model_indices(train_config, K):
    model_indices_cfg = train_config.get("model_indices", "all")
    if model_indices_cfg is None or model_indices_cfg == "all":
        return list(range(K))
    if isinstance(model_indices_cfg, int):
        model_indices = [model_indices_cfg]
    elif isinstance(model_indices_cfg, list):
        model_indices = model_indices_cfg
    else:
        raise ValueError("train.model_indices 仅支持 'all'、整数或整数列表")

    normalized = []
    for k in model_indices:
        if not isinstance(k, int):
            raise ValueError("train.model_indices 列表中的元素必须是整数")
        if k < 0 or k >= K:
            raise ValueError(f"train.model_indices 包含越界索引 {k}，应在 [0, {K - 1}]")
        if k not in normalized:
            normalized.append(k)
    if not normalized:
        raise ValueError("train.model_indices 解析后为空，请至少指定一个子模型索引")
    return normalized


def build_binary_labels(labels, k):
    if k == 0:
        return (labels == 0).long()
    return (labels == k).long()


def select_criterion(edl_loss_type):
    if edl_loss_type == "mse":
        return edl_mse_loss
    if edl_loss_type == "digamma":
        return edl_digamma_loss
    if edl_loss_type == "log":
        return edl_log_loss
    raise ValueError(f"不支持的 EDL 损失: {edl_loss_type}")


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

    # ResNet 系列：默认仅训练最后分类层 fc
    if hasattr(model, "fc"):
        for n, p in model.named_parameters():
            if n.startswith("fc."):
                p.requires_grad = True
        return

    # 兜底：名称中包含 fc 的层
    for n, p in model.named_parameters():
        if "fc" in n:
            p.requires_grad = True


def get_lambda(epoch_idx, warmup_epochs, ramp_epochs, lambda_unc):
    if epoch_idx < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(lambda_unc)
    progress = min(1.0, float(epoch_idx - warmup_epochs + 1) / float(ramp_epochs))
    return float(lambda_unc) * progress


def calc_base_loss(
    models,
    labels,
    logits_list,
    criterion,
    epoch_idx,
    annealing_step,
    device,
    kl_weight,
):
    loss_total = torch.tensor(0.0, device=device)
    for k, logits in enumerate(logits_list):
        binary_labels = build_binary_labels(labels, k)
        y_binary = one_hot_embedding(binary_labels, 2).float().to(device)
        n_pos = (binary_labels == 1).sum().item()
        n_neg = (binary_labels == 0).sum().item()
        pos_w = n_neg / max(n_pos, 1)
        sample_w = torch.where(
            binary_labels == 1,
            torch.tensor(pos_w, device=device, dtype=torch.float),
            torch.ones_like(binary_labels, device=device, dtype=torch.float),
        )
        loss_k = criterion(
            logits,
            y_binary,
            epoch_idx,
            2,
            annealing_step,
            device,
            sample_weight=sample_w,
            kl_weight=kl_weight,
        )
        loss_total = loss_total + loss_k
    return loss_total / max(len(models), 1)


def nvf_wrong_expert_mask(labels: torch.Tensor, num_experts: int, exclude_model0_on_fault: bool) -> torch.Tensor:
    """
    NvF 错专家掩码 [B, K]：True 表示该样本在该列上计入 L_unc wrong / L_neg。
    exclude_model0_on_fault=True 且 y>0 时，第 0 列（model_0）永不视为错专家，避免与「故障上判异常」的 L_base 冲突。
    """
    wrong = ~F.one_hot(labels, num_classes=num_experts).bool()
    if exclude_model0_on_fault:
        wrong[labels > 0, 0] = False
    return wrong


def calc_uncertainty_loss(
    u_stack,
    labels,
    match_weight,
    wrong_weight,
    wrong_mode,
    u_margin,
    exclude_model0_on_fault: bool = False,
):
    # 匹配专家：真实类对应 expert 的不确定度应低
    u_match = u_stack.gather(1, labels.view(-1, 1)).squeeze(1)
    loss_match = u_match.mean()

    wrong_mask = nvf_wrong_expert_mask(labels, u_stack.shape[1], exclude_model0_on_fault)
    wrong_flat = u_stack[wrong_mask]
    if wrong_flat.numel() == 0:
        loss_wrong = torch.zeros((), device=u_stack.device, dtype=u_stack.dtype)
    elif wrong_mode == "target1":
        loss_wrong = ((1.0 - wrong_flat) ** 2).mean()
    else:
        loss_wrong = torch.relu(float(u_margin) - wrong_flat).mean()

    return float(match_weight) * loss_match + float(wrong_weight) * loss_wrong, loss_match, loss_wrong


def mean_u_match_and_wrong(u_stack, labels, exclude_model0_on_fault: bool = False):
    """匹配专家与错专家的平均不确定度 u=2/S（用于曲线诊断，非损失项）。"""
    u_match = u_stack.gather(1, labels.view(-1, 1)).squeeze(1)
    wrong_mask = nvf_wrong_expert_mask(labels, u_stack.shape[1], exclude_model0_on_fault)
    wrong_flat = u_stack[wrong_mask]
    if wrong_flat.numel() == 0:
        u_wrong_mean = torch.zeros((), device=u_stack.device, dtype=u_stack.dtype)
    else:
        u_wrong_mean = wrong_flat.mean()
    return u_match.mean(), u_wrong_mean


def calc_wrong_pos_logit_loss(logits_list, labels, neg_margin, exclude_model0_on_fault: bool = False):
    """
    直接压制错专家正类 logit：
    L_neg = mean( relu(pos_logit_wrong - neg_margin) )
    仅对每个样本的非目标专家位置计算。
    """
    pos_logits = torch.stack([logits[:, 1] for logits in logits_list], dim=1)  # [B, K]
    wrong_mask = nvf_wrong_expert_mask(labels, pos_logits.shape[1], exclude_model0_on_fault)
    wrong_pos = pos_logits[wrong_mask]
    if wrong_pos.numel() == 0:
        return torch.zeros((), device=pos_logits.device, dtype=pos_logits.dtype)
    return torch.relu(wrong_pos - float(neg_margin)).mean()


def forward_logits_and_u(models, inputs):
    logits_list = []
    u_list = []
    for model in models:
        logits_k = model(inputs)
        logits_k = logits_k[0] if isinstance(logits_k, tuple) else logits_k
        logits_list.append(logits_k)
        evidence = relu_evidence(logits_k)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1)
        u_list.append(2.0 / S)
    return logits_list, torch.stack(u_list, dim=1)


def estimate_reference_losses(
    models,
    train_loader,
    device,
    criterion,
    annealing_step,
    kl_weight,
    match_weight,
    wrong_weight,
    wrong_mode,
    u_margin,
    use_wrong_pos_loss,
    neg_margin,
    ref_batches,
    exclude_model0_on_fault: bool = False,
):
    for m in models:
        m.eval()

    base_sum = 0.0
    unc_sum = 0.0
    neg_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if ref_batches > 0 and batch_idx >= ref_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits_list, u_stack = forward_logits_and_u(models, inputs)
            base_loss = calc_base_loss(
                models=models,
                labels=labels,
                logits_list=logits_list,
                criterion=criterion,
                epoch_idx=0,
                annealing_step=annealing_step,
                device=device,
                kl_weight=kl_weight,
            )
            unc_loss, _, _ = calc_uncertainty_loss(
                u_stack=u_stack,
                labels=labels,
                match_weight=match_weight,
                wrong_weight=wrong_weight,
                wrong_mode=wrong_mode,
                u_margin=u_margin,
                exclude_model0_on_fault=exclude_model0_on_fault,
            )
            if use_wrong_pos_loss:
                neg_loss = calc_wrong_pos_logit_loss(
                    logits_list=logits_list,
                    labels=labels,
                    neg_margin=neg_margin,
                    exclude_model0_on_fault=exclude_model0_on_fault,
                )
            else:
                neg_loss = torch.tensor(0.0, device=device)

            base_sum += base_loss.item()
            unc_sum += unc_loss.item()
            neg_sum += neg_loss.item()
            n_batches += 1

    if n_batches <= 0:
        return 1.0, 1.0, 1.0
    ref_base = base_sum / n_batches
    ref_unc = unc_sum / n_batches
    # neg loss 关闭时给 1.0，避免无意义归一化。
    ref_neg = (neg_sum / n_batches) if use_wrong_pos_loss else 1.0
    return max(ref_base, 0.0), max(ref_unc, 0.0), max(ref_neg, 0.0)


def load_models_from_checkpoint(checkpoint_dir, K, backbone_type, device):
    models = []
    for k in range(K):
        path_k = os.path.join(checkpoint_dir, f"model_{k}.pth")
        if not os.path.exists(path_k):
            raise FileNotFoundError(f"未找到子模型权重: {path_k}")
        model_k = get_model(backbone_type, num_classes=2).to(device)
        ckpt = torch.load(path_k, map_location=device, weights_only=True)
        state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        model_k.load_state_dict(state, strict=True)
        models.append(model_k)
    return models


def main():
    parser = argparse.ArgumentParser(description="NvF 联合微调：提升错专家不确定度")
    parser.add_argument("--config", type=str, default="configs/bench_NvF_LaoDA_joint_uncertainty.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    joint_cfg = train_config.get("joint_finetune", {})

    known_classes = data_config["openset"]["known_classes"]
    K = len(known_classes)
    backbone_type = model_config.get("type", "LaoDA")
    model_indices = resolve_model_indices(train_config, K)
    if model_indices != list(range(K)):
        raise ValueError("联合微调阶段要求 train.model_indices='all'，需同时优化全部子模型。")

    input_ckpt_dir = joint_cfg.get("input_checkpoint_dir")
    if not input_ckpt_dir:
        raise ValueError("请在 train.joint_finetune.input_checkpoint_dir 指定基线子模型目录")
    input_ckpt_dir = os.path.normpath(input_ckpt_dir)
    if not os.path.isabs(input_ckpt_dir):
        input_ckpt_dir = os.path.normpath(os.path.join(os.getcwd(), input_ckpt_dir))
    if not os.path.isdir(input_ckpt_dir):
        raise FileNotFoundError(f"基线子模型目录不存在: {input_ckpt_dir}")

    checkpoint_dir = train_config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_size = int(train_config.get("batch_size", 256))
    num_epochs = int(train_config.get("num_epochs", 20))
    learning_rate = float(train_config.get("learning_rate", 1e-4))
    device = torch.device(train_config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    edl_loss_type = train_config.get("edl_loss_type", "mse")
    annealing_step = int(train_config.get("edl_annealing_step", 10))
    kl_weight = float(train_config.get("kl_weight", 1.0))
    seed = int(train_config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    head_only = bool(joint_cfg.get("head_only", True))
    train_fc1 = bool(joint_cfg.get("train_fc1", False))
    warmup_epochs = int(joint_cfg.get("warmup_epochs", 5))
    ramp_epochs = int(joint_cfg.get("ramp_epochs", 5))
    lambda_unc = float(joint_cfg.get("lambda_unc", 0.1))
    wrong_mode = joint_cfg.get("wrong_mode", "margin")
    u_margin = float(joint_cfg.get("u_margin", 0.8))
    match_weight = float(joint_cfg.get("match_weight", 1.0))
    wrong_weight = float(joint_cfg.get("wrong_weight", 1.0))
    wrong_unc_exclude_model0_on_fault = bool(joint_cfg.get("wrong_unc_exclude_model0_on_fault", False))
    use_wrong_pos_loss = bool(joint_cfg.get("use_wrong_pos_loss", False))
    lambda_neg = float(joint_cfg.get("lambda_neg", 0.0))
    neg_margin = float(joint_cfg.get("neg_margin", 0.0))
    neg_warmup_epochs = int(joint_cfg.get("neg_warmup_epochs", warmup_epochs))
    neg_ramp_epochs = int(joint_cfg.get("neg_ramp_epochs", ramp_epochs))
    max_batches = int(joint_cfg.get("max_batches_per_epoch", 0))
    norm_cfg = joint_cfg.get("loss_normalization", {})
    norm_enable = bool(norm_cfg.get("enable", False))
    norm_mode = str(norm_cfg.get("mode", "fixed_ref"))
    norm_ref_batches = int(norm_cfg.get("ref_batches", 20))
    norm_eps = float(norm_cfg.get("eps", 1e-8))

    if wrong_mode not in ("margin", "target1"):
        raise ValueError("joint_finetune.wrong_mode 仅支持 margin 或 target1")
    if norm_enable and norm_mode != "fixed_ref":
        raise ValueError("joint_finetune.loss_normalization.mode 当前仅支持 fixed_ref")

    print(f"K={K}, backbone={backbone_type}, device={device}")
    print(f"输入基线权重目录: {input_ckpt_dir}")
    print(
        f"joint: head_only={head_only}, train_fc1={train_fc1}, warmup={warmup_epochs}, "
        f"ramp={ramp_epochs}, lambda_unc={lambda_unc}, wrong_mode={wrong_mode}, u_margin={u_margin}, "
        f"use_wrong_pos_loss={use_wrong_pos_loss}, lambda_neg={lambda_neg}, neg_margin={neg_margin}, "
        f"neg_warmup={neg_warmup_epochs}, neg_ramp={neg_ramp_epochs}, "
        f"wrong_unc_exclude_model0_on_fault={wrong_unc_exclude_model0_on_fault}, "
        f"loss_norm_enable={norm_enable}, loss_norm_mode={norm_mode}, loss_norm_ref_batches={norm_ref_batches}"
    )

    train_dataset = get_dataset(data_config, backbone_type, split="train")
    val_dataset = get_dataset(data_config, backbone_type, split="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    models = load_models_from_checkpoint(input_ckpt_dir, K, backbone_type, device)
    for m in models:
        set_trainable_for_joint(m, backbone_type=backbone_type, head_only=head_only, train_fc1=train_fc1)

    params = []
    for idx, m in enumerate(models):
        names = summarize_trainable_params(m)
        print(f"模型 {idx} 可训练参数数: {len(names)}")
        params.extend([p for p in m.parameters() if p.requires_grad])
    if not params:
        raise RuntimeError("联合微调无可训练参数，请检查 head_only/train_fc1 设置")

    optimizer_type = train_config.get("optimizer", "Adam")
    momentum = float(train_config.get("momentum", 0.9))
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    if optimizer_type == "SGD":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    criterion = select_criterion(edl_loss_type)
    scheduler_conf = train_config.get("scheduler", {})
    if scheduler_conf.get("use_scheduler", False) and scheduler_conf.get("scheduler_type", "StepLR") == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_conf.get("step_size", 40)),
            gamma=float(scheduler_conf.get("gamma", 0.1)),
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)

    ref_base = 1.0
    ref_unc = 1.0
    ref_neg = 1.0
    if norm_enable:
        ref_base, ref_unc, ref_neg = estimate_reference_losses(
            models=models,
            train_loader=train_loader,
            device=device,
            criterion=criterion,
            annealing_step=annealing_step,
            kl_weight=kl_weight,
            match_weight=match_weight,
            wrong_weight=wrong_weight,
            wrong_mode=wrong_mode,
            u_margin=u_margin,
            use_wrong_pos_loss=use_wrong_pos_loss,
            neg_margin=neg_margin,
            ref_batches=norm_ref_batches,
            exclude_model0_on_fault=wrong_unc_exclude_model0_on_fault,
        )
    print(
        f"loss_norm_ref: base={ref_base:.6f}, unc={ref_unc:.6f}, neg={ref_neg:.6f}, eps={norm_eps:.1e}"
    )

    history = {
        "train_total_loss": [],
        "train_total_raw_loss": [],
        "train_base_loss": [],
        "train_unc_loss": [],
        "train_neg_loss": [],
        "train_base_norm_loss": [],
        "train_unc_norm_loss": [],
        "train_neg_norm_loss": [],
        "train_u_match": [],
        "train_u_wrong": [],
        "val_total_loss": [],
        "val_total_raw_loss": [],
        "val_base_loss": [],
        "val_unc_loss": [],
        "val_neg_loss": [],
        "val_base_norm_loss": [],
        "val_unc_norm_loss": [],
        "val_neg_norm_loss": [],
        "val_u_match": [],
        "val_u_wrong": [],
    }

    def write_history_csv(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,train_total,train_total_raw,train_base,train_unc,train_neg,"
                "train_u_match,train_u_wrong,"
                "train_base_norm,train_unc_norm,train_neg_norm,"
                "val_total,val_total_raw,val_base,val_unc,val_neg,"
                "val_u_match,val_u_wrong,"
                "val_base_norm,val_unc_norm,val_neg_norm,"
                "ref_base,ref_unc,ref_neg\n"
            )
            for i in range(len(history["train_total_loss"])):
                f.write(
                    f"{i + 1},"
                    f"{history['train_total_loss'][i]:.6f},"
                    f"{history['train_total_raw_loss'][i]:.6f},"
                    f"{history['train_base_loss'][i]:.6f},"
                    f"{history['train_unc_loss'][i]:.6f},"
                    f"{history['train_neg_loss'][i]:.6f},"
                    f"{history['train_u_match'][i]:.6f},"
                    f"{history['train_u_wrong'][i]:.6f},"
                    f"{history['train_base_norm_loss'][i]:.6f},"
                    f"{history['train_unc_norm_loss'][i]:.6f},"
                    f"{history['train_neg_norm_loss'][i]:.6f},"
                    f"{history['val_total_loss'][i]:.6f},"
                    f"{history['val_total_raw_loss'][i]:.6f},"
                    f"{history['val_base_loss'][i]:.6f},"
                    f"{history['val_unc_loss'][i]:.6f},"
                    f"{history['val_neg_loss'][i]:.6f},"
                    f"{history['val_u_match'][i]:.6f},"
                    f"{history['val_u_wrong'][i]:.6f},"
                    f"{history['val_base_norm_loss'][i]:.6f},"
                    f"{history['val_unc_norm_loss'][i]:.6f},"
                    f"{history['val_neg_norm_loss'][i]:.6f},"
                    f"{ref_base:.6f},"
                    f"{ref_unc:.6f},"
                    f"{ref_neg:.6f}\n"
                )

    def save_curve_png(path):
        _configure_matplotlib_cjk_font()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
        ax1.plot(history["train_total_loss"], label="TrainTotal")
        ax1.plot(history["val_total_loss"], label="ValTotal")
        ax1.plot(history["train_base_loss"], "--", label="TrainBase")
        ax1.plot(history["val_base_loss"], "--", label="ValBase")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.legend(fontsize=8)
        ax1.grid(True)

        ax2.plot(history["train_u_match"], label="Train U(match)")
        ax2.plot(history["val_u_match"], label="Val U(match)")
        ax2.plot(history["train_u_wrong"], "--", label="Train U(wrong mean)")
        ax2.plot(history["val_u_wrong"], "--", label="Val U(wrong mean)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean u = 2/S")
        ax2.set_title("正确专家 vs 错专家 平均不确定度")
        ax2.set_ylim(0.0, 1.05)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(fontsize=7)
        ax2.grid(True)

        ax3.plot(history["train_unc_loss"], label="Train L_unc")
        ax3.plot(history["val_unc_loss"], label="Val L_unc")
        ax3.plot(history["train_neg_loss"], label="Train L_neg")
        ax3.plot(history["val_neg_loss"], label="Val L_neg")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Aux loss")
        ax3.set_title("L_unc 合计与 L_neg")
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.legend(fontsize=8)
        ax3.grid(True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    best_val_total = float("inf")
    train_start_time = datetime.now()
    save_experiment_info(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        start_time=train_start_time,
    )

    for epoch in range(num_epochs):
        cur_lambda = get_lambda(epoch, warmup_epochs, ramp_epochs, lambda_unc)
        cur_lambda_neg = get_lambda(epoch, neg_warmup_epochs, neg_ramp_epochs, lambda_neg)
        for m in models:
            m.train()
        train_total_sum = 0.0
        train_total_raw_sum = 0.0
        train_base_sum = 0.0
        train_unc_sum = 0.0
        train_neg_sum = 0.0
        train_base_norm_sum = 0.0
        train_unc_norm_sum = 0.0
        train_neg_norm_sum = 0.0
        train_u_match_sum = 0.0
        train_u_wrong_sum = 0.0
        train_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits_list, u_stack = forward_logits_and_u(models, inputs)

            base_loss = calc_base_loss(
                models=models,
                labels=labels,
                logits_list=logits_list,
                criterion=criterion,
                epoch_idx=epoch,
                annealing_step=annealing_step,
                device=device,
                kl_weight=kl_weight,
            )
            unc_loss, _, _ = calc_uncertainty_loss(
                u_stack=u_stack,
                labels=labels,
                match_weight=match_weight,
                wrong_weight=wrong_weight,
                wrong_mode=wrong_mode,
                u_margin=u_margin,
                exclude_model0_on_fault=wrong_unc_exclude_model0_on_fault,
            )
            u_match_m, u_wrong_m = mean_u_match_and_wrong(
                u_stack, labels, exclude_model0_on_fault=wrong_unc_exclude_model0_on_fault
            )
            if use_wrong_pos_loss:
                neg_loss = calc_wrong_pos_logit_loss(
                    logits_list=logits_list,
                    labels=labels,
                    neg_margin=neg_margin,
                    exclude_model0_on_fault=wrong_unc_exclude_model0_on_fault,
                )
            else:
                neg_loss = torch.tensor(0.0, device=device)

            raw_total_loss = base_loss + cur_lambda * unc_loss + cur_lambda_neg * neg_loss
            base_loss_n = base_loss / (ref_base + norm_eps)
            unc_loss_n = unc_loss / (ref_unc + norm_eps)
            neg_loss_n = neg_loss / (ref_neg + norm_eps)
            norm_total_loss = base_loss_n + cur_lambda * unc_loss_n + cur_lambda_neg * neg_loss_n
            total_loss = norm_total_loss if norm_enable else raw_total_loss
            total_loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            train_samples += bs
            train_total_sum += total_loss.item() * bs
            train_total_raw_sum += raw_total_loss.item() * bs
            train_base_sum += base_loss.item() * bs
            train_unc_sum += unc_loss.item() * bs
            train_neg_sum += neg_loss.item() * bs
            train_base_norm_sum += base_loss_n.item() * bs
            train_unc_norm_sum += unc_loss_n.item() * bs
            train_neg_norm_sum += neg_loss_n.item() * bs
            train_u_match_sum += u_match_m.item() * bs
            train_u_wrong_sum += u_wrong_m.item() * bs

        for m in models:
            m.eval()
        val_total_sum = 0.0
        val_total_raw_sum = 0.0
        val_base_sum = 0.0
        val_unc_sum = 0.0
        val_neg_sum = 0.0
        val_base_norm_sum = 0.0
        val_unc_norm_sum = 0.0
        val_neg_norm_sum = 0.0
        val_u_match_sum = 0.0
        val_u_wrong_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits_list, u_stack = forward_logits_and_u(models, inputs)

                base_loss = calc_base_loss(
                    models=models,
                    labels=labels,
                    logits_list=logits_list,
                    criterion=criterion,
                    epoch_idx=epoch,
                    annealing_step=annealing_step,
                    device=device,
                    kl_weight=kl_weight,
                )
                unc_loss, _, _ = calc_uncertainty_loss(
                    u_stack=u_stack,
                    labels=labels,
                    match_weight=match_weight,
                    wrong_weight=wrong_weight,
                    wrong_mode=wrong_mode,
                    u_margin=u_margin,
                    exclude_model0_on_fault=wrong_unc_exclude_model0_on_fault,
                )
                u_match_m, u_wrong_m = mean_u_match_and_wrong(
                    u_stack, labels, exclude_model0_on_fault=wrong_unc_exclude_model0_on_fault
                )
                if use_wrong_pos_loss:
                    neg_loss = calc_wrong_pos_logit_loss(
                        logits_list=logits_list,
                        labels=labels,
                        neg_margin=neg_margin,
                        exclude_model0_on_fault=wrong_unc_exclude_model0_on_fault,
                    )
                else:
                    neg_loss = torch.tensor(0.0, device=device)
                raw_total_loss = base_loss + cur_lambda * unc_loss + cur_lambda_neg * neg_loss
                base_loss_n = base_loss / (ref_base + norm_eps)
                unc_loss_n = unc_loss / (ref_unc + norm_eps)
                neg_loss_n = neg_loss / (ref_neg + norm_eps)
                norm_total_loss = base_loss_n + cur_lambda * unc_loss_n + cur_lambda_neg * neg_loss_n
                total_loss = norm_total_loss if norm_enable else raw_total_loss

                bs = inputs.size(0)
                val_samples += bs
                val_total_sum += total_loss.item() * bs
                val_total_raw_sum += raw_total_loss.item() * bs
                val_base_sum += base_loss.item() * bs
                val_unc_sum += unc_loss.item() * bs
                val_neg_sum += neg_loss.item() * bs
                val_base_norm_sum += base_loss_n.item() * bs
                val_unc_norm_sum += unc_loss_n.item() * bs
                val_neg_norm_sum += neg_loss_n.item() * bs
                val_u_match_sum += u_match_m.item() * bs
                val_u_wrong_sum += u_wrong_m.item() * bs

        scheduler.step()

        train_total = train_total_sum / max(train_samples, 1)
        train_base = train_base_sum / max(train_samples, 1)
        train_unc = train_unc_sum / max(train_samples, 1)
        train_neg = train_neg_sum / max(train_samples, 1)
        train_total_raw = train_total_raw_sum / max(train_samples, 1)
        train_base_norm = train_base_norm_sum / max(train_samples, 1)
        train_unc_norm = train_unc_norm_sum / max(train_samples, 1)
        train_neg_norm = train_neg_norm_sum / max(train_samples, 1)
        train_u_match = train_u_match_sum / max(train_samples, 1)
        train_u_wrong = train_u_wrong_sum / max(train_samples, 1)
        val_total = val_total_sum / max(val_samples, 1)
        val_base = val_base_sum / max(val_samples, 1)
        val_unc = val_unc_sum / max(val_samples, 1)
        val_neg = val_neg_sum / max(val_samples, 1)
        val_total_raw = val_total_raw_sum / max(val_samples, 1)
        val_base_norm = val_base_norm_sum / max(val_samples, 1)
        val_unc_norm = val_unc_norm_sum / max(val_samples, 1)
        val_neg_norm = val_neg_norm_sum / max(val_samples, 1)
        val_u_match = val_u_match_sum / max(val_samples, 1)
        val_u_wrong = val_u_wrong_sum / max(val_samples, 1)
        history["train_total_loss"].append(train_total)
        history["train_total_raw_loss"].append(train_total_raw)
        history["train_base_loss"].append(train_base)
        history["train_unc_loss"].append(train_unc)
        history["train_neg_loss"].append(train_neg)
        history["train_base_norm_loss"].append(train_base_norm)
        history["train_unc_norm_loss"].append(train_unc_norm)
        history["train_neg_norm_loss"].append(train_neg_norm)
        history["train_u_match"].append(train_u_match)
        history["train_u_wrong"].append(train_u_wrong)
        history["val_total_loss"].append(val_total)
        history["val_total_raw_loss"].append(val_total_raw)
        history["val_base_loss"].append(val_base)
        history["val_unc_loss"].append(val_unc)
        history["val_neg_loss"].append(val_neg)
        history["val_base_norm_loss"].append(val_base_norm)
        history["val_unc_norm_loss"].append(val_unc_norm)
        history["val_neg_norm_loss"].append(val_neg_norm)
        history["val_u_match"].append(val_u_match)
        history["val_u_wrong"].append(val_u_wrong)

        # 训练中实时刷新：每个 epoch 覆盖保存曲线与历史 CSV
        save_curve_png(os.path.join(checkpoint_dir, "training_curves_joint_uncertainty.png"))
        write_history_csv(os.path.join(checkpoint_dir, "joint_history.csv"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"lambda={cur_lambda:.4f} lambda_neg={cur_lambda_neg:.4f} "
                f"train_total={train_total:.4f} train_total_raw={train_total_raw:.4f} "
                f"train_base={train_base:.4f} train_unc={train_unc:.4f} train_neg={train_neg:.4f} "
                f"u_match={train_u_match:.4f} u_wrong={train_u_wrong:.4f} "
                f"train_base_n={train_base_norm:.4f} train_unc_n={train_unc_norm:.4f} train_neg_n={train_neg_norm:.4f} "
                f"val_total={val_total:.4f} val_total_raw={val_total_raw:.4f} "
                f"val_base={val_base:.4f} val_unc={val_unc:.4f} val_neg={val_neg:.4f} "
                f"val_u_match={val_u_match:.4f} val_u_wrong={val_u_wrong:.4f}"
            )

        if val_total < best_val_total:
            best_val_total = val_total
            for k in range(K):
                path_k = os.path.join(checkpoint_dir, f"model_{k}.pth")
                torch.save(
                    {
                        "state_dict": models[k].state_dict(),
                        "k": k,
                        "best_val_total_loss": best_val_total,
                    },
                    path_k,
                )

        # 每个 epoch 额外保存，便于对比不同阶段结果
        if train_config.get("save_every_epoch", False):
            ep_dir = os.path.join(checkpoint_dir, "epochs", str(epoch + 1))
            os.makedirs(ep_dir, exist_ok=True)
            for k in range(K):
                path_k = os.path.join(ep_dir, f"model_{k}.pth")
                torch.save({"state_dict": models[k].state_dict(), "k": k, "epoch": epoch + 1}, path_k)

    # 保存最终权重（防止无最佳保存）
    for k in range(K):
        path_k = os.path.join(checkpoint_dir, f"model_{k}.pth")
        if not os.path.exists(path_k):
            torch.save({"state_dict": models[k].state_dict(), "k": k}, path_k)

    # 再保存一次最终版本（与训练中实时刷新保持一致）
    save_curve_png(os.path.join(checkpoint_dir, "training_curves_joint_uncertainty.png"))
    write_history_csv(os.path.join(checkpoint_dir, "joint_history.csv"))

    save_experiment_info(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        start_time=train_start_time,
        end_time=datetime.now(),
    )
    print(f"联合微调完成。最佳 val_total_loss={best_val_total:.6f}")
    print(f"输出目录: {checkpoint_dir}")


if __name__ == "__main__":
    main()

