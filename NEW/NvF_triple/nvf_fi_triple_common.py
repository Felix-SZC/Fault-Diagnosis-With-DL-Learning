# Fi-only NvF：三态监督 [1,0]/[0,1]/[0,0] 与 EDL-MSE（可选对 [0,0] 屏蔽 KL）
# 独立模块，不修改 common/edl_losses.py

from __future__ import annotations

import torch
from common.edl_losses import relu_evidence, loglikelihood_loss, kl_divergence


def triple_target_batch(labels: torch.Tensor, head_k: int, device: torch.device) -> torch.Tensor:
    """
    对 fault 专家 head_k（全局类索引 k，1<=k<=K-1）构造 [B,2] 目标。
    正常(0) -> [1,0]；第 k 类故障 -> [0,1]；其它已知故障 -> [0,0]。
    """
    labels = labels.to(device)
    y = torch.zeros(labels.size(0), 2, device=device, dtype=torch.float32)
    n = labels == 0
    f = labels == head_k
    y[n, 0] = 1.0
    y[n, 1] = 0.0
    y[f, 0] = 0.0
    y[f, 1] = 1.0
    o = ~(n | f)
    y[o, 0] = 0.0
    y[o, 1] = 0.0
    return y


def triple_state_batch(labels: torch.Tensor, head_k: int) -> torch.Tensor:
    """0=正常态, 1=本类故障, 2=其它故障（对应该头 [0,0]）"""
    labels = labels.long()
    out = torch.full_like(labels, 2)
    out[labels == 0] = 0
    out[labels == head_k] = 1
    return out


def triple_sample_weights(
    labels: torch.Tensor,
    head_k: int,
    device: torch.device,
    w_normal: float = 1.0,
    w_fault: float = 1.0,
    w_other: float = 1.0,
) -> torch.Tensor:
    """按三态常数权重，返回 [B]。"""
    st = triple_state_batch(labels, head_k).to(device)
    w = torch.ones_like(st, dtype=torch.float32, device=device)
    w[st == 0] = float(w_normal)
    w[st == 1] = float(w_fault)
    w[st == 2] = float(w_other)
    return w


def fi_triple_edl_mse_loss(
    logits: torch.Tensor,
    y_triple: torch.Tensor,
    epoch_num: int,
    annealing_step: int,
    device: torch.device,
    sample_weight: torch.Tensor | None = None,
    kl_weight: float = 1.0,
    zero_kl_for_other_fault: bool = True,
) -> torch.Tensor:
    """
    EDL-MSE：数据项同 edl_mse；若 zero_kl_for_other_fault=True，对 y=[0,0] 的样本 KL 项置零。
    """
    evidence = relu_evidence(logits)
    alpha = evidence + 1
    per_ll = loglikelihood_loss(y_triple, alpha, device=device)

    if annealing_step is None or annealing_step <= 0:
        annealing_coef = torch.tensor(0.0, dtype=torch.float32, device=device)
    else:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32, device=device),
            torch.tensor(float(epoch_num) / float(annealing_step), dtype=torch.float32, device=device),
        )
    kl_alpha = (alpha - 1) * (1.0 - y_triple) + 1.0
    per_kl = annealing_coef * float(kl_weight) * kl_divergence(kl_alpha, 2, device=device)
    if zero_kl_for_other_fault:
        not_other = (y_triple.sum(dim=1, keepdim=True) > 0).float()
        per_kl = per_kl * not_other
    per_total = per_ll + per_kl
    if sample_weight is not None:
        sw = sample_weight.to(device).float().view(-1, 1)
        return (sw * per_total).sum() / sw.sum().clamp(min=1e-8)
    return per_total.mean()


def fi_triple_preds_and_ood(
    p0_stack: torch.Tensor,
    p1_stack: torch.Tensor,
    u_stack: torch.Tensor,
    tau_normal: float = 0.5,
    tau_fault: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fi-only 三态推理（无 model0）。
    p0_stack, p1_stack, u_stack: [B, M]，M=K-1，列 m 对应故障类索引 m+1。

    闭集预测 preds: 0=正常；否则在 p1>tau_fault 的头中取 u 最小，pred = 列索引+1。
    全体正常模式：∀m p0>tau_normal 且 p1<tau_fault -> pred=0；
      此时 ood_score 为各头 vacuity u_k 的 max（非 mean），避免平均压低分数。
    至少一头 p1>tau_fault：ood_score 为胜者头的 u。
    若无 p1>tau_fault 且非全体正常：ood_score=1.0，pred=argmax p1+1（闭集占位）。
    """
    device = p0_stack.device
    B, _M = p0_stack.shape
    dtype = p0_stack.dtype

    all_normal = ((p0_stack > tau_normal) & (p1_stack < tau_fault)).all(dim=1)
    valid_fault = p1_stack > tau_fault
    n_valid = valid_fault.long().sum(dim=1)

    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    u_masked = torch.where(valid_fault, u_stack, inf)
    rel_min = u_masked.argmin(dim=1)
    fault_pred = rel_min + 1
    _max_p1, argmax_p1 = p1_stack.max(dim=1)

    row = torch.arange(B, device=device)
    winner_u = u_stack[row, rel_min]
    preds = torch.where(
        all_normal,
        torch.zeros(B, dtype=torch.long, device=device),
        torch.where(n_valid >= 1, fault_pred.long(), argmax_p1.long() + 1),
    )
    ood_score = torch.where(
        all_normal,
        u_stack.max(dim=1).values,
        torch.where(n_valid >= 1, winner_u, torch.ones(B, device=device, dtype=dtype)),
    )
    return preds, ood_score


def discover_epochs_fi(checkpoint_dir: str, K: int) -> list[int]:
    """epochs/<e>/ 下需存在 model_1.pth .. model_{K-1}.pth"""
    import os

    epochs_dir = os.path.join(checkpoint_dir, "epochs")
    if not os.path.isdir(epochs_dir):
        return []
    epochs = []
    for name in os.listdir(epochs_dir):
        if not name.isdigit():
            continue
        e = int(name)
        sub = os.path.join(epochs_dir, name)
        if os.path.isdir(sub) and all(
            os.path.exists(os.path.join(sub, f"model_{k}.pth")) for k in range(1, K)
        ):
            epochs.append(e)
    return sorted(epochs)
