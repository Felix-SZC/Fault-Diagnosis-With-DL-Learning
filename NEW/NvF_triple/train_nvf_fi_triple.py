# Fi-only 阶段一：仅训练 model_1..model_{K-1}，子集「正常+故障i」，标签 [1,0]/[0,1]
# 从 train_nvf.py 复制骨架；不修改 train_nvf.py

import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.utils.helpers import load_config, save_experiment_info
from common.utils.data_loader import NpyPackDataset
from common.utils.data_loader_1d import NpyPackDataset1D
from common.edl_losses import relu_evidence
from common.utils.edl_helpers import one_hot_embedding
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


def run_fi_only_training(config):
    data_config = config["data"]
    train_config = config["train"]
    model_config = config["model"]

    K = len(data_config["openset"]["known_classes"])
    if K < 2:
        raise ValueError("Fi-only 至少需要 1 个正常类 + 1 个故障类（K>=2）")

    batch_size = train_config.get("batch_size", 32)
    backbone_type = model_config.get("type", "LaoDA")
    train_dataset = get_dataset(data_config, backbone_type, split="train")
    val_dataset = get_dataset(data_config, backbone_type, split="test")

    device = torch.device(train_config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    annealing_step = int(train_config.get("edl_annealing_step", 10))
    kl_weight = float(train_config.get("kl_weight", 1.0))
    from common.edl_losses import edl_mse_loss, edl_digamma_loss, edl_log_loss

    edl_loss_type = train_config.get("edl_loss_type", "mse")
    if edl_loss_type == "mse":
        criterion = edl_mse_loss
    elif edl_loss_type == "digamma":
        criterion = edl_digamma_loss
    elif edl_loss_type == "log":
        criterion = edl_log_loss
    else:
        raise ValueError(f"不支持的 EDL 损失: {edl_loss_type}")

    learning_rate = float(train_config.get("learning_rate", 0.0001))
    num_epochs = int(train_config.get("num_epochs", 60))
    checkpoint_dir = train_config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    scheduler_conf = train_config.get("scheduler", {})
    use_scheduler = scheduler_conf.get("use_scheduler", False)
    sch_type = scheduler_conf.get("scheduler_type", "StepLR") if use_scheduler else None
    step_size = int(scheduler_conf.get("step_size", 40))
    gamma = float(scheduler_conf.get("gamma", 0.1))
    optimizer_type = train_config.get("optimizer", "Adam")
    momentum = float(train_config.get("momentum", 0.9))
    weight_decay = float(train_config.get("weight_decay", 1e-4))

    train_start_time = datetime.now()
    save_experiment_info(
        config,
        checkpoint_dir,
        model=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        start_time=train_start_time,
    )

    model_kw = {"num_classes": 2}
    print(f"Fi-only 阶段一：训练子模型 k=1..{K - 1}（不训练 model_0）")

    for k in range(1, K):
        print(f"\n{'=' * 60}")
        print(f"训练二分类模型 k = {k} / {K - 1}（正常 vs 第 {k} 类故障，子集）")
        train_indices = np.where((train_dataset.y == 0) | (train_dataset.y == k))[0]
        val_indices = np.where((val_dataset.y == 0) | (val_dataset.y == k))[0]
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        train_pos = int(np.sum(train_dataset.y[train_indices] == k))
        train_neg = int(np.sum(train_dataset.y[train_indices] == 0))
        val_pos = int(np.sum(val_dataset.y[val_indices] == k))
        val_neg = int(np.sum(val_dataset.y[val_indices] == 0))

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        print(f"  训练集: 正类(故障{k})={train_pos}, 负类(正常)={train_neg}")
        print(f"  验证集: 正类={val_pos}, 负类={val_neg}")

        torch.manual_seed(int(train_config.get("seed", 42)))
        model = get_model(backbone_type, **model_kw).to(device)
        if optimizer_type == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if use_scheduler and sch_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)

        best_val_acc = 0.0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            model.train()
            train_loss = train_correct = train_total = 0.0
            train_tp = train_fp = train_fn = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                binary_labels = (labels == k).long()
                optimizer.zero_grad()
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out
                y_binary = one_hot_embedding(binary_labels, 2).float().to(device)
                n_pos = (binary_labels == 1).sum().item()
                n_neg = (binary_labels == 0).sum().item()
                pos_w = n_neg / max(n_pos, 1)
                sample_w = torch.where(
                    binary_labels == 1,
                    torch.tensor(pos_w, device=device, dtype=torch.float),
                    torch.ones_like(binary_labels, device=device, dtype=torch.float),
                )
                loss = criterion(
                    logits, y_binary, epoch, 2, annealing_step, device, sample_weight=sample_w, kl_weight=kl_weight
                )
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    evidence = relu_evidence(logits)
                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    preds = (probs[:, 1] > 0.5).long()
                train_loss += loss.item() * inputs.size(0)
                train_correct += (preds == binary_labels).sum().item()
                train_total += inputs.size(0)
                train_tp += ((preds == 1) & (binary_labels == 1)).sum().item()
                train_fp += ((preds == 1) & (binary_labels == 0)).sum().item()
                train_fn += ((preds == 0) & (binary_labels == 1)).sum().item()

            train_loss /= max(len(train_subset), 1)
            train_acc = 100.0 * train_correct / max(train_total, 1)

            model.eval()
            val_loss = val_correct = val_total = 0.0
            val_tp = val_fp = val_fn = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    binary_labels = (labels == k).long()
                    out = model(inputs)
                    logits = out[0] if isinstance(out, tuple) else out
                    y_binary = one_hot_embedding(binary_labels, 2).float().to(device)
                    n_pos = (binary_labels == 1).sum().item()
                    n_neg = (binary_labels == 0).sum().item()
                    pos_w = n_neg / max(n_pos, 1)
                    sample_w = torch.where(
                        binary_labels == 1,
                        torch.tensor(pos_w, device=device, dtype=torch.float),
                        torch.ones_like(binary_labels, device=device, dtype=torch.float),
                    )
                    loss = criterion(
                        logits, y_binary, epoch, 2, annealing_step, device, sample_weight=sample_w, kl_weight=kl_weight
                    )
                    evidence = relu_evidence(logits)
                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    preds = (probs[:, 1] > 0.5).long()
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (preds == binary_labels).sum().item()
                    val_total += inputs.size(0)
                    val_tp += ((preds == 1) & (binary_labels == 1)).sum().item()
                    val_fp += ((preds == 1) & (binary_labels == 0)).sum().item()
                    val_fn += ((preds == 0) & (binary_labels == 1)).sum().item()

            val_loss /= max(len(val_subset), 1)
            val_acc = 100.0 * val_correct / max(val_total, 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch [{epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                path_k = os.path.join(checkpoint_dir, f"model_{k}.pth")
                torch.save({"state_dict": model.state_dict(), "k": k, "best_val_acc": best_val_acc}, path_k)

        path_k = os.path.join(checkpoint_dir, f"model_{k}.pth")
        if not os.path.exists(path_k):
            torch.save({"state_dict": model.state_dict(), "k": k, "best_val_acc": best_val_acc}, path_k)
        print(f"模型 k={k} 已保存: {path_k}, best_val_acc={best_val_acc:.2f}%")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Fi-only Model k={k} (Normal vs Fault {k})")
        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"], label="Val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(history["train_acc"], label="Train")
        ax2.plot(history["val_acc"], label="Val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(checkpoint_dir, f"training_curves_{k}.png"), dpi=150)
        plt.close(fig)

    save_experiment_info(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        history=None,
        best_val_acc=None,
        start_time=train_start_time,
        end_time=datetime.now(),
    )
    print(f"\nFi-only 阶段一完成（model_1..model_{K - 1}）。")


def main():
    parser = argparse.ArgumentParser(description="Fi-only NvF 阶段一：仅训练故障子模型")
    parser.add_argument("--config", type=str, default="configs/bench_NvF_LaoDA_fi_triple.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_fi_only_training(config)


if __name__ == "__main__":
    main()
