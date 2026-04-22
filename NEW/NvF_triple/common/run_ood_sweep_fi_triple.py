from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_class_id(name: str) -> int:
    digits = []
    for ch in name:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        raise ValueError(f"Cannot parse class id from: {name!r}")
    return int("".join(digits))


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def build_stage_cfgs(
    base_stage1: dict,
    base_stage2: dict,
    known_classes: list[str],
    unknown_class: str,
    ood_id: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> tuple[dict, dict]:
    stage1 = yaml.safe_load(yaml.safe_dump(base_stage1, allow_unicode=True))
    stage2 = yaml.safe_load(yaml.safe_dump(base_stage2, allow_unicode=True))

    stage1["data"]["openset"]["known_classes"] = list(known_classes)
    stage1["data"]["openset"]["unknown_classes"] = [unknown_class]
    stage2["data"]["openset"]["known_classes"] = list(known_classes)
    stage2["data"]["openset"]["unknown_classes"] = [unknown_class]

    stage1_ckpt = f"checkpoints/LaoDA/bench_NvF_fi_triple/OOD_{ood_id}/stage1"
    stage2_ckpt = f"checkpoints/LaoDA/bench_NvF_fi_triple/OOD_{ood_id}/stage2_joint"

    stage1["train"]["checkpoint_dir"] = stage1_ckpt
    stage2["train"]["checkpoint_dir"] = stage2_ckpt
    stage2["train"]["fi_joint_finetune"]["input_checkpoint_dir"] = stage1_ckpt

    # DataLoader speed options for both stages.
    stage1["train"]["num_workers"] = int(num_workers)
    stage1["train"]["pin_memory"] = bool(pin_memory)
    stage1["train"]["persistent_workers"] = bool(persistent_workers)
    stage2["train"]["num_workers"] = int(num_workers)
    stage2["train"]["pin_memory"] = bool(pin_memory)
    stage2["train"]["persistent_workers"] = bool(persistent_workers)

    # No per-epoch checkpoints, keep only best + last.
    stage1["train"]["save_every_epoch"] = False
    stage2["train"]["save_every_epoch"] = False

    # Stage2: disable white-noise injection, keep data augmentation.
    if "white_noise_injection" in stage2["train"]["fi_joint_finetune"]:
        stage2["train"]["fi_joint_finetune"]["white_noise_injection"]["enable"] = False

    return stage1, stage2


def prepare_last_checkpoint_dir(stage2_dir: Path, k: int) -> Path:
    last_dir = stage2_dir / "last_models"
    last_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, k):
        src = stage2_dir / f"model_last_{i}.pth"
        if not src.exists():
            raise FileNotFoundError(f"Missing last model file: {src}")
        dst = last_dir / f"model_{i}.pth"
        shutil.copy2(src, dst)
    return last_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep OOD classes 2..9 for Fi-triple two-stage training and testing."
    )
    parser.add_argument(
        "--stage1_config",
        type=str,
        default="configs/bench_NvF_LaoDA_fi_triple.yaml",
    )
    parser.add_argument(
        "--stage2_config",
        type=str,
        default="configs/bench_NvF_LaoDA_fi_triple_joint.yaml",
    )
    parser.add_argument(
        "--ood_ids",
        type=str,
        default="5,6,8,9",
        help="Comma-separated OOD class ids to run, e.g. 2,7,9",
    )
    parser.add_argument(
        "--generated_config_dir",
        type=str,
        default="configs/generated_ood_sweep",
    )
    parser.add_argument(
        "--threshold_from_val",
        action="store_true",
        help="Pass --threshold_from_val when running test_NvF_fi_triple.py",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers for stage1/stage2.")
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable DataLoader pin_memory for stage1/stage2.",
    )
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable DataLoader persistent_workers for stage1/stage2 (effective when num_workers>0).",
    )
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    stage1_cfg_path = (workdir / args.stage1_config).resolve()
    stage2_cfg_path = (workdir / args.stage2_config).resolve()
    generated_dir = (workdir / args.generated_config_dir).resolve()
    generated_dir.mkdir(parents=True, exist_ok=True)

    base_stage1 = load_yaml(stage1_cfg_path)
    base_stage2 = load_yaml(stage2_cfg_path)

    stage1_known = base_stage1["data"]["openset"]["known_classes"]
    stage1_unknown = base_stage1["data"]["openset"].get("unknown_classes", [])
    all_classes = stage1_known + stage1_unknown
    id2name = {parse_class_id(name): name for name in all_classes}

    requested_ood_ids = [int(x.strip()) for x in args.ood_ids.split(",") if x.strip()]
    for ood_id in requested_ood_ids:
        if ood_id not in id2name:
            raise ValueError(f"OOD id {ood_id} is not found in configs classes: {sorted(id2name.keys())}")
        if ood_id == 1:
            raise ValueError("Class 1 is normal class, cannot be set as OOD in this sweep.")

    for ood_id in requested_ood_ids:
        ood_name = id2name[ood_id]
        known_classes = [id2name[i] for i in sorted(id2name.keys()) if i != ood_id]
        print(f"\n========== OOD_{ood_id} ({ood_name}) ==========")

        stage1_cfg, stage2_cfg = build_stage_cfgs(
            base_stage1=base_stage1,
            base_stage2=base_stage2,
            known_classes=known_classes,
            unknown_class=ood_name,
            ood_id=ood_id,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )

        stage1_gen_path = generated_dir / f"bench_NvF_LaoDA_fi_triple_OOD_{ood_id}.yaml"
        stage2_gen_path = generated_dir / f"bench_NvF_LaoDA_fi_triple_joint_OOD_{ood_id}.yaml"
        save_yaml(stage1_cfg, stage1_gen_path)
        save_yaml(stage2_cfg, stage2_gen_path)

        run_cmd([sys.executable, "train_nvf_fi_triple.py", "--config", str(stage1_gen_path)], cwd=workdir)
        run_cmd([sys.executable, "train_nvf_fi_joint.py", "--config", str(stage2_gen_path)], cwd=workdir)

        stage2_dir = (workdir / stage2_cfg["train"]["checkpoint_dir"]).resolve()
        best_out = stage2_dir / "test_fi" / "best"
        last_out = stage2_dir / "test_fi" / "last"

        test_cmd_best = [
            sys.executable,
            "test_NvF_fi_triple.py",
            "--config",
            str(stage2_gen_path),
            "--checkpoint",
            str(stage2_dir),
            "--output_dir",
            str(best_out),
        ]
        if args.threshold_from_val:
            test_cmd_best.append("--threshold_from_val")
        run_cmd(test_cmd_best, cwd=workdir)

        k = len(stage2_cfg["data"]["openset"]["known_classes"])
        last_ckpt_dir = prepare_last_checkpoint_dir(stage2_dir=stage2_dir, k=k)
        test_cmd_last = [
            sys.executable,
            "test_NvF_fi_triple.py",
            "--config",
            str(stage2_gen_path),
            "--checkpoint",
            str(last_ckpt_dir),
            "--output_dir",
            str(last_out),
        ]
        if args.threshold_from_val:
            test_cmd_last.append("--threshold_from_val")
        run_cmd(test_cmd_last, cwd=workdir)

    print("\nAll requested OOD sweeps completed.")


if __name__ == "__main__":
    main()
