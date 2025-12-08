import os


TARGET_NAMES = {
    "1. 原始时域图.png": ["原始时域图.png"],
    "2. 原始频域图.png": ["原始频域图.png"],
    "3. 带通时域图.png": ["带通时域图.png", "带通滤波时域图.png"],
    "4. 带通频域图.png": ["带通频域图.png", "带通滤波频域图.png"],
    "5. 包络时域图.png": ["包络时域图.png", "包络谱时域图.png"],
    "6. 包络谱.png": ["包络谱.png", "包络谱频域图.png"],
    # 先把最老的总图名统一成带序号的，随后再附加文件名
    "0. 分析全流程图.png": ["分析全流程图.png"],
}


def rename_dir(dir_path: str) -> None:
    # 若目标名已存在则不覆盖；按映射顺序尝试重命名
    for new_name, old_candidates in TARGET_NAMES.items():
        new_path = os.path.join(dir_path, new_name)
        if os.path.exists(new_path):
            continue
        for old_name in old_candidates:
            old_path = os.path.join(dir_path, old_name)
            if os.path.exists(old_path):
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except OSError as e:
                    print(f"Skip (error): {old_path} -> {new_path}: {e}")
                break

    # 将总图名改为包含文件名（目录名即文件 stem）
    stem = os.path.basename(dir_path)
    src_candidates = [
        os.path.join(dir_path, "0. 分析全流程图.png"),
        os.path.join(dir_path, "分析全流程图.png"),
    ]
    dst = os.path.join(dir_path, f"0. 分析全流程图 - {stem}.png")
    if not os.path.exists(dst):
        for src in src_candidates:
            if os.path.exists(src):
                try:
                    os.rename(src, dst)
                    print(f"Renamed: {src} -> {dst}")
                except OSError as e:
                    print(f"Skip (error): {src} -> {dst}: {e}")
                break


def rename_all(results_root: str = "./results") -> None:
    if not os.path.isdir(results_root):
        print(f"Results dir not found: {results_root}")
        return
    for entry in os.listdir(results_root):
        subdir = os.path.join(results_root, entry)
        if os.path.isdir(subdir):
            rename_dir(subdir)


if __name__ == "__main__":
    rename_all()


