import os
import shutil
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIGS_DIR = os.path.join(ROOT, 'configs')
RESULTS_DIR = os.path.join(ROOT, 'results')
MAIN_CONFIG = os.path.join(CONFIGS_DIR, 'config.yaml')


def run_once() -> int:
    """Run signal_analysis.py once using current configs/config.yaml."""
    script = os.path.join(ROOT, 'signal_analysis.py')
    # Use the same Python interpreter
    return subprocess.call([sys.executable, script])


def main() -> None:
    if not os.path.isdir(RESULTS_DIR):
        print(f'Results dir not found: {RESULTS_DIR}')
        return

    # Backup original config if exists
    backup_path = None
    if os.path.exists(MAIN_CONFIG):
        backup_path = MAIN_CONFIG + '.bak'
        shutil.copyfile(MAIN_CONFIG, backup_path)
        print(f'Backed up original config to: {backup_path}')

    try:
        for entry in sorted(os.listdir(RESULTS_DIR)):
            subdir = os.path.join(RESULTS_DIR, entry)
            cfg = os.path.join(subdir, 'config.yaml')
            if not os.path.isdir(subdir) or not os.path.isfile(cfg):
                continue

            # Overwrite main config with this run's config
            shutil.copyfile(cfg, MAIN_CONFIG)
            print(f'Applied config: {cfg} -> {MAIN_CONFIG}')

            # Run analysis to regenerate figures/title
            code = run_once()
            if code != 0:
                print(f'Run failed for {entry} (exit={code}), continue to next.')
            else:
                print(f'Completed: {entry}')
    finally:
        # Restore original config if there was one
        if backup_path and os.path.exists(backup_path):
            shutil.copyfile(backup_path, MAIN_CONFIG)
            os.remove(backup_path)
            print('Restored original configs/config.yaml')


if __name__ == '__main__':
    main()


