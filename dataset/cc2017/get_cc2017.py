# file: get_cc2017.py
# this script downloads the raw cc2017 dataset from Hugging Face
# run with `python -m dataset.cc2017.get_cc2017`

from __future__ import annotations

# [SETUP] ===============================
TARGET_PATH = './data/cc2017'
HF_HOME = './data/cache/huggingface'
HF_ENDPOINT = None
DATASET_REPO = 'gongzx/cc2017_dataset'
# =======================================

import fnmatch
from pathlib import Path

from dataset.hf_downloader import HF_REPO


def download_files(repo_id: str, repo_type: str, target: Path,
                   patterns: list[str], revision: str = 'main') -> None:
    # Check each allowlist pattern, so unavailable assets cannot silently succeed.
    repo = HF_REPO(repo_id = repo_id, repo_type = repo_type, revision = revision,
                   hf_home = HF_HOME, hf_endpoint = HF_ENDPOINT)
    files = repo.get_file_tree()
    missing = [pattern for pattern in patterns if not any(fnmatch.fnmatch(name, pattern) for name in files)]
    if missing:
        raise FileNotFoundError(f'{repo_id} ({revision}) has no files matching {missing}. '
                                'Obtain these upstream assets before running the dependent stage.')
    selected = [name for name in files if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)]
    print(f'\033[1;34m[cc2017] INFO\033[0m: Downloading {len(selected)} files from {repo_id}.', flush = True)
    repo.download(target_path = str(target), keep_patterns = selected)
    absent = [name for name in selected if not (target / name).is_file()]
    if absent:
        raise FileNotFoundError(f'Incomplete download in {target}: {absent}')


if __name__ == '__main__':
    target = Path(TARGET_PATH)
    target.mkdir(parents = True, exist_ok = True)

    # Download the raw tensors without constructing a database or changing signals.
    download_files(DATASET_REPO, 'dataset', target, ['*.pt', '*.pth', '*.bin'])
    print(f'\033[1;32m[cc2017] OK\033[0m: Dataset saved under {target}.', flush = True)
