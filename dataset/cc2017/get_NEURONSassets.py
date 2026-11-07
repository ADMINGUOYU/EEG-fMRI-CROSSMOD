# file: get_NEURONSassets.py
# this script downloads the pretrained weights and annotations required by NEURONS from Hugging Face
# run with `python -m dataset.cc2017.get_NEURONSassets`
# run before NEURONS training/reconstruction; independent of get_cc2017.py
from __future__ import annotations

# [SETUP] ===============================
TARGET_PATH = './data/cc2017/pretrained_weights'
HF_HOME = './data/cache/huggingface'
HF_ENDPOINT = None
DOWNLOAD_ANNOTATIONS = True
ANNOTATION_PATH = './data/cc2017'
ANNOTATION_REPO = 'McGregorW/NEURONS'
ANNOTATION_PATTERNS = ['masks/*', 'qwen_annotation/*']
DELETE_COMPRESSED_FILES_WHEN_DONE = False
DOWNLOAD_MINDEYE = True
DOWNLOAD_ANIMATEDIFF = True
# Optional trained NEURONS weights; leave empty when training from scratch.
NEURONS_PATTERNS: list[str] = []
# Example: ['EXP/exp_neurons/subj_1/checkpoints/*.pth']
NEURONS_PATH = './data/cc2017/NEURONS'
# =======================================

import fnmatch
from pathlib import Path
import tarfile

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
    print(f'\033[1;34m[NEURONS] INFO\033[0m: Downloading {len(selected)} files from {repo_id}.', flush = True)
    repo.download(target_path = str(target), keep_patterns = selected)
    absent = [name for name in selected if not (target / name).is_file()]
    if absent:
        raise FileNotFoundError(f'Incomplete download in {target}: {absent}')


if __name__ == '__main__':
    target = Path(TARGET_PATH)
    target.mkdir(parents = True, exist_ok = True)

    # Match the MindEye filenames expected by the original NEURONS checkpoints.
    if DOWNLOAD_MINDEYE:
        aliases = {
            'train_logs/final_subj01_pretrained_40sess_24bs/last.pth': 'last.pth',
            'convnext_xlarge_alpha0.75_fullckpt.pth': 'convnext_xlarge_alpha0.75_fullckpt.ckpt',
        }
        download_files('pscotti/mindeyev2', 'dataset', target,
                       ['unclip6_epoch0_step110000.ckpt', 'sd_image_var_autoenc.pth', *aliases])
        for source, destination in aliases.items():
            # A symlink avoids duplicating multi-gigabyte checkpoints.
            link = target / destination
            if not link.exists():
                link.symlink_to(source)

    # Keep the original AnimateDiff asset grouping used by the inference config.
    if DOWNLOAD_ANIMATEDIFF:
        download_files('guoyww/animatediff', 'model', target / 'Motion_Module',
                       ['v3_sd15_adapter.ckpt', 'v3_sd15_mm.ckpt'])
        download_files('guoyww/animatediff', 'model', target / 'SparseCtrl',
                       ['v3_sd15_sparsectrl_rgb.ckpt'])
        download_files('guoyww/animatediff_t2i_backups', 'model', target / 'DreamBooth_LoRA',
                       ['realisticVisionV60B1_v51VAE.safetensors'])

    # Published experiment checkpoints are opt-in; the full repository exceeds 60 GB.
    if NEURONS_PATTERNS:
        download_files('McGregorW/NEURONS', 'model', Path(NEURONS_PATH), NEURONS_PATTERNS)

    # Store task annotations beside the raw dataset and extract the supplied masks.
    if DOWNLOAD_ANNOTATIONS:
        annotation_path = Path(ANNOTATION_PATH)
        download_files(ANNOTATION_REPO, 'model', annotation_path, ANNOTATION_PATTERNS)
        for archive in sorted((annotation_path / 'masks').glob('*.tar.gz')):
            print(f'\033[1;34m[NEURONS] INFO\033[0m: Extracting {archive}.', flush = True)
            with tarfile.open(archive) as handle:
                handle.extractall(archive.parent, filter = 'data')
            if DELETE_COMPRESSED_FILES_WHEN_DONE:
                archive.unlink()
        print(f'\033[1;32m[NEURONS] OK\033[0m: Annotations saved under {annotation_path}.', flush = True)
    print(f'\033[1;32m[NEURONS] OK\033[0m: Assets saved under {target}.', flush = True)
