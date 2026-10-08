# file: get_CineBrain.py
# this script downloads the CineBrain dataset from Hugging Face
from __future__ import annotations

# [REPO INFO] ===========================
REPO_ID = "Fudan-fMRI/CineBrain"
TARGET_PATH = "./data/CineBrain"
IGNORE_PATTERNS = None
HF_HOME = "./data/cache/huggingface"
HF_ENDPOINT = None
# =======================================

# import libraries
import os
import subprocess
from dataset.hf_downloader import *

# Entry point
# run with `python -m dataset.CineBrain.get_CineBrain`
if __name__ == "__main__":

    # create HF dataset obj
    dataset = HF_REPO(
        repo_id = REPO_ID,
        repo_type = "dataset",
        hf_home = HF_HOME,
        hf_endpoint = HF_ENDPOINT,
    )

    # List files in the dataset
    files = dataset.get_file_tree()
    print(f"\033[1;34m[HF_REPO] INFO\033[0m: Files in {REPO_ID}:")
    for f in files:
        print(f"  - {f}")

    # Download the dataset
    dataset.download(
        target_path = TARGET_PATH,
        ignore_patterns = IGNORE_PATTERNS,
    )

    # Go to the target dir and check integrity
    # get files under the target path
    downloaded_files = []
    for root, dirs, files in os.walk(TARGET_PATH):
        for file in files:
            downloaded_files.append(os.path.relpath(os.path.join(root, file), TARGET_PATH))
    # check if all files are downloaded
    missing_files = set(files) - set(downloaded_files)
    if missing_files:
        print(f"\033[1;31m[HF_REPO] ERROR\033[0m: Missing files in {TARGET_PATH}:")
        for f in missing_files:
            print(f"  - {f}")
# [ERROR TERMINATION]
        exit(1)

    # for all those tars, try to untar
    tar_count = 0
    for file in downloaded_files:
        if file.endswith(".tar"):
            tar_path = os.path.join(TARGET_PATH, file)
            print(f"\033[1;34m[HF_REPO] INFO\033[0m: Extracting {tar_path}.")
            try:
                subprocess.run(["tar", "-xf", tar_path, "-C", TARGET_PATH], check = True)
            except subprocess.CalledProcessError as e:
                print(f"\033[1;31m[HF_REPO] ERROR\033[0m: Failed to extract {tar_path}. Error: {e}")
# [ERROR TERMINATION]
                exit(1)
            print(f"\033[1;32m[HF_REPO] OK\033[0m: Extracted {tar_path}.")
            tar_count += 1

    # print a summary of the downloaded files
    print(f"\n\033[1;32m[HF_REPO] OK\033[0m: Downloaded {len(downloaded_files)} files to {TARGET_PATH}.")
    print(f"\033[1;32m[HF_REPO] OK\033[0m: Extracted {tar_count} tar files.")
