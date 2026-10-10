# file: get_CineBrain.py
# this script downloads the CineBrain dataset from Hugging Face
# run with `python -m dataset.CineBrain.get_CineBrain`
from __future__ import annotations

# [REPO INFO] ===========================
REPO_ID = "Fudan-fMRI/CineBrain"
TARGET_PATH = "./data/CineBrain"
HF_HOME = "./data/cache/huggingface"
HF_ENDPOINT = None
DELETE_COMPRESSED_FILES_WHEN_DONE = False
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

    # Do file list processing if needed
    finalised_files = files
    print(f"\033[1;34m[HF_REPO] INFO\033[0m: Files to download ({REPO_ID}):")
    for f in finalised_files:
        print(f"  - {f}")

    # Download the dataset
    dataset.download(
        target_path = TARGET_PATH,
        keep_patterns = finalised_files,
    )

    # Go to the target dir and check integrity
    # get files under the target path
    downloaded_files = []
    for root, dirs, files in os.walk(TARGET_PATH):
        for file in files:
            downloaded_files.append(os.path.relpath(os.path.join(root, file), TARGET_PATH))
    # check if all files are downloaded
    missing_files = set(finalised_files) - set(downloaded_files)
    if missing_files:
        print(f"\033[1;31m[HF_REPO] ERROR\033[0m: Missing files in {TARGET_PATH}:")
        for f in missing_files:
            print(f"  - {f}")
# [ERROR TERMINATION]
        exit(1)

    # for all those tars, try to untar
    tar_count = 0
    # for those .zip files, try to unzip
    zip_count = 0
    for file in downloaded_files:
        # get file path and path to file where we are decompressing
        filepath_absolute = os.path.join(TARGET_PATH, file)
        decompress_path_absolute = os.path.dirname(filepath_absolute)

        if file.endswith(".tar"):
            print(f"\033[1;34m[HF_REPO] INFO\033[0m: Extracting {filepath_absolute}.")
            try:
                subprocess.run(["tar", "-xf", filepath_absolute, "-C", decompress_path_absolute], check = True)
            except subprocess.CalledProcessError as e:
                print(f"\033[1;31m[HF_REPO] ERROR\033[0m: Failed to extract {filepath_absolute}. Error: {e}")
# [ERROR TERMINATION]
                exit(1)
            print(f"\033[1;32m[HF_REPO] OK\033[0m: Extracted {filepath_absolute}.")
            tar_count += 1
        if file.endswith(".zip"):
            print(f"\033[1;34m[HF_REPO] INFO\033[0m: Extracting {filepath_absolute}.")
            try:
                subprocess.run(["unzip", "-o", filepath_absolute, "-d", decompress_path_absolute], check = True)
            except subprocess.CalledProcessError as e:
                print(f"\033[1;31m[HF_REPO] ERROR\033[0m: Failed to extract {filepath_absolute}. Error: {e}")
# [ERROR TERMINATION]
                exit(1)
            print(f"\033[1;32m[HF_REPO] OK\033[0m: Extracted {filepath_absolute}.")
            zip_count += 1

    # remove some bad bad files from macOS
    bad_files = [".DS_Store", "__MACOSX"]
    for bad_file in bad_files:
        bad_file_path = os.path.join(TARGET_PATH, bad_file)
        if os.path.exists(bad_file_path):
            print(f"\033[1;34m[HF_REPO] INFO\033[0m: Removing {bad_file_path}.")
            try:
                if os.path.isdir(bad_file_path):
                    subprocess.run(["rm", "-rf", bad_file_path], check = True)
                else:
                    os.remove(bad_file_path)
            except Exception as e:
                print(f"\033[1;31m[HF_REPO] ERROR\033[0m: Failed to remove {bad_file_path}. Error: {e}")
    # remove compressed files if needed
    if DELETE_COMPRESSED_FILES_WHEN_DONE:
        for file in finalised_files:
            if file.endswith(".tar") or file.endswith(".zip"):
                compressed_file_path = os.path.join(TARGET_PATH, file)
                if os.path.exists(compressed_file_path):
                    print(f"\033[1;34m[HF_REPO] INFO\033[0m: Removing {compressed_file_path}.")
                    try:
                        os.remove(compressed_file_path)
                    except Exception as e:
                        print(f"\033[1;31m[HF_REPO] ERROR\033[0m: Failed to remove {compressed_file_path}. Error: {e}")

    # print a summary of the downloaded files
    print(f"\n\033[1;32m[HF_REPO] OK\033[0m: Downloaded {len(downloaded_files)} files to {TARGET_PATH}.")
    print(f"\033[1;32m[HF_REPO] OK\033[0m: Extracted {tar_count} tar files.")
    print(f"\033[1;32m[HF_REPO] OK\033[0m: Extracted {zip_count} zip files.")
