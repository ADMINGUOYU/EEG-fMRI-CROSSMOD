# file: vis_CineBrain.py
# visualise CineBrain dataset and basic data parameters
# run with: `python -m dataset.CineBrain.vis_CineBrain`

# NOTE:
# dataset location: data/CineBrain
# folder structure:
"""
data/CineBrain
├── clips <- 000000.mp4 to 008099.mp4, (33 frames per clip)
├── sub-0001
│   ├── eeg_02
│   ├── sub-0001 <- NOTE: this is unnecessary (fMRI raw)
│   │   ├── anat
│   │   ├── fmap
│   │   └── func
│   └── visual_audio
├── sub-0002
│   ├── eeg_02
│   └── visual_audio
├── sub-0003
│   ├── eeg_02
│   └── visual_audio
├── sub-0004
│   ├── eeg_02
│   └── visual_audio
├── sub-0005
│   ├── eeg_02
│   ├── sub-0005 <- NOTE: this is unnecessary (fMRI raw)
│   │   ├── anat
│   │   ├── fmap
│   │   └── func
│   └── visual_audio
├── sub-0006
│   ├── eeg_02
│   └── visual_audio
├── sub01_beh <- have information of the 20 trails a subject watched (i.e. which episode)
├── sub02_beh
├── sub03_beh
├── sub04_beh
├── sub05_beh
└── sub06_beh
"""

# import libraries
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset.CineBrain.fMRI_Helper import *
from dataset.CineBrain.EEG_Helper import *


# Entry point
if __name__ == "__main__":

    # we first explore the file tree and count the number
    # of files under EEG dir and fMRI dir
    dataset_dir = "./data/CineBrain"
    EEG_dir_name = "eeg_02"
    fMRI_dir_name = "visual_audio"
    subject_dir_pattern = re.compile(r"sub-\d{4}")
    vis_save_path = "./dataset/CineBrain/visualisation/"

    # NOTE: shape check size, to check all, set to None
    number_of_files_to_check = 10

    # dict log
    log_count = {
        "subject": [],
        "EEG_file_count": [],
        "fMRI_file_count": [],
        "missing_EEG": [],
        "missing_fMRI": []
    }
    log_shape = {
        "subject": [],
        "EEG_shape": [],
        "fMRI_shape": []
    }

    # iterate through the dataset folder
    for subject_dir in tqdm(os.listdir(dataset_dir), 
                            desc = "Processing subjects",
                            position = 0, leave = True):
        subject_path = os.path.join(dataset_dir, subject_dir)
        if os.path.isdir(subject_path) and subject_dir_pattern.match(subject_dir):
            # count EEG files
            EEG_path = os.path.join(subject_path, EEG_dir_name)
            # get the set of names, they all use <idx>.npy
            EEG_set = set(os.listdir(EEG_path)) if os.path.exists(EEG_path) else set()
            # for each EEG file, get the shape (we only check files .npy)
            # only check first 10 files
            EEG_shapes = []
            for eeg_file in tqdm(list(EEG_set)[:number_of_files_to_check],
                                 desc = "Iter. EEG files",
                                 position = 1, leave = False):
                if eeg_file.endswith(".npy"):
                    eeg_file_path = os.path.join(EEG_path, eeg_file)
                    eeg_shape = np.load(eeg_file_path).shape
                    EEG_shapes.append(eeg_shape)
            # make it a set of unique shapes
            EEG_shapes_set = set(EEG_shapes)
            # turn it back into a list of strings for logging
            EEG_shapes_str = [str(shape) for shape in EEG_shapes_set]

            # count fMRI files
            fMRI_path = os.path.join(subject_path, fMRI_dir_name)
            # get the set of names, they all use <idx>.npy
            fMRI_set = set(os.listdir(fMRI_path)) if os.path.exists(fMRI_path) else set()
            # for each fMRI file, get the shape (we only check files .npy)
            fMRI_shapes = []
            for fmri_file in tqdm(list(fMRI_set)[:number_of_files_to_check],
                                  desc = "Iter. fMRI files",
                                  position = 1, leave = False):
                if fmri_file.endswith(".npy"):
                    fmri_file_path = os.path.join(fMRI_path, fmri_file)
                    fmri_shape = np.load(fmri_file_path).shape
                    fMRI_shapes.append(fmri_shape)         
            # make it a set of unique shapes
            fMRI_shapes_set = set(fMRI_shapes)
            # turn it back into a list of strings for logging
            fMRI_shapes_str = [str(shape) for shape in fMRI_shapes_set]

            # calculate missing files
            missing_EEG: list[str] = list(fMRI_set - EEG_set)
            missing_fMRI: list[str] = list(EEG_set - fMRI_set)

            # log the counts
            log_count["subject"].append(subject_dir)
            log_count["EEG_file_count"].append(len(EEG_set))
            log_count["fMRI_file_count"].append(len(fMRI_set))
            log_count["missing_EEG"].append(missing_EEG)
            log_count["missing_fMRI"].append(missing_fMRI)

            # log the shapes
            log_shape["subject"].append(subject_dir)
            log_shape["EEG_shape"].append(EEG_shapes_str)
            log_shape["fMRI_shape"].append(fMRI_shapes_str)

    # convert to DataFrame
    df_count = pd.DataFrame(log_count)
    print("\033[1;34m[INFO]\033[0m\033[1m Summary of EEG and fMRI file counts per subject:\033[0m")
    print(df_count)
    df_shape = pd.DataFrame(log_shape)
    print(f"\033[1;34m[INFO]\033[0m\033[1m Summary of EEG and fMRI file shapes per subject:\033[0m (checked {number_of_files_to_check if number_of_files_to_check is not None else 'all'} files)")
    print(df_shape)

    # visualise fMRI data for subject 0001 - 0th segment
    print("\033[1;34m[INFO]\033[0m\033[1m Visualising fMRI data for subject 0001 - 0th segment:\033[0m")
    subj01_0 = np.load(os.path.join(dataset_dir, "sub-0001", fMRI_dir_name, "0.npy"))
    visualise_fmri_signal_static(
        subj01_0,
        plot_title = "Subject 0001 - 0th segment fMRI signal",
        save_path = os.path.join(vis_save_path, "sub-0001", "0_fMRI", "0_static.png")
    )
    visualize_fmri_mesh(
        subj01_0,
        output_dir = os.path.join(vis_save_path, "sub-0001", "0_fMRI"),
    )
    del subj01_0
    print("\033[1;34m[INFO]\033[0m\033[1m Visualisation saved to:\033[0m", os.path.join(vis_save_path, "sub-0001"))

    # visualise EEG data for subject 0001 - 0th segment
    print("\033[1;34m[INFO]\033[0m\033[1m Visualising EEG data for subject 0001 - 0th segment:\033[0m")
    subj01_0_eeg = np.load(os.path.join(dataset_dir, "sub-0001", EEG_dir_name, "0.npy"))
    subj01_0_eeg = subj01_0_eeg[:64, :]  # first 64 is the actual EEG channel
    visualise_eeg_signal_static(
        raw_data = subj01_0_eeg,
        raw_fs = 1000,
        plot_title = "Subject 0001 - 0th segment EEG signal",
        output_dir = os.path.join(vis_save_path, "sub-0001", "0_EEG"),
        processed_data = preprocess_eeg(subj01_0_eeg),
        processed_fs = 200,
        processed_low_cut = LOW_CUT_HZ,
        processed_high_cut = HIGH_CUT_HZ
    )
    del subj01_0_eeg
    print("\033[1;34m[INFO]\033[0m\033[1m Visualisation saved to:\033[0m", os.path.join(vis_save_path, "sub-0001"))
