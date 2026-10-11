# file: EEG_Helper.py
# helps anything with our EEG data (each clip 800 points - 0.8s/1,000 Hz)
# NOTE: DO NOT RUN THIS DIRECTLY

# import general libraries
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, welch
import mne

# EEG info
EEG_seq_len = 800  # each clip is 0.8s long, sampled at 1,000 Hz
EEG_sampling_rate = 1000  # Hz
NOTCH_FREQ_HZ = 50.0 # Hz, to remove electrical line noise
# Default preprocessing parameters from the CineBrain paper
LOW_CUT_HZ = 0.1
HIGH_CUT_HZ = 30.0
DOWNSAMPLE_RATE_HZ = 200 # A standard rate for processed EEG
FILTER_ORDER = 4
# random SEED
RNG_SEED = 42

# Preprocessing function for EEG data
def preprocess_eeg(
        eeg_data: np.ndarray,
        ecg_data: np.ndarray = None,
        original_fs: int = EEG_sampling_rate,
        new_fs: int = DOWNSAMPLE_RATE_HZ,
        low_cut: float = LOW_CUT_HZ,
        high_cut: float = HIGH_CUT_HZ
    ) -> np.ndarray:

    # check input dimensions
    if eeg_data.ndim != 2:
        raise ValueError(f"\033[1;31m[ERROR]\033[0m EEG data must be a 2D array (channels, samples). Got shape: {eeg_data.shape}")
    # log input dimensions
    eeg_channel_count = eeg_data.shape[0]

    # if ECG provided, concat to ECG to EEG and record each modality's channel count
    if ecg_data is not None:
        if ecg_data.ndim == 1:
            ecg_data = ecg_data[np.newaxis, :]  # Convert to 2D array with shape (1, n_samples)
        elif ecg_data.ndim != 2:
            raise ValueError(f"\033[1;31m[ERROR]\033[0m ECG data must be a 1D or 2D array. Got shape: {ecg_data.shape}")

        if ecg_data.shape[1] != eeg_data.shape[1]:
            raise ValueError(
                f"\033[1;31m[ERROR]\033[0m ECG and EEG must have the same sample count. EEG: {eeg_data.shape[1]}, ECG: {ecg_data.shape[1]}"
            )

        # Concatenate ECG and EEG data along the channel axis
        ecg_for_ica = ecg_data[:1, :]
        if ecg_data.shape[0] > 1:
            print(f"\033[1;33m[WARN]\033[0m Received {ecg_data.shape[0]} ECG channels. Using the first channel for ICA guidance.")
        combined_data = np.concatenate((eeg_data, ecg_for_ica), axis = 0)
        ecg_channel_count = ecg_data.shape[0]
    else:
        combined_data = eeg_data
        ecg_channel_count = 0
        ecg_for_ica = None
    
    # 1. Band-pass filtering
    nyquist_freq = 0.5 * original_fs
    low = low_cut / nyquist_freq
    high = high_cut / nyquist_freq
    b_band, a_band = butter(FILTER_ORDER, [low, high], btype = 'band')
    filtered_data = filtfilt(b_band, a_band, combined_data, axis = -1)

    # 2. Notch filtering to remove electrical line noise
    # A quality factor of 30 is standard for a narrow notch
    quality_factor = 30.0
    b_notch, a_notch = iirnotch(NOTCH_FREQ_HZ, quality_factor, original_fs)
    notched_data = filtfilt(b_notch, a_notch, filtered_data, axis = -1)

    # 3. Downsampling
    # ship if the new sampling rate is higher or equal to the original
    if new_fs >= original_fs:
        print(f"\033[1;33m[WARN]\033[0m New sampling rate ({new_fs} Hz) is not lower than original ({original_fs} Hz). Skipping downsampling.")
        preprocessed_data = notched_data
        current_fs = original_fs
    else:
        downsample_factor = int(original_fs / new_fs)
        preprocessed_data = notched_data[:, ::downsample_factor]
        current_fs = new_fs

    # 4. ECG-Informed ICA Artifact Removal
    if ecg_data is not None:
        # Calculate the total duration in seconds of the provided signal
        # we terminate the ICA if the signal is too short (so don't provide a single 0.8s clip if you're using CineBrain)
        total_duration_sec = preprocessed_data.shape[1] / current_fs
        THRESHOLD_ICA_SEC = 30  # seconds
        if total_duration_sec < THRESHOLD_ICA_SEC:
            print(f"\033[1;33m[WARN]\033[0m Total duration of EEG data ({total_duration_sec:.2f}s) is less than the threshold ({THRESHOLD_ICA_SEC}s)")
            # ask user if to continue or not
            user_input = input("\033[1;33m[WARN]\033[0m Do you want to continue with ICA artifact removal? (y/n): ")
            if user_input.lower() != 'y':
                print(f"\033[1;34m[INFO]\033[0m Skipping ICA artifact removal due to short duration.")
                # detach ECG data from EEG data and return only the EEG channels
                return preprocessed_data[:eeg_channel_count, :]

        # we start ICA artifact removal
        eeg_channels = preprocessed_data[:eeg_channel_count, :]
        ecg_channel = preprocessed_data[eeg_channel_count:eeg_channel_count + ecg_channel_count, :]

        ecg_channel_names = [f"ECG{idx + 1}" for idx in range(ecg_channel_count)]
        ch_names = [f"EEG{idx + 1}" for idx in range(eeg_channel_count)] + ecg_channel_names
        ch_types = ["eeg"] * eeg_channel_count + ["ecg"] * ecg_channel_count

        # Create MNE info block
        info = mne.create_info(ch_names = ch_names, sfreq = current_fs, ch_types = ch_types)

        # Stack channels to feed into MNE RawArray
        combined_matrix = np.vstack((eeg_channels, ecg_channel))
        raw = mne.io.RawArray(data = combined_matrix, info = info, verbose = False)

        if eeg_channel_count < 2:
            print("\033[1;33m[WARN]\033[0m ICA requires at least 2 EEG channels. Returning filtered EEG without ICA.")
            return eeg_channels

        # Fit ICA on the EEG channels only
        ica = mne.preprocessing.ICA(
            n_components = 0.95,
            method = "fastica",
            rng = RNG_SEED,
            max_iter = "auto"
        )

        try:
            try:
                ica.fit(inst = raw, picks = "eeg", verbose = False)
            except Exception:
                fallback_components = max(2, eeg_channel_count - 1)
                ica = mne.preprocessing.ICA(
                    n_components = fallback_components,
                    method = "fastica",
                    rng = RNG_SEED,
                    max_iter = "auto"
                )
                ica.fit(inst = raw, picks = "eeg", verbose = False)

            bad_component_indices = set()
            for ecg_name in ecg_channel_names:
                # Find and mark cardiac components using the cross-correlation of each ECG channel
                ecg_indices, _ = ica.find_bads_ecg(
                    inst = raw,
                    ch_name = ecg_name,
                    method = "correlation",
                    threshold = 0.35,
                    verbose = False
                )
                bad_component_indices.update(int(idx) for idx in ecg_indices)

            if len(bad_component_indices) > 0:
                bad_component_indices = sorted(bad_component_indices)
                print(f"\033[1;32m[OK]\033[0m Identified and removed {len(bad_component_indices)} unique cardiac component(s): {bad_component_indices}")
                ica.exclude = bad_component_indices
            else:
                print(f"\033[1;33m[INFO]\033[0m No components exceeded the ECG correlation threshold. Keeping all components.")

            # Reconstruct clean EEG channels
            cleaned_raw = ica.apply(inst = raw.copy(), verbose = False)
            final_output = cleaned_raw.get_data(picks = "eeg")
            return final_output
        except Exception as exc:
            print(f"\033[1;33m[WARN]\033[0m ICA artifact removal failed ({exc}). Returning filtered EEG without ICA.")
            return eeg_channels

    else:
        print(f"\033[1;34m[INFO]\033[0m No ECG data provided. Skipping ICA artifact removal.")
        return preprocessed_data

# Plotting function for EEG data
def visualise_eeg_signal_static(
        raw_data: np.ndarray,
        raw_fs: int,
        plot_title: str,
        output_dir: str,
        processed_data: np.ndarray = None,
        processed_fs: int = None,
        processed_low_cut: float = None,
        processed_high_cut: float = None,
    ) -> None:

    # if processed is given, other axes must be given too
    if processed_data is not None:
        if processed_fs is None \
            or processed_low_cut is None \
            or processed_high_cut is None:
            print(f"\033[1;31m[ERROR]\033[0m Processed data provided but processed configs are None.")
            return

    # get shapes
    num_channels, num_raw_points = raw_data.shape
    if processed_data is not None:
        _, num_processed_points = processed_data.shape

    # create time axes
    time_raw = np.arange(num_raw_points) / raw_fs
    if processed_data is not None:
        time_processed = np.arange(num_processed_points) / processed_fs

    # Figure 1: Time-Series Butterfly Plot
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize = (14, 10), sharex = False)
    fig1.suptitle(f'EEG Time-Series Comparison: {plot_title}', fontsize = 16)

    ax1.plot(time_raw, raw_data.T, color = 'darkgray', lw = 0.5, alpha = 0.8)
    ax1.set_title(f'Raw Data ({num_channels} Channels)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (µV)')
    ax1.grid(True, linestyle = '--', alpha = 0.6)
    ax1.set_xlim(0, time_raw[-1])

    if processed_data is not None:
        ax2.plot(time_processed, processed_data.T, color = 'dodgerblue', lw = 0.5, alpha = 0.8)
        ax2.set_title(f'Processed Data (Filtered {processed_low_cut}-{processed_high_cut} Hz, Notched at {NOTCH_FREQ_HZ} Hz, Downsampled to {processed_fs} Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Voltage (µV)')
        ax2.grid(True, linestyle = '--', alpha = 0.6)
        ax2.set_xlim(0, time_processed[-1])
    else:
        # show a message if processed data is not provided
        ax2.text(0.5, 0.5, "N/A",
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 fontsize = 14,
                 color = 'red')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Figure 2: Power Spectral Density (PSD) Plot
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize = (14, 6), sharey = True)
    fig2.suptitle(f'Power Spectral Density (PSD) Comparison: {plot_title}', fontsize = 16)

    freqs_raw, psd_raw = welch(raw_data, fs = raw_fs, nperseg = int(raw_fs), axis = -1)
    ax3.semilogy(freqs_raw, np.mean(psd_raw, axis = 0), color = 'darkgray')
    ax3.set_title('PSD of Raw Data')
    ax3.axvline(NOTCH_FREQ_HZ, color = 'red', linestyle = ':', alpha = 0.7, label = f'{NOTCH_FREQ_HZ} Hz Line Noise')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power/Frequency (dB/Hz)')
    ax3.set_xlim(0, 80)
    ax3.grid(True, which = "both", ls = "--", alpha = 0.6)
    ax3.legend()

    if processed_data is not None:
        freqs_proc, psd_proc = welch(processed_data, fs = processed_fs, nperseg = int(processed_fs), axis = -1)
        ax4.semilogy(freqs_proc, np.mean(psd_proc, axis = 0), color = 'dodgerblue')
        ax4.axvspan(processed_low_cut, processed_high_cut, color = 'green', alpha = 0.1, label = f'Pass-band ({processed_low_cut}-{processed_high_cut} Hz)')
        ax4.set_title('PSD of Processed Data')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_xlim(0, 80)
        ax4.grid(True, which = "both", ls = "--", alpha = 0.6)
        ax4.legend()
    else:
        # show a message if processed data is not provided
        ax4.text(0.5, 0.5, "N/A",
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 fontsize = 14,
                 color = 'red')
    
    plt.tight_layout(rect = [0, 0.03, 1, 0.95])

    # create the save directory if it doesn't exist
    os.makedirs(output_dir, exist_ok = True)
    # save the figures
    fig1.savefig(os.path.join(output_dir, "time_series.png"), dpi = 300)
    fig2.savefig(os.path.join(output_dir, "psd.png"), dpi = 300)
    plt.close(fig1)
    plt.close(fig2)


# Dummy entry
if __name__ == "__main__":
    raise NotImplementedError("\033[1;31m[ERROR]\033[0m This file is not meant to be run directly.")
