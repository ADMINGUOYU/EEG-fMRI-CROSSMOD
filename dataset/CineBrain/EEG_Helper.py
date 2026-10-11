# file: EEG_Helper.py
# helps anything with our EEG data (each clip 800 points - 0.8s/1,000 Hz)
# NOTE: DO NOT RUN THIS DIRECTLY

# import general libraries
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, welch

# EEG info
EEG_seq_len = 800  # each clip is 0.8s long, sampled at 1,000 Hz
EEG_sampling_rate = 1000  # Hz
NOTCH_FREQ_HZ = 50.0 # Hz, to remove electrical line noise
# Default preprocessing parameters from the CineBrain paper
LOW_CUT_HZ = 0.1
HIGH_CUT_HZ = 30.0
DOWNSAMPLE_RATE_HZ = 200 # A standard rate for processed EEG
FILTER_ORDER = 4

# Preprocessing function for EEG data
def preprocess_eeg(
        eeg_data: np.ndarray,
        original_fs: int = EEG_sampling_rate,
        new_fs: int = DOWNSAMPLE_RATE_HZ,
        low_cut: float = LOW_CUT_HZ,
        high_cut: float = HIGH_CUT_HZ
    ) -> np.ndarray:
    
    # 1. Band-pass filtering
    nyquist_freq = 0.5 * original_fs
    low = low_cut / nyquist_freq
    high = high_cut / nyquist_freq
    b_band, a_band = butter(FILTER_ORDER, [low, high], btype = 'band')
    filtered_data = filtfilt(b_band, a_band, eeg_data, axis = -1)

    # 2. Notch filtering to remove electrical line noise
    # A quality factor of 30 is standard for a narrow notch
    quality_factor = 30.0
    b_notch, a_notch = iirnotch(NOTCH_FREQ_HZ, quality_factor, original_fs)
    notched_data = filtfilt(b_notch, a_notch, filtered_data, axis = -1)

    # 3. Downsampling
    # ship if the new sampling rate is higher or equal to the original
    if new_fs >= original_fs:
        print(f"\033[1;31m[WARN]\033[0m New sampling rate ({new_fs} Hz) is not lower than original ({original_fs} Hz). Skipping downsampling.")
        return notched_data
    else:
        downsample_factor = int(original_fs / new_fs)
        downsampled_data = notched_data[:, ::downsample_factor]
        return downsampled_data

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
