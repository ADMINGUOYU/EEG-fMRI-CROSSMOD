# file: fMRI_Helper.py
# helps anything with our (18946,) dimension fMRI data
# NOTE: DO NOT RUN THIS DIRECTLY

# import general libraries
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

# fMRI voxels' info
# define shape
fMRI_shape = (18946,)
# Define the 29 visual and 31 auditory HCP-MMP parcel names used in CineBrain
VISUAL_ROIS = {
    'V1', 'V2', 'V3', 'V3A', 'V3B', 'V3CD', 'V4', 'LO1', 'LO2', 'LO3',
    'PIT', 'V4t', 'V6', 'V6A', 'V7', 'V8', 'PH', 'FFC', 'IP0', 'MT',
    'MST', 'FST', 'VVC', 'VMV1', 'VMV2', 'VMV3', 'PHA1', 'PHA2', 'PHA3'
}
AUDITORY_ROIS = {
    '4', '7AL', '7Am', '7m', '7PC', '7PL', '7Pm', '8Ad', '8Av', '8BM',
    '8C', '9a', '9p', '10d', '10v', '44', '45', '46', '47m', 'A1',
    'IPS1', 'p32', 'PGp', 's32', 'STGa', 'STSda', 'STSdp', 'STSva',
    'STSvp', 'TPOJ1', 'TPOJ2', 'TPOJ3'
}


# fMRI visualiser, plot as signal and shows statistics as variables
def visualise_fmri_signal_static(
        fmri_data: np.ndarray,
        plot_title: str,
        save_path: str,
    ) -> None:

    # check shape
    if fmri_data.shape != fMRI_shape:
        print(f"\033[1;31m[ERROR]\033[0m Invalid shape for fMRI data: expected {fMRI_shape}, got {fmri_data.shape}")
        return

    # Split data into visual and auditory cortex based on known partitioning
    visual_roi = fmri_data[:8405]
    auditory_roi = fmri_data[8405:]

    # Create subplots for visualisation
    fig, axes = plt.subplots(1, 3, figsize = (20, 5))
    fig.suptitle(f"fMRI Signal Stat.: {plot_title}", fontsize = 16)

    # 1. Full 1D signal profile
    axes[0].plot(fmri_data, color = 'black', lw = 0.5, alpha = 0.8)
    axes[0].axvline(8405, color = 'crimson', linestyle = '--', label = 'Visual/Auditory Boundary')
    axes[0].set_title("Concatenated BOLD Signal")
    axes[0].set_xlabel("Vertex Index")
    axes[0].set_ylabel("Signal Amplitude (Z-Score)")
    axes[0].legend()
    axes[0].grid(True, linestyle = '--', alpha = 0.6)

    # 2. Distribution comparison
    axes[1].hist(visual_roi, bins = 50,
                 alpha = 0.7,
                 edgecolor = 'black',
                 color = 'dodgerblue',
                 label = f'Visual (n = {len(visual_roi)})',
                 density = True)
    axes[1].hist(auditory_roi,
                 bins = 50,
                 alpha = 0.7,
                 edgecolor = 'black',
                 color = 'darkorange',
                 label = f'Auditory (n = {len(auditory_roi)})',
                 density = True)
    axes[1].set_title("Signal Amplitude Distribution")
    axes[1].set_xlabel("Signal Amplitude (Z-Score)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # 3. Mean Absolute Activation
    categories = ['Visual Cortex', 'Auditory Cortex']
    mean_activations = [np.mean(np.abs(visual_roi)), np.mean(np.abs(auditory_roi))]
    std_devs = [np.std(visual_roi), np.std(auditory_roi)]
    axes[2].bar(categories, mean_activations, yerr = std_devs, capsize = 5, color = ['dodgerblue', 'darkorange'])
    axes[2].set_title("Mean Absolute Activation (|Z|)")
    axes[2].set_ylabel("Mean |Z| ± Std. Dev.")

    plt.tight_layout(rect = [0, 0.03, 1, 0.95])

    # make directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    # save & close
    plt.savefig(save_path, dpi = 300)
    plt.close(fig)

# fMRI visualiser, plot voxels in mesh surface
# NOTE: requires hcp-utils and nilearn
def visualize_fmri_mesh(
        fMRI: np.ndarray,
        output_dir: str,
    ) -> None:

    # first we try to import the libraries
    try:
        import hcp_utils as hcp
        from nilearn import plotting
        # try plotly too, nilearn uses it for 3D plotting
        import plotly
    except ImportError as e:
        print(f"\033[1;31m[ERROR]\033[0m Required libraries not found: {e}.")
        print("        Please install hcp-utils, nilearn, and plotly to use this function.")
        print("        run `pip install hcp-utils nilearn plotly`")
        return

    # assert shape
    assert fMRI.shape == fMRI_shape, f"Expected shape {fMRI_shape}, got {fMRI.shape}"

    # set paths
    os.makedirs(output_dir, exist_ok = True)
    output_html: str = os.path.join(output_dir, "brain_mesh_3d.html")
    output_png: str = os.path.join(output_dir, "brain_mesh_views.png")

    # Step A: Reconstruct the full 91,282 grayordinates array
    full_grayordinates = np.zeros(91282, dtype = np.float32)

    # define a helper to map the indices
    def get_cinebrain_indices():
        # Load HCP-MMP parcellation labels and mapping
        labels = hcp.mmp.labels # {parcel_id: 'L_V1_ROI', ...}
        map_all = hcp.mmp.map_all # shape: (91282,)
        # Identify parcel IDs matching visual and auditory names
        visual_ids = {
            pid for pid, name in labels.items() 
            if any(name.startswith(f"L_{roi}_") or name.startswith(f"R_{roi}_") or 
                   name == f"L_{roi}" or name == f"R_{roi}" for roi in VISUAL_ROIS)
        }
        auditory_ids = {
            pid for pid, name in labels.items() 
            if any(name.startswith(f"L_{roi}_") or name.startswith(f"R_{roi}_") or 
                   name == f"L_{roi}" or name == f"R_{roi}" for roi in AUDITORY_ROIS)
        }
        # Extract vertex indices in 91,282 grayordinate space
        visual_idx = np.where(np.isin(map_all, list(visual_ids)))[0]
        auditory_idx = np.where(np.isin(map_all, list(auditory_ids)))[0]
        # Concatenate visual (8405) + auditory (10541) -> total 18946
        combined_idx = np.concatenate([visual_idx, auditory_idx])
        return combined_idx

    indices = get_cinebrain_indices()
    full_grayordinates[indices[:len(fMRI)]] = fMRI

    # Step B: Map 91k grayordinates to 64k surface vertices for plotting
    surface_data = hcp.cortex_data(full_grayordinates)

    # 1. Generate Interactive 3D WebGL Surface (open in any web browser)
    view = plotting.view_surf(
        surf_mesh = hcp.mesh.inflated,
        surf_map = surface_data,
        bg_map = hcp.mesh.sulc,
        threshold = 0.1,
        cmap = 'cold_hot',
        title = "fMRI Activation Mesh",
        title_fontsize = 18,
    )
    view.save_as_html(output_html)

    # 2. Generate Static 4-View Figure (Left/Right, Lateral/Medial)
    fig, axes = plt.subplots(2, 2,
                             subplot_kw = {'projection': '3d'},
                             figsize = (16, 12))
    
    lh_data = hcp.left_cortex_data(full_grayordinates)
    rh_data = hcp.right_cortex_data(full_grayordinates)
    
    # Left Lateral & Medial
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated_left, lh_data, hemi = 'left', view = 'lateral',
        bg_map = hcp.mesh.sulc_left, cmap = 'cold_hot', axes = axes[0, 0], title = "Left Lateral"
    )
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated_left, lh_data, hemi = 'left', view = 'medial',
        bg_map = hcp.mesh.sulc_left, cmap = 'cold_hot', axes = axes[0, 1], title = "Left Medial"
    )
    # Right Lateral & Medial
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated_right, rh_data, hemi = 'right', view = 'lateral',
        bg_map = hcp.mesh.sulc_right, cmap = 'cold_hot', axes = axes[1, 0], title = "Right Lateral"
    )
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated_right, rh_data, hemi = 'right', view = 'medial',
        bg_map = hcp.mesh.sulc_right, cmap = 'cold_hot', axes = axes[1, 1], title = "Right Medial"
    )

    # save the static figure
    # plt.tight_layout(): same axes don't support this (the colour bar might shift)
    plt.savefig(output_png, dpi = 300)
    plt.close(fig)

    # unload the libraries to free memory
    del hcp, plotting, plotly


# Dummy entry
if __name__ == "__main__":
    raise NotImplementedError("\033[1;31m[ERROR]\033[0m This file is not meant to be run directly.")
