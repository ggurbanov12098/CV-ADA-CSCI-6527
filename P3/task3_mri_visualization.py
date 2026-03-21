"""
=============================================================================
Task 3 - Visualise MRI DICOM Data
=============================================================================
Loads a DICOM file, prints its metadata, and renders the image slices
or frames with adjustable windowing.

Source:  https://physionet.org/content/images/1.0.0/E1154S7I.dcm

Usage:
    python task3_mri_visualization.py
"""

import os
import sys

try:
    import pydicom
except ImportError:
    sys.exit("pydicom is required. Install with:  pip install pydicom")

import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

DICOM_PATH = "E1154S7I.dcm"
OUTPUT_DIR = os.path.join("output", "task3")


# ──────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────

def print_metadata(ds: pydicom.Dataset) -> str:
    """
    Extract and display key DICOM metadata fields.
    Returns the formatted string for saving to a file.
    """
    fields = [
        ("Patient Name",       "PatientName"),
        ("Patient ID",         "PatientID"),
        ("Patient Age",        "PatientAge"),
        ("Patient Sex",        "PatientSex"),
        ("Modality",           "Modality"),
        ("Study Description",  "StudyDescription"),
        ("Series Description", "SeriesDescription"),
        ("Institution",        "InstitutionName"),
        ("Manufacturer",       "Manufacturer"),
        ("Study Date",         "StudyDate"),
        ("Image Size",         None),
        ("Pixel Spacing",      "PixelSpacing"),
        ("Bits Allocated",     "BitsAllocated"),
        ("Bits Stored",        "BitsStored"),
        ("Number of Frames",   "NumberOfFrames"),
        ("Rows",               "Rows"),
        ("Columns",            "Columns"),
    ]

    lines = []
    lines.append("=" * 60)
    lines.append("DICOM Metadata")
    lines.append("=" * 60)

    for label, attr in fields:
        if attr is None:
            # Special case: computed field
            if label == "Image Size":
                rows = getattr(ds, "Rows", "?")
                cols = getattr(ds, "Columns", "?")
                frames = getattr(ds, "NumberOfFrames", 1)
                value = f"{rows} x {cols} x {frames} frames"
            else:
                value = "N/A"
        else:
            value = getattr(ds, attr, "N/A")
        lines.append(f"  {label:.<30s} {value}")

    text = "\n".join(lines)
    print(text)
    return text


# ──────────────────────────────────────────────────────────────────────
# Windowing helper
# ──────────────────────────────────────────────────────────────────────

def apply_window(pixel_array: np.ndarray, centre: float, width: float) -> np.ndarray:
    """
    Apply a DICOM-style window/level transformation.
        visible range = [centre - width/2, centre + width/2]
    Values outside the range are clipped to 0 or 255.
    """
    lower = centre - width / 2
    upper = centre + width / 2
    windowed = np.clip(pixel_array, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    return windowed


# ──────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────

def visualise_single_frame(pixel_array: np.ndarray, output_dir: str) -> None:
    """Display a single-frame DICOM image with different window presets."""
    # Auto window: use full data range
    p_low, p_high = np.percentile(pixel_array, [1, 99])
    auto_centre = (p_low + p_high) / 2
    auto_width = p_high - p_low

    m_max = pixel_array.max()
    windows = {
        "Auto (1-99 percentile)":  (auto_centre, auto_width),
        "Soft Tissue (Relative)":  (m_max * 0.40, m_max * 0.40),
        "Bone (Relative)":         (m_max * 0.20, m_max * 0.35),
        "Full Range":              (m_max / 2, m_max),
    }

    fig, axes = plt.subplots(1, len(windows), figsize=(5 * len(windows), 5))
    for ax, (label, (c, w)) in zip(axes, windows.items()):
        img = apply_window(pixel_array.astype(np.float64), c, w)
        ax.imshow(img, cmap="gray")
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle("MRI Image - Different Window Presets", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "mri_windows.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Window presets figure → {out_path}")


def visualise_multi_frame(pixel_array: np.ndarray, output_dir: str) -> None:
    """Display a grid of slices from a multi-frame DICOM."""
    n_frames = pixel_array.shape[0]
    max_display = min(n_frames, 25)  # cap for readability

    # Evenly sample slices
    indices = np.linspace(0, n_frames - 1, max_display, dtype=int)

    n_cols = 5
    n_rows = int(np.ceil(max_display / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        frame = pixel_array[idx]
        p_low, p_high = np.percentile(frame, [1, 99])
        img = apply_window(frame.astype(np.float64), (p_low + p_high) / 2, p_high - p_low)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Slice {idx}", fontsize=8)
        axes[i].axis("off")

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"MRI Slices ({n_frames} total, showing {max_display})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "mri_slices_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Slice grid figure → {out_path}")

    # Save a few individual slices
    key_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    for idx in key_indices:
        frame = pixel_array[idx]
        p_low, p_high = np.percentile(frame, [1, 99])
        img = apply_window(frame.astype(np.float64), (p_low + p_high) / 2, p_high - p_low)

        fig_s, ax_s = plt.subplots(figsize=(5, 5))
        ax_s.imshow(img, cmap="gray")
        ax_s.set_title(f"MRI Slice {idx}/{n_frames - 1}")
        ax_s.axis("off")
        plt.tight_layout()

        path = os.path.join(output_dir, f"mri_slice_{idx:04d}.png")
        fig_s.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  ✓ Individual slice → {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Task 3  Visualise MRI DICOM Data")
    print("-" * 40)

    # Load DICOM
    print(f"\nLoading DICOM file: {DICOM_PATH}")
    ds = pydicom.dcmread(DICOM_PATH)

    # Print and save metadata
    meta_text = print_metadata(ds)
    meta_path = os.path.join(OUTPUT_DIR, "metadata.txt")
    with open(meta_path, "w") as f:
        f.write(meta_text)
    print(f"\n  ✓ Metadata saved → {meta_path}")

    # Get pixel data
    pixel_array = ds.pixel_array
    print(f"\n  Pixel array shape: {pixel_array.shape}")
    print(f"  Pixel dtype:       {pixel_array.dtype}")
    print(f"  Min / Max values:  {pixel_array.min()} / {pixel_array.max()}")

    # Visualise
    if pixel_array.ndim == 3:
        print(f"\n  Multi-frame image with {pixel_array.shape[0]} frames/slices.")
        visualise_multi_frame(pixel_array, OUTPUT_DIR)
        # Also show windowed version of a middle slice
        mid = pixel_array.shape[0] // 2
        visualise_single_frame(pixel_array[mid], OUTPUT_DIR)
    else:
        print("\n  Single-frame image.")
        visualise_single_frame(pixel_array, OUTPUT_DIR)

    print(f"\nDone! All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
