"""
=============================================================================
Dynamic Interactive Visualisation Tool
=============================================================================
A single GUI application (matplotlib + widgets) that lets you interactively
explore all three datasets:

  Tab 1 - MRI Slices   : slider to scroll through 76 DICOM frames + window
  Tab 2 - Chemical      : toggle between original and each denoised version
  Tab 3 - Speckle       : toggle between original and each denoised version

Usage:
    python interactive_viewer.py
"""

import os
import sys
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

# ── Optional DICOM import ────────────────────────────────────────────
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


# =====================================================================
# Configuration
# =====================================================================

DICOM_PATH = "E1154S7I.dcm"
CHEMICAL_DIR = os.path.join("noisy", "chemical")
SPECKLE_DIR = os.path.join("noisy", "speckle")

CHEMICAL_FILES = sorted(glob.glob(os.path.join(CHEMICAL_DIR, "*.png")))
SPECKLE_FILES = sorted(
    glob.glob(os.path.join(SPECKLE_DIR, "*.png"))
    + glob.glob(os.path.join(SPECKLE_DIR, "*.jpeg"))
    + glob.glob(os.path.join(SPECKLE_DIR, "*.jpg"))
)

# Reuse filter helpers from the task scripts
MEDIAN_KSIZE = 3
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
MORPH_SE = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
CRIMMINS_ITERS = 3


# =====================================================================
# Image‑processing helpers (same logic as task scripts)
# =====================================================================

def median_filter(img):
    return cv2.medianBlur(img, MEDIAN_KSIZE)

def bilateral_filter(img):
    return cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

def morph_opening(img):
    """Opening DIRECT on dark-on-white: does NOT remove dark noise (educational only)."""
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, MORPH_SE)

def morph_closing(img):
    """Closing on INVERTED image (2x2 SE): bridges gaps in 1px lines."""
    inv = cv2.bitwise_not(img)
    small_se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, small_se)
    return cv2.bitwise_not(closed)

def combined_pipeline(img):
    """Otsu -> Invert -> Dilate -> Area Filter -> Erode -> Invert."""
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(thresh)

    small_se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(inv, small_se, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    MIN_AREA = 10
    filtered = np.zeros_like(dilated)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            filtered[labels == i] = 255

    eroded = cv2.erode(filtered, small_se, iterations=1)
    return cv2.bitwise_not(eroded)


def _crimmins_pass(img, dr, dc):
    h, w = img.shape
    out = img.astype(np.float64).copy()
    r0, r1 = max(0, -dr), h - max(0, dr)
    c0, c1 = max(0, -dc), w - max(0, dc)
    centre = out[r0:r1, c0:c1]
    neighbour = out[r0+dr:r1+dr, c0+dc:c1+dc]
    diff = neighbour - centre
    centre[diff > 0] += 1
    centre[diff < 0] -= 1
    return np.clip(out, 0, 255).astype(np.uint8)


def crimmins_filter(img, iters=CRIMMINS_ITERS):
    dirs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    result = img.copy()
    for _ in range(iters):
        for d in dirs:
            result = _crimmins_pass(result, *d)
    return result


def fft_lowpass(img, cutoff_ratio=0.08):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    radius = int(cutoff_ratio * np.sqrt(rows**2 + cols**2))
    f = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    y, x = np.ogrid[:rows, :cols]
    mask = (np.sqrt((y - crow)**2 + (x - ccol)**2) <= radius).astype(np.float64)
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(f * mask)))
    return np.clip(result, 0, 255).astype(np.uint8)


def unsharp_mask(img, sigma=2.0, strength=0.7):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)


def apply_window(arr, centre, width):
    lo, hi = centre - width / 2, centre + width / 2
    w = np.clip(arr, lo, hi)
    return ((w - lo) / (hi - lo) * 255).astype(np.uint8)


# =====================================================================
# 1. MRI Slice Viewer
# =====================================================================

def launch_mri_viewer():
    if not HAS_PYDICOM:
        print("pydicom not installed - skipping MRI viewer.")
        return
    if not os.path.isfile(DICOM_PATH):
        print(f"DICOM file not found: {DICOM_PATH}")
        return

    ds = pydicom.dcmread(DICOM_PATH)
    pixel_array = ds.pixel_array.astype(np.float64)

    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, ...]

    n_frames = pixel_array.shape[0]

    # Precompute auto-window per slice
    p_lo = np.percentile(pixel_array, 1)
    p_hi = np.percentile(pixel_array, 99)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title("MRI Slice Viewer")
    plt.subplots_adjust(bottom=0.22, right=0.78)

    img_display = ax.imshow(
        apply_window(pixel_array[0], (p_lo + p_hi) / 2, p_hi - p_lo),
        cmap="gray",
    )
    ax.set_title("MRI - Slice 0 / {}".format(n_frames - 1))
    ax.axis("off")

    # Slice slider
    ax_slice = plt.axes([0.18, 0.08, 0.55, 0.03])
    slider_slice = Slider(ax_slice, "Slice", 0, n_frames - 1, valinit=0, valstep=1)

    # Window centre slider
    ax_wc = plt.axes([0.18, 0.04, 0.55, 0.03])
    slider_wc = Slider(ax_wc, "W-Centre", 0, pixel_array.max(), valinit=(p_lo + p_hi) / 2)

    # Window width slider
    ax_ww = plt.axes([0.18, 0.00, 0.55, 0.03])
    slider_ww = Slider(ax_ww, "W-Width", 1, pixel_array.max(), valinit=p_hi - p_lo)

    # Preset window radio buttons
    ax_radio = plt.axes([0.80, 0.30, 0.18, 0.30])
    radio = RadioButtons(ax_radio, ("Auto", "Soft Tissue", "Bone", "Full Range"))

    m_max = pixel_array.max()
    presets = {
        "Auto":        ((p_lo + p_hi) / 2, p_hi - p_lo),
        "Soft Tissue": (m_max * 0.40, m_max * 0.40),
        "Bone":        (m_max * 0.20, m_max * 0.35),
        "Full Range":  (m_max / 2, m_max),
    }

    def update(_=None):
        idx = int(slider_slice.val)
        frame = pixel_array[idx]
        img = apply_window(frame, slider_wc.val, slider_ww.val)
        img_display.set_data(img)
        ax.set_title(f"MRI - Slice {idx} / {n_frames - 1}")
        fig.canvas.draw_idle()

    def on_preset(label):
        c, w = presets[label]
        slider_wc.set_val(c)
        slider_ww.set_val(w)

    slider_slice.on_changed(update)
    slider_wc.on_changed(update)
    slider_ww.on_changed(update)
    radio.on_clicked(on_preset)

    plt.show()


# =====================================================================
# 2. Chemical Noise - Before / After Viewer
# =====================================================================

def launch_chemical_viewer():
    if not CHEMICAL_FILES:
        print("No chemical images found.")
        return

    # Precompute all filter results for each image
    images = {}
    for path in CHEMICAL_FILES:
        name = os.path.basename(path)
        orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue
        images[name] = {
            "Original":          orig,
            "Median":            median_filter(orig),
            "Bilateral":         bilateral_filter(orig),
            "Morph. Opening":    morph_opening(orig),
            "Morph. Closing":    morph_closing(orig),
            "Combined Pipeline": combined_pipeline(orig),
        }

    names = list(images.keys())
    filters = list(list(images.values())[0].keys())

    fig, (ax_orig, ax_filt) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title("Chemical Noise - Dynamic Viewer")
    plt.subplots_adjust(bottom=0.10, right=0.72)

    # Initial display
    cur_name = names[0]
    cur_filter = filters[1]  # start on Median

    def draw_images():
        """Clear axes and redraw -- handles different image dimensions."""
        orig_img = images[cur_name]["Original"]
        filt_img = images[cur_name][cur_filter]

        ax_orig.clear()
        ax_orig.imshow(orig_img, cmap="gray", aspect="equal")
        ax_orig.set_title("Original (Noisy)")
        ax_orig.axis("off")

        ax_filt.clear()
        ax_filt.imshow(filt_img, cmap="gray", aspect="equal")
        ax_filt.set_title(cur_filter)
        ax_filt.axis("off")

        fig.suptitle(f"Chemical Image: {cur_name}", fontsize=13, fontweight="bold")
        fig.canvas.draw_idle()

    draw_images()

    # Image selector
    ax_img_radio = plt.axes([0.74, 0.55, 0.24, 0.35])
    ax_img_radio.set_title("Image", fontsize=10, fontweight="bold")
    radio_img = RadioButtons(ax_img_radio, names)

    # Filter selector
    ax_flt_radio = plt.axes([0.74, 0.10, 0.24, 0.40])
    ax_flt_radio.set_title("Filter", fontsize=10, fontweight="bold")
    radio_flt = RadioButtons(ax_flt_radio, filters[1:])  # exclude "Original"

    def on_img(label):
        nonlocal cur_name
        cur_name = label
        draw_images()

    def on_flt(label):
        nonlocal cur_filter
        cur_filter = label
        draw_images()

    radio_img.on_clicked(on_img)
    radio_flt.on_clicked(on_flt)

    plt.show()


# =====================================================================
# 3. Speckle Noise - Before / After Viewer
# =====================================================================

def launch_speckle_viewer():
    if not SPECKLE_FILES:
        print("No speckle images found.")
        return

    print("  Pre-processing speckle images (Crimmins may take a moment)…")
    images = {}
    for path in SPECKLE_FILES:
        name = os.path.basename(path)
        orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue
        cr = crimmins_filter(orig)
        images[name] = {
            "Original":           orig,
            "Crimmins":           cr,
            "FFT Lowpass":        fft_lowpass(orig),
            "Crimmins + Unsharp": unsharp_mask(cr),
        }
        print(f"    ✓ {name}")

    names = list(images.keys())
    filters = list(list(images.values())[0].keys())

    fig, (ax_orig, ax_filt) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title("Speckle Noise - Dynamic Viewer")
    plt.subplots_adjust(bottom=0.10, right=0.72)

    cur_name = names[0]
    cur_filter = filters[1]

    def draw_images():
        """Clear axes and redraw -- handles different image dimensions."""
        orig_img = images[cur_name]["Original"]
        filt_img = images[cur_name][cur_filter]

        ax_orig.clear()
        ax_orig.imshow(orig_img, cmap="gray", aspect="equal")
        ax_orig.set_title("Original (Speckled)")
        ax_orig.axis("off")

        ax_filt.clear()
        ax_filt.imshow(filt_img, cmap="gray", aspect="equal")
        ax_filt.set_title(cur_filter)
        ax_filt.axis("off")

        fig.suptitle(f"Speckle Image: {cur_name}", fontsize=13, fontweight="bold")
        fig.canvas.draw_idle()

    draw_images()

    # Image selector
    ax_img_radio = plt.axes([0.74, 0.55, 0.24, 0.35])
    ax_img_radio.set_title("Image", fontsize=10, fontweight="bold")
    radio_img = RadioButtons(ax_img_radio, names)

    # Filter selector
    ax_flt_radio = plt.axes([0.74, 0.15, 0.24, 0.35])
    ax_flt_radio.set_title("Method", fontsize=10, fontweight="bold")
    radio_flt = RadioButtons(ax_flt_radio, filters[1:])

    def on_img(label):
        nonlocal cur_name
        cur_name = label
        draw_images()

    def on_flt(label):
        nonlocal cur_filter
        cur_filter = label
        draw_images()

    radio_img.on_clicked(on_img)
    radio_flt.on_clicked(on_flt)

    plt.show()


# =====================================================================
# Main menu
# =====================================================================

def main():
    print("=" * 55)
    print("  Dynamic Interactive Visualisation Tool")
    print("=" * 55)
    print()
    print("  1 - MRI Slice Viewer      (scroll through DICOM frames)")
    print("  2 - Chemical Noise Viewer  (compare denoising filters)")
    print("  3 - Speckle Noise Viewer   (compare removal methods)")
    print("  a - Launch ALL viewers sequentially")
    print("  q - Quit")
    print()

    while True:
        choice = input("Select viewer [1/2/3/a/q]: ").strip().lower()

        if choice == "1":
            launch_mri_viewer()
        elif choice == "2":
            launch_chemical_viewer()
        elif choice == "3":
            launch_speckle_viewer()
        elif choice == "a":
            launch_mri_viewer()
            launch_chemical_viewer()
            launch_speckle_viewer()
        elif choice == "q":
            print("Goodbye!")
            break
        else:
            print("  Invalid choice. Try 1, 2, 3, a, or q.")


if __name__ == "__main__":
    main()
