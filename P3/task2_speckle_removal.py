"""
=============================================================================
Task 2 - Speckle Noise Removal
=============================================================================
Removes speckle noise from grayscale images using non-ML techniques:

  1. Crimmins Speckle Removal  - complementary hulling (8-neighbour compare)
  2. FFT Lowpass Filter        - frequency-domain suppression
  3. Unsharp Masking           - edge sharpening after denoising

Usage:
    python task2_speckle_removal.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

INPUT_DIR = os.path.join("noisy", "speckle")
OUTPUT_DIR = os.path.join("output", "task2")

IMAGE_FILES = ["1.jpeg", "3.png", "4.png"]

# Crimmins iterations
CRIMMINS_ITERATIONS = 3

# FFT lowpass radius (fraction of image diagonal)
FFT_CUTOFF_RATIO = 0.08

# Unsharp masking parameters
UNSHARP_SIGMA = 2.0
UNSHARP_STRENGTH = 0.7


# ──────────────────────────────────────────────────────────────────────
# Crimmins Speckle Removal (Complementary Hulling)
# ──────────────────────────────────────────────────────────────────────

def _crimmins_pass(img: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """
    One pass of Crimmins complementary hulling along direction (dr, dc).

    For every pixel, compare it to its neighbour in the given direction:
      • If the pixel is darker  ➜ increment it (bring closer to neighbour)
      • If the pixel is brighter ➜ decrement it
    This progressively smooths speckle while broadly preserving edges.
    """
    h, w = img.shape
    out = img.astype(np.float64).copy()

    r_start = max(0, -dr)
    r_end   = h - max(0, dr)
    c_start = max(0, -dc)
    c_end   = w - max(0, dc)

    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            nr, nc = r + dr, c + dc
            diff = out[nr, nc] - out[r, c]
            if diff > 0:
                out[r, c] += 1      # pixel is darker  → lift
            elif diff < 0:
                out[r, c] -= 1      # pixel is brighter → lower

    return np.clip(out, 0, 255).astype(np.uint8)


def crimmins_speckle_removal(
    img: np.ndarray, iterations: int = CRIMMINS_ITERATIONS
) -> np.ndarray:
    """
    Full Crimmins speckle removal using complementary hulling.

    Each iteration processes the image along all 8 compass directions
    (N, NE, E, SE, S, SW, W, NW), incrementing darker pixels and
    decrementing brighter ones relative to their neighbours.
    """
    # The 8 compass direction offsets (row_delta, col_delta)
    directions = [
        (-1,  0),  # N
        (-1,  1),  # NE
        ( 0,  1),  # E
        ( 1,  1),  # SE
        ( 1,  0),  # S
        ( 1, -1),  # SW
        ( 0, -1),  # W
        (-1, -1),  # NW
    ]

    result = img.copy()
    for i in range(iterations):
        for dr, dc in directions:
            result = _crimmins_pass(result, dr, dc)
        print(f"    Crimmins iteration {i + 1}/{iterations} complete")

    return result


# ──────────────────────────────────────────────────────────────────────
# Optimised Crimmins via vectorised NumPy (much faster)
# ──────────────────────────────────────────────────────────────────────

def _crimmins_pass_fast(img: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Vectorised single-direction Crimmins pass using NumPy slicing."""
    h, w = img.shape
    out = img.astype(np.float64).copy()

    r_start = max(0, -dr)
    r_end   = h - max(0, dr)
    c_start = max(0, -dc)
    c_end   = w - max(0, dc)

    centre = out[r_start:r_end, c_start:c_end]
    neighbour = out[r_start + dr:r_end + dr, c_start + dc:c_end + dc]

    diff = neighbour - centre
    centre[diff > 0] += 1
    centre[diff < 0] -= 1

    return np.clip(out, 0, 255).astype(np.uint8)


def crimmins_speckle_removal_fast(
    img: np.ndarray, iterations: int = CRIMMINS_ITERATIONS
) -> np.ndarray:
    """Vectorised Crimmins speckle removal (recommended for real usage)."""
    directions = [
        (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
        ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
    ]
    result = img.copy()
    for i in range(iterations):
        for dr, dc in directions:
            result = _crimmins_pass_fast(result, dr, dc)
        print(f"    Crimmins (fast) iteration {i + 1}/{iterations} complete")
    return result


# ──────────────────────────────────────────────────────────────────────
# FFT Lowpass Filter
# ──────────────────────────────────────────────────────────────────────

def fft_lowpass_filter(
    img: np.ndarray, cutoff_ratio: float = FFT_CUTOFF_RATIO
) -> np.ndarray:
    """
    Apply a circular lowpass filter in the frequency domain.

    Steps:
        1. Compute the 2-D FFT and shift the zero-frequency to centre.
        2. Create a circular mask that preserves low frequencies.
        3. Multiply the spectrum by the mask.
        4. Inverse FFT back to the spatial domain.
    """
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    radius = int(cutoff_ratio * np.sqrt(rows**2 + cols**2))

    # Forward FFT
    f_transform = np.fft.fft2(img.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)

    # Circular lowpass mask
    mask = np.zeros((rows, cols), dtype=np.float64)
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    mask[dist <= radius] = 1.0

    # Apply mask and inverse FFT
    f_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_filtered)
    result = np.fft.ifft2(f_ishift)
    result = np.abs(result)

    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
# Unsharp Masking
# ──────────────────────────────────────────────────────────────────────

def unsharp_mask(
    img: np.ndarray,
    sigma: float = UNSHARP_SIGMA,
    strength: float = UNSHARP_STRENGTH,
) -> np.ndarray:
    """
    Sharpen an image by subtracting a blurred copy:
        sharpened = original + strength × (original − blurred)
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# ──────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────

def save_comparison_figure(
    original: np.ndarray,
    results: dict,
    image_name: str,
    output_dir: str,
) -> None:
    """Save a side-by-side comparison of all denoising methods."""
    n_cols = 1 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original (Speckled)")
    axes[0].axis("off")

    for idx, (label, img) in enumerate(results.items(), start=1):
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(label)
        axes[idx].axis("off")

    fig.suptitle(
        f"Speckle Removal Comparison - {image_name}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{image_name}_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved comparison figure → {out_path}")


def save_individual_results(
    results: dict,
    image_name: str,
    output_dir: str,
) -> None:
    """Save each denoised result as a separate image file."""
    for label, img in results.items():
        safe_label = label.lower().replace(" ", "_").replace("→", "to")
        out_path = os.path.join(output_dir, f"{image_name}_{safe_label}.png")
        cv2.imwrite(out_path, img)


# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────

def print_analysis() -> None:
    """Print observations on speckle removal techniques."""
    print("\n" + "=" * 70)
    print("ANALYSIS - Speckle Removal Techniques")
    print("=" * 70)
    print("""
Crimmins Speckle Removal:
  • Directly addresses speckle by iteratively adjusting each pixel
    towards its local neighbourhood average across 8 directions.
  • Effective at reducing granular speckle while preserving larger
    structural edges (roads, bones, etc.).
  • Multiple iterations further smooth the result but may start to
    soften fine details.

FFT Lowpass Filtering:
  • Removes high-frequency components (sharp noise) from the spectrum.
  • Very effective at producing a visually smoother image but can
    introduce ringing artefacts near strong edges.

Unsharp Masking (post-processing):
  • Re-sharpens edges that may have been softened by aggressive
    denoising, restoring some structural clarity.

Recommendation:
  The Crimmins method is purpose-built for speckle and delivers the
  best balance of noise suppression and edge preservation. Combining
  it with a mild unsharp mask further improves perceived sharpness.

Note: No Machine Learning algorithms were used.
""")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full Task-2 pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Task 2 - Speckle Noise Removal")
    print("-" * 40)

    for fname in IMAGE_FILES:
        img_path = os.path.join(INPUT_DIR, fname)
        image_name = os.path.splitext(fname)[0]
        print(f"\nProcessing: {fname}")

        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            print(f"  ✗ Could not read {img_path}, skipping.")
            continue

        # 1. Crimmins (fast vectorised version)
        print("  Applying Crimmins speckle removal …")
        crimmins_result = crimmins_speckle_removal_fast(original, CRIMMINS_ITERATIONS)

        # 2. FFT Lowpass
        print("  Applying FFT lowpass filter …")
        fft_result = fft_lowpass_filter(original)

        # 3. Unsharp mask (applied to Crimmins output for sharpening)
        print("  Applying Unsharp Masking …")
        crimmins_sharpened = unsharp_mask(crimmins_result)

        results = {
            "Crimmins":               crimmins_result,
            "FFT Lowpass":            fft_result,
            "Crimmins + Unsharp":     crimmins_sharpened,
        }

        save_individual_results(results, image_name, OUTPUT_DIR)
        save_comparison_figure(original, results, image_name, OUTPUT_DIR)

    print_analysis()
    print("Done! All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
