"""
=============================================================================
Task 1 - Noise Removal & Reconstruction of Chemical Formula Images
=============================================================================
Applies several denoising filters and morphological operations to clean
noisy chemical-structure images, then saves side-by-side comparison figures.

Filters tested:
  1. Median Filter         - non-linear; excellent for salt-and-pepper noise
                             (BUT destroys 1-pixel thin lines in binary images!)
  2. Bilateral Filter      - edge-preserving Gaussian smoothing
  3. Morphological Opening - shown for comparison; does NOT actually remove
                             dark noise on a dark-on-white image (see note)
  4. Morphological Closing - thickens/reconnects broken bond lines
  5. Combined Pipeline     - Dilation -> Connected Components Filter -> Erosion

NOTE ON THE 1-PIXEL PARADOX:
  Chemical images are purely binary with 1-pixel thick lines and 1-pixel
  thick noise dots. 
  - Standard Morphology fails: Opening directly expands dark specks. Opening
    inverted destroys the 1px lines completely. 
  - Standard Median Filter fails: 3x3 median wipes out 1px lines.
  - The True Solution: Invert -> Dilate (to reconnect lines into massive blobs
    and expand noise to 2x2) -> Filter by connected component area (removes
    small noise blobs) -> Erode back down to 1px -> Invert to original.

Usage:
    python task1_chemical_noise.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

INPUT_DIR = os.path.join("noisy", "chemical")
OUTPUT_DIR = os.path.join("output", "task1")

IMAGE_FILES = ["inchi10.png", "inchi11.png", "inchi12.png"]

# Median filter kernel size (must be odd)
MEDIAN_KSIZE = 3

# Bilateral filter parameters
BILATERAL_D = 9          # diameter of pixel neighbourhood
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Morphological structuring element
MORPH_KSIZE = 3
MORPH_SE = cv2.getStructuringElement(
    cv2.MORPH_RECT, (MORPH_KSIZE, MORPH_KSIZE)
)


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load an image in grayscale and verify it was read."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def apply_median_filter(img: np.ndarray) -> np.ndarray:
    """Apply a median filter - ideal for salt-and-pepper noise."""
    return cv2.medianBlur(img, MEDIAN_KSIZE)


def apply_bilateral_filter(img: np.ndarray) -> np.ndarray:
    """Apply a bilateral filter - smooths while preserving edges."""
    return cv2.bilateralFilter(
        img, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    )


def apply_opening(img: np.ndarray) -> np.ndarray:
    """
    Morphological Opening applied DIRECTLY to the dark-on-white image.

    Opening = erosion (local min) -> dilation (local max).
    On a dark-on-white image, erosion replaces each pixel with its
    local MINIMUM, which EXPANDS dark regions (including noise).
    Dilation then restores them.  Net effect: dark specks are
    perfectly preserved -- Opening does NOT remove them.

    Included here for educational comparison only.
    """
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, MORPH_SE)


def apply_closing(img: np.ndarray) -> np.ndarray:
    """
    Morphological Closing to thicken and reconnect dark bond lines.

    Works in the INVERTED domain (lines = white on black):
      1. Invert image (chemical lines become white foreground)
      2. Closing = dilation -> erosion on the white lines:
         dilation grows the white lines, bridging nearby gaps;
         erosion shrinks them back but bridges stay connected.
      3. Invert back to dark-on-white

    A small 2x2 SE is used because the lines are only ~1 pixel wide;
    a 3x3 erosion step would destroy them entirely.
    """
    inv = cv2.bitwise_not(img)
    small_se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, small_se)
    return cv2.bitwise_not(closed)


def apply_combined_pipeline(img: np.ndarray) -> np.ndarray:
    """
    Combined pipeline: Otsu -> Invert -> Dilate -> Area Filter -> Erode -> Invert.

    This solves the "1-Pixel Paradox" where standard morphology or median filters
    destroy the 1-pixel thick chemical lines. 
    Steps:
      1. Otsu threshold and invert (white structures on black background).
      2. Dilate (2x2): Thickens 1px lines to 2px, bridging gaps. 1x1 noise
         becomes 2x2 (area = 4). Chemical structures become massive blobs.
      3. Connected Components Area Filter: Keep only blobs with area >= 10.
         This perfectly deletes the isolated noise specks.
      4. Erode (2x2): Shrink surviving structures back to 1px thickness.
      5. Invert back to dark-on-white.
    """
    # 1. Binarize and invert
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(thresh)

    # 2. Dilate to reconnect fragments and thicken lines
    small_se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(inv, small_se, iterations=1)

    # 3. Connected Components Filter by Area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    # Create mask of only large components (area > 10)
    MIN_AREA = 10
    filtered = np.zeros_like(dilated)
    for i in range(1, num_labels):  # Skip 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            filtered[labels == i] = 255

    # 4. Erode back to original line thickness
    eroded = cv2.erode(filtered, small_se, iterations=1)

    # 5. Invert back to normal polarity
    return cv2.bitwise_not(eroded)


def save_comparison_figure(
    original: np.ndarray,
    results: dict,
    image_name: str,
    output_dir: str,
) -> None:
    """
    Create and save a comparison figure showing the original image
    alongside every filtered version.
    """
    n_cols = 1 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Original
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original (Noisy)")
    axes[0].axis("off")

    # Filtered results
    for idx, (label, img) in enumerate(results.items(), start=1):
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(label)
        axes[idx].axis("off")

    fig.suptitle(f"Filter Comparison - {image_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{image_name}_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison figure -> {out_path}")


def save_individual_results(
    results: dict,
    image_name: str,
    output_dir: str,
) -> None:
    """Save each filtered image individually."""
    for label, img in results.items():
        safe_label = label.lower().replace(" ", "_").replace(".", "")
        out_path = os.path.join(output_dir, f"{image_name}_{safe_label}.png")
        cv2.imwrite(out_path, img)


# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────

def print_analysis() -> None:
    """Print a brief analytical summary of the filters tried."""
    print("\n" + "=" * 70)
    print("ANALYSIS - Filter Effectiveness for Chemical Structure Images")
    print("=" * 70)
    print("""
Best Filter:
  - Combined Pipeline (Dilation + Connected Components Area Filter): 
    The original images suffer from the "1-Pixel Paradox". Because both 
    the noise specks and the actual chemical lines are exactly 1-pixel thick 
    binary structures, traditional filters fail catastrophically:
      * 3x3 Median Blur completely wipes out the 1px thick lines.
      * Morphological Opening (inverted) destroys the 1px lines.
    
    The true solution is to Dilate the inverted image to bridge broken lines 
    into massive structures, while the noise only dilates to 2x2. A connected 
    component area filter easily deletes the noise (<10px area), and Erosion 
    restores the lines perfectly.

Worst Filter:
  - Median Filter / Standard Morphology: When applied to purely binary images 
    with 1-pixel structures, they cannot distinguish between small noise and 
    thin lines without destroying the lines entirely. Bilateral Filter also 
    does nothing useful here since the images contain no grayscale gradients.

Morphological Operations:
  - Opening (applied directly on dark-on-white) does NOT remove dark
    noise. Erosion takes the local MINIMUM, which expands dark specks;
    dilation restores them. Net result: noise is preserved.
  - Closing (applied on inverted image with a small 2x2 SE) bridges
    gaps in thin chemical-bond lines, but it preserves all noise.
""")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full Task-1 pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Task 1 - Noise Removal & Reconstruction (Chemical Images)")
    print("-" * 58)

    for fname in IMAGE_FILES:
        img_path = os.path.join(INPUT_DIR, fname)
        image_name = os.path.splitext(fname)[0]
        print(f"\nProcessing: {fname}")

        original = load_image(img_path)

        # Apply each filter / operation
        results = {
            "Median Filter":       apply_median_filter(original),
            "Bilateral Filter":    apply_bilateral_filter(original),
            "Morph. Opening":      apply_opening(original),
            "Morph. Closing":      apply_closing(original),
            "Combined Pipeline":   apply_combined_pipeline(original),
        }

        # Save outputs
        save_individual_results(results, image_name, OUTPUT_DIR)
        save_comparison_figure(original, results, image_name, OUTPUT_DIR)

    # Print analysis
    print_analysis()

    print("Done! All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
