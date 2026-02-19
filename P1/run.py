import argparse
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.census import census_transform
from src.disparity import compute_disparity, compute_disparity_both, lr_check, postprocess
from src.utils import load_gray, resize_gray_pair, discover_pairs, ensure_dir


def parse_args():
    ap = argparse.ArgumentParser(description="Stereo Disparity via Census Transform")
    ap.add_argument("--pairs_dir", type=str, default="images/original")
    ap.add_argument("--out_dir", type=str, default="images/outputs")
    ap.add_argument("--window", type=int, default=9, help="Census window size (odd, >=3)")
    ap.add_argument("--max_disp", type=int, default=64, help="Max disparity to search")
    ap.add_argument("--agg", type=int, default=51, help="Cost aggregation window (odd, 0=off)")
    ap.add_argument("--scale", type=float, default=1.0, help="Resize scale")
    ap.add_argument("--lr_thresh", type=int, default=2, help="LR consistency threshold")
    ap.add_argument("--no_viz", action="store_true", help="Skip visualization")
    return ap.parse_args()


def visualize(left, right, disparity, valid_pct, dt, args, save_path=None):
    """3-panel visualization: Left | Right | Disparity with stats."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(left, cmap="gray")
    axes[0].set_title("Left Image")
    axes[0].axis("off")

    axes[1].imshow(right, cmap="gray")
    axes[1].set_title("Right Image")
    axes[1].axis("off")

    im = axes[2].imshow(disparity, cmap="inferno")
    axes[2].set_title("Disparity Map")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Disparity (px)")

    # Stats as text
    nz = disparity[disparity > 0].astype(np.float32)
    stats = (f"window={args.window}  max_disp={args.max_disp}  agg={args.agg}\n"
             f"valid={valid_pct:.1f}%  time={dt:.2f}s")
    if nz.size:
        stats += f"\nmean={nz.mean():.1f}  p95={np.percentile(nz,95):.0f}  max={nz.max():.0f}"
    fig.text(0.5, 0.01, stats, ha="center", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()


def main():
    args = parse_args()
    assert args.window >= 3 and args.window % 2 == 1, "window must be odd and >=3"

    pairs = discover_pairs(args.pairs_dir)
    if not pairs:
        print(f"No pairs found in {args.pairs_dir}")
        return

    print(f"Found {len(pairs)} stereo pair(s)")
    ensure_dir(args.out_dir)

    for p in pairs:
        print(f"\n=== Pair: {p['name']} ===")
        grayL = load_gray(p["left"])
        grayR = load_gray(p["right"])
        grayL, grayR = resize_gray_pair(grayL, grayR, args.scale)

        t0 = time.perf_counter()

        descL = census_transform(grayL, args.window)
        descR = census_transform(grayR, args.window)

        dispL, dispR = compute_disparity_both(descL, descR, args.max_disp, agg=args.agg)
        ok_mask = lr_check(dispL, dispR, thresh=args.lr_thresh)
        disp_final = postprocess(dispL, ok_mask)

        dt = time.perf_counter() - t0
        valid_pct = ok_mask.mean() * 100.0
        print(f"  window={args.window}  time={dt:.2f}s  valid={valid_pct:.1f}%")

        if not args.no_viz:
            disp_norm = cv2.normalize(disp_final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            tag = f"{p['name']}-w{args.window}-d{args.max_disp}-a{args.agg}"
            save_path = os.path.join(args.out_dir, f"{tag}_viz.png")
            visualize(grayL, grayR, disp_norm, valid_pct, dt, args, save_path=save_path)

    print("Done!")


if __name__ == "__main__":
    main()
