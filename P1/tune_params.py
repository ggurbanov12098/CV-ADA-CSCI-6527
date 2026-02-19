#!/usr/bin/env python3
"""Grid-search tuner for Census stereo parameters."""
import argparse
import csv
import os
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np

from src.census import census_transform
from src.disparity import compute_disparity_both, lr_check, postprocess, warmup
from src.utils import load_gray, resize_gray_pair, discover_pairs, ensure_dir


def _percentile(x, q):
    return float(np.percentile(x, q)) if x.size else 0.0


def _evaluate(job):
    """Evaluate one config."""
    cfg, descL, descR, t_census = job

    t0 = time.perf_counter()
    dispL, dispR = compute_disparity_both(descL, descR, cfg["max_disp"], agg=cfg["agg"])
    ok = lr_check(dispL, dispR, thresh=cfg["lr_thresh"])
    disp = postprocess(dispL, ok)
    t_match = time.perf_counter() - t0

    valid = float(ok.mean() * 100)
    nz = disp[disp > 0].astype(np.float32)
    p95 = _percentile(nz, 95)
    p99 = _percentile(nz, 99)
    spread = _percentile(nz, 95) - _percentile(nz, 5) if nz.size else 0
    t_total = t_census + t_match

    clip_pen = 0.0
    if nz.size and (p99 / max(cfg["max_disp"], 1)) > 0.95:
        clip_pen = 10 * ((p99 / max(cfg["max_disp"], 1)) - 0.95)

    # Mild log-time penalty: ~0 at 1s, ~3.5 at 10s, ~5.3 at 30s, ~6 at 60s
    time_pen = 3 * np.log1p(t_total)
    score = valid + 0.2 * spread - time_pen - clip_pen

    return {**cfg, "score": score, "valid": valid, "time": t_total,
            "p95": p95, "p99": p99, "max": float(nz.max()) if nz.size else 0}


def main():
    ap = argparse.ArgumentParser(description="Tune stereo parameters")
    ap.add_argument("--pairs_dir", default="images/original")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--max_disps", type=int, nargs="+", default=[64, 96, 128])
    ap.add_argument("--windows", type=int, nargs="+", default=[7, 9, 11])
    ap.add_argument("--aggs", type=int, nargs="+", default=[25, 51, 75])
    ap.add_argument("--lr_threshes", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    n_workers = args.workers or multiprocessing.cpu_count()
    print(f"Using {n_workers} workers. Warming up Numba...")
    warmup()
    print("Done.\n")

    pairs = discover_pairs(args.pairs_dir)
    if not pairs:
        raise SystemExit(f"No pairs in {args.pairs_dir}")

    out_csv = "images/outputs/tune_log.csv"
    ensure_dir(os.path.dirname(out_csv))
    fields = ["pair", "scale", "max_disp", "window", "agg", "lr_thresh",
              "score", "valid", "time", "p95", "p99", "max"]

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            wr.writeheader()

        for p in pairs:
            print(f"== Pair: {p['name']} ==")

            # Cache census transforms
            cache = {}
            for scale, window in product(args.scales, args.windows):
                gL, gR = load_gray(p["left"]), load_gray(p["right"])
                gL, gR = resize_gray_pair(gL, gR, scale)
                t0 = time.perf_counter()
                dL, dR = census_transform(gL, window), census_transform(gR, window)
                cache[(scale, window)] = (dL, dR, time.perf_counter() - t0)

            # Build jobs
            jobs = []
            for scale, max_disp, window, agg, lr_thresh in product(
                args.scales, args.max_disps, args.windows, args.aggs, args.lr_threshes
            ):
                dL, dR, tc = cache[(scale, window)]
                cfg = dict(scale=scale, max_disp=max_disp, window=window,
                           agg=agg, lr_thresh=lr_thresh)
                jobs.append((cfg, dL, dR, tc))

            print(f"  {len(jobs)} configs...")
            results = []
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for r in ex.map(_evaluate, jobs):
                    r["pair"] = p["name"]
                    results.append(r)

            wr.writerows(results)
            f.flush()

            # Print top results
            results.sort(key=lambda r: r["score"], reverse=True)
            print(f"\n  Top {min(args.topk, len(results))} by score:")
            for r in results[:args.topk]:
                print(f"    score={r['score']:.1f} valid={r['valid']:.1f}% "
                      f"t={r['time']:.2f}s | s={r['scale']} d={r['max_disp']} "
                      f"w={r['window']} agg={r['agg']} lr={r['lr_thresh']} "
                      f"| p99={r['p99']:.0f}")

    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()