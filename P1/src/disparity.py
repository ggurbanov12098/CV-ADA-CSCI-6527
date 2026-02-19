import numpy as np
import cv2
from numba import njit, prange

_POPCNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


@njit(parallel=True, cache=True)
def _disparity_noagg(descL, descR, dmax, popcnt):
    H, W, B = descL.shape
    disp = np.zeros((H, W), dtype=np.uint16)
    for y in prange(H):
        for x in range(W):
            best = np.int32(999_999)
            dx_end = min(dmax + 1, x + 1)
            for dx in range(dx_end):
                cost = np.int32(0)
                for b in range(B):
                    cost += popcnt[descL[y, x, b] ^ descR[y, x - dx, b]]
                if cost < best:
                    best = cost
                    disp[y, x] = dx
    return disp


@njit(parallel=True, cache=True)
def _hamming_cost_slice(descL, descR, dx, popcnt):
    H, W, B = descL.shape
    cost = np.full((H, W), 999_999, dtype=np.int32)
    for y in prange(H):
        for x in range(dx, W):
            c = np.int32(0)
            for b in range(B):
                c += popcnt[descL[y, x, b] ^ descR[y, x - dx, b]]
            cost[y, x] = c
    return cost


@njit(parallel=True, cache=True)
def _wta_update(best_cost, best_disp, cost, d):
    H, W = best_cost.shape
    for y in prange(H):
        for x in range(W):
            if cost[y, x] < best_cost[y, x]:
                best_cost[y, x] = cost[y, x]
                best_disp[y, x] = d


@njit(parallel=True, cache=True)
def _lr_check(dispL, dispR, thresh):
    H, W = dispL.shape
    ok = np.zeros((H, W), dtype=np.bool_)
    for y in prange(H):
        for x in range(W):
            d = np.int32(dispL[y, x])
            xr = x - d
            if 0 <= xr < W:
                diff = d - np.int32(dispR[y, xr])
                if diff < 0:
                    diff = -diff
                if diff <= thresh:
                    ok[y, x] = True
    return ok


def compute_disparity(descL, descR, dmax, agg=0):
    """Left disparity map from census descriptors."""
    if not agg or agg < 3:
        return _disparity_noagg(descL, descR, dmax, _POPCNT_LUT)
    H, W, _ = descL.shape
    best_cost = np.full((H, W), 999_999, dtype=np.int32)
    best_disp = np.zeros((H, W), dtype=np.uint16)
    for dx in range(dmax + 1):
        cost = _hamming_cost_slice(descL, descR, dx, _POPCNT_LUT)
        cost = cv2.boxFilter(cost, -1, (agg, agg), normalize=False,
                             borderType=cv2.BORDER_REPLICATE)
        _wta_update(best_cost, best_disp, cost, dx)
    return best_disp


def compute_disparity_both(descL, descR, dmax, agg=0):
    """Compute left and right disparity maps (right via flip trick)."""
    dL = compute_disparity(descL, descR, dmax, agg=agg)
    dR = compute_disparity(
        np.ascontiguousarray(descR[:, ::-1, :]),
        np.ascontiguousarray(descL[:, ::-1, :]),
        dmax, agg=agg,
    )[:, ::-1].copy()
    return dL, dR


def lr_check(dispL, dispR, thresh=1):
    """Boolean mask: True where L/R disparities agree within thresh."""
    return _lr_check(dispL.astype(np.uint16), dispR.astype(np.uint16), thresh)


def postprocess(disp, ok_mask):
    """Zero out invalid pixels and apply 5x5 median filter."""
    out = disp.copy()
    out[~ok_mask] = 0
    return cv2.medianBlur(out.astype(np.uint16), 5)


def warmup():
    """Pre-compile Numba kernels."""
    t = np.zeros((4, 4, 1), dtype=np.uint8)
    _disparity_noagg(t, t, 1, _POPCNT_LUT)
    _hamming_cost_slice(t, t, 0, _POPCNT_LUT)
    _wta_update(np.zeros((4, 4), np.int32), np.zeros((4, 4), np.uint16),
                np.zeros((4, 4), np.int32), 0)
    _lr_check(np.zeros((4, 4), np.uint16), np.zeros((4, 4), np.uint16), 1)
