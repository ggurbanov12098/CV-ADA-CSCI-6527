# src/census.py
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _census_numba(gray, window):
    """Numba-accelerated census transform with edge-replicate padding."""
    H, W = gray.shape
    r = window // 2
    nbits = window * window - 1
    nbytes = (nbits + 7) // 8
    desc = np.zeros((H, W, nbytes), dtype=np.uint8)

    for y in prange(H):
        for x in range(W):
            center = gray[y, x]
            bit_idx = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = min(max(y + dy, 0), H - 1)
                    nx = min(max(x + dx, 0), W - 1)
                    if gray[ny, nx] < center:
                        byte_idx = bit_idx >> 3
                        within = bit_idx & 7
                        desc[y, x, byte_idx] |= np.uint8(1 << within)
                    bit_idx += 1
    return desc


def census_transform(gray: np.ndarray, window: int) -> np.ndarray:
    """Census transform: per-pixel binary descriptor comparing neighbours to center."""
    assert window % 2 == 1 and window >= 3
    return _census_numba(gray.astype(np.uint8), window)
