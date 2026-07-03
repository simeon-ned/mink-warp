import numpy as np


def get_epsilon(dtype: np.dtype) -> float:
    return {
        np.dtype("float32"): 1e-5,
        np.dtype("float64"): 1e-10,
    }[dtype]


def skew(x: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from a 3-vector. Supports leading batch dims."""
    x = np.asarray(x)
    if x.shape[-1] != 3:
        raise ValueError(f"Expected last dim 3, got {x.shape}")
    wx, wy, wz = x[..., 0], x[..., 1], x[..., 2]
    o = np.zeros_like(wx)
    return np.stack(
        [
            np.stack([o, -wz, wy], axis=-1),
            np.stack([wz, o, -wx], axis=-1),
            np.stack([-wy, wx, o], axis=-1),
        ],
        axis=-2,
    )
