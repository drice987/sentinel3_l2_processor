from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def upscale_tie_grid(
    coarse_array: np.ndarray, target_shape: tuple[int, int], method: str = "linear"
) -> np.ndarray:
    """Upscale any coarse tie-point variable grid to the full image resolution.

    Args:
        coarse_array (np.ndarray): 2D numpy array of coarse tie-points.
        target_shape (tuple[int, int]): Dimensions (rows, cols) of the destination grid.
        method (str, optional): Interpolation method ('linear', 'nearest'). Defaults to 'linear'.

    Returns:
        np.ndarray: High-resolution 2D array matching `target_shape`.
    """
    num_tie_rows, num_tie_cols = coarse_array.shape
    num_full_rows, num_full_cols = target_shape

    coarse_y = np.linspace(0, num_full_rows - 1, num_tie_rows)
    coarse_x = np.linspace(0, num_full_cols - 1, num_tie_cols)

    interpolator = RegularGridInterpolator(
        (coarse_y, coarse_x),
        coarse_array,
        method=method,
        bounds_error=False,
        fill_value=None,
    )

    y_grid, x_grid = np.meshgrid(
        np.arange(num_full_rows), np.arange(num_full_cols), indexing="ij"
    )
    points = np.stack([y_grid.ravel(), x_grid.ravel()], axis=-1)

    return interpolator(points).reshape(target_shape)
