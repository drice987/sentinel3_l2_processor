from __future__ import annotations

import numpy as np


def compute_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Vegetation Index (NDVI)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nir_band - red_band) / (nir_band + red_band)


def generate_rgb_composite(corrected_arrays: dict[str, np.ndarray],target_bands: list[int],mode: str = "rgb",gamma: float = 1.0,) -> np.ndarray:
    """Process aerosol subtractions and normalize RGB channels for visualization.

    Args:
        corrected_arrays (dict[str, np.ndarray]): Dictionary mapping band keys to
            corrected surface reflectance arrays.
        target_bands (list[int]): List of band numbers corresponding to RGB channels
            (e.g., [8, 6, 4] for Red, Green, Blue).
        mode (str, optional): Composite mode ('rgb' or 'aerosol_rgb'). Defaults to 'rgb'.
        gamma (float, optional): Gamma correction factor applied to enhance contrast.
            Defaults to 1.0.

    Returns:
        np.ndarray: 3D array of shape (rows, cols, 3) representing normalized,
        gamma-corrected RGB values in range [0, 1].
    """
    bands_copy = {k: v.copy() for k, v in corrected_arrays.items()}

    # Optional Near-Infrared dark-target haze subtraction over water
    if mode == "aerosol_rgb":
        haze_map = bands_copy["band_17"]
        water_mask = haze_map < 0.1
        for b in target_bands[:3]:
            key = f"band_{b}"
            bands_copy[key] = np.where(
                water_mask, bands_copy[key] - haze_map, bands_copy[key]
            )

    rgb = np.dstack(
        (
            bands_copy[f"band_{target_bands[0]}"],
            bands_copy[f"band_{target_bands[1]}"],
            bands_copy[f"band_{target_bands[2]}"],
        )
    )

    # 1st to 95th percentile contrast stretching
    rgb_min, rgb_max = np.nanpercentile(rgb, 1), np.nanpercentile(rgb, 95)
    normalized = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0.0, 1.0)

    return normalized ** (1.0 / gamma)
