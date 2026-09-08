from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .io import extract_band_properties


def compute_cloud_mask(
    folder_path: Path,
    sza: np.ndarray,
    u_factor: float,
    brightness_threshold: float = 0.35,
    ndvi_cloud_min: float = -0.05,
    ndvi_cloud_max: float = 0.15,
) -> np.ndarray:
    """Detect clouds using TOA reflectance ratio and NIR brightness thresholding.

    Args:
        folder_path (Path): Path to unzipped Sentinel-3 OLCI L1B product directory.
        sza (np.ndarray): Full-resolution Solar Zenith Angle grid in degrees.
        u_factor (float): Inverse-squared Earth-Sun distance correction factor.
        brightness_threshold (float, optional): Absolute TOA reflectance cutoff for
            NIR Band 17 (865 nm). Defaults to 0.35.
        ndvi_cloud_min (float, optional): Minimum TOA NDVI cutoff for cloud detection.
            Defaults to -0.05.
        ndvi_cloud_max (float, optional): Maximum TOA NDVI cutoff for cloud detection.
            Defaults to 0.15.

    Returns:
        np.ndarray: A boolean 2D array of shape (rows, columns) where True indicates
        a cloudy pixel to be masked.
    """
    target_bands = [8, 17]
    band_props = extract_band_properties(folder_path, target_bands, verbose=False)

    cos_sza = np.clip(np.cos(np.radians(sza)), 1e-5, 1.0)
    rho_bands = {}

    for b in target_bands:
        props = band_props[f"band_{b}"]
        e0_adjusted = props["e0"] * u_factor
        with xr.open_dataset(folder_path / props["file"]) as ds:
            var_name = list(ds.data_vars)[0]
            radiance = ds[var_name].values

        rho_bands[b] = (np.pi * radiance) / (e0_adjusted * cos_sza)

    with np.errstate(divide="ignore", invalid="ignore"):
        bright_cloud_mask = rho_bands[17] > brightness_threshold
        init_ndvi = (rho_bands[17] - rho_bands[8]) / (rho_bands[17] + rho_bands[8])

    ndvi_mask = (
        (init_ndvi > ndvi_cloud_min)
        & (init_ndvi < ndvi_cloud_max)
        & (rho_bands[17] > ndvi_cloud_max)
    )
    return bright_cloud_mask | ndvi_mask
