from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

logger = logging.getLogger(__name__)

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Read configuration parameters from a YAML file."""
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_band_properties(folder_path: Path, band_numbers: list[int], verbose: bool = True) -> dict[str, dict]:
    """Dynamically extract central wavelengths and nominal solar flux (E0).

    Reads `instrument_data.nc` from the Sentinel-3 L1B product to retrieve
    band-specific solar irradiance and nominal center wavelengths.

    Args:
        folder_path (Path): Path to the Sentinel-3 OLCI L1B product directory.
        band_numbers (list[int]): List of integer band numbers to extract (e.g., [8, 17]).
        verbose (bool, optional): If True, logs extracted band properties. Defaults to True.

    Returns:
        dict[str, dict]: Nested dictionary mapping band keys (e.g., 'band_8') to
        dictionaries containing 'file', 'wl' (nm), and 'e0' (mW m^-2 nm^-1).
    """
    instrument_file = folder_path / "instrument_data.nc"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with xr.open_dataset(instrument_file) as ds_instr:
            bands_dict = {}
            for band_num in band_numbers:
                idx = band_num - 1
                wl = float(ds_instr["lambda0"].values[idx, :].mean())
                e0 = float(ds_instr["solar_flux"].values[idx, :].mean())

                bands_dict[f"band_{band_num}"] = {
                    "file": f"Oa{band_num:02d}_radiance.nc",
                    "wl": wl,
                    "e0": e0,
                }
                if verbose:
                    logger.info("Band %02d: Wavelength = %.2f nm, E0 = %.2f", band_num, wl, e0)

    return bands_dict


def load_radiance_bands(folder_path: Path,band_info: dict,cloud_mask: np.ndarray | None = None,) -> dict[str, np.ndarray]:
    """Load raw radiance arrays and optionally apply a cloud mask.

    Args:
        folder_path (Path): Path to the Sentinel-3 OLCI L1B product directory.
        band_info (dict): Dictionary mapping band names to properties ('file', 'wl', 'e0').
        cloud_mask (np.ndarray | None, optional): 2D boolean mask where True indicates
            a cloud pixel to be masked as NaN. Defaults to None.

    Returns:
        dict[str, np.ndarray]: Dictionary mapping band keys to raw radiance arrays
        (with cloud pixels masked as NaN if a cloud mask was provided).
    """
    radiance = {}
    for band_key, props in band_info.items():
        radiance_path = folder_path / props["file"]
        with xr.open_dataset(radiance_path) as ds:
            var_name = list(ds.data_vars)[0]
            raw_rad = ds[var_name].values

        if cloud_mask is not None:
            raw_rad = np.where(cloud_mask, np.nan, raw_rad)

        radiance[band_key] = raw_rad

    return radiance


def export_netcdf(
    mode: str,
    output_path: Path,
    target_bands: list[int],
    corrected_arrays: dict[str, np.ndarray],
    ndvi_array: np.ndarray | None = None,
) -> None:
    """Export processed surface reflectance products to a NetCDF file.

    Args:
        mode (str): Processing mode ('ndvi', 'rgb', or 'aerosol_rgb').
        output_path (Path): Destination file path for the NetCDF product.
        target_bands (list[int]): List of band numbers included in the product.
        corrected_arrays (dict[str, np.ndarray]): Dictionary mapping band names to
            corrected surface reflectance arrays.
        ndvi_array (np.ndarray | None, optional): Processed NDVI array if running in
            'ndvi' mode; otherwise None. Defaults to None.

    Raises:
        ValueError: If an unrecognized mode is provided.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "ndvi":
        ds_out = xr.Dataset(
            data_vars={"ndvi": (["y", "x"], ndvi_array)},
            attrs={"description": "Sentinel-3 OLCI Surface NDVI", "sensor": "OLCI"},
        )
    elif mode in ["rgb", "aerosol_rgb"]:
        ds_out = xr.Dataset(
            data_vars={
                "red": (["y", "x"], corrected_arrays[f"band_{target_bands[0]}"]),
                "green": (["y", "x"], corrected_arrays[f"band_{target_bands[1]}"]),
                "blue": (["y", "x"], corrected_arrays[f"band_{target_bands[2]}"]),
            },
            attrs={
                "description": f"Sentinel-3 OLCI Surface Composite ({mode})",
                "sensor": "OLCI",
            },
        )
    else:
        raise ValueError(f"Unsupported export mode: '{mode}'")

    ds_out.to_netcdf(output_path)
