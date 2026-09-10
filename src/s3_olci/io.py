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
    latitudes: np.ndarray | None = None,
    longitudes: np.ndarray | None = None,
    ndvi_array: np.ndarray | None = None,
) -> None:
    """Export processed surface reflectance products adhering to CF conventions.

    Writes calibrated surface reflectance or computed index grids to a compressed
    NetCDF-4 file, embedding spatial coordinates and standard CF-1.8 attributes.

    Args:
        mode (str): Processing mode ('ndvi', 'rgb', or 'aerosol_rgb').
        output_path (Path): Destination file path for the NetCDF product.
        target_bands (list[int]): List of band numbers included in the product.
        corrected_arrays (dict[str, np.ndarray]): Dictionary mapping band names to
            corrected surface reflectance arrays.
        latitudes (np.ndarray | None, optional): 2D array of pixel latitude
            coordinates in degrees north. Defaults to None.
        longitudes (np.ndarray | None, optional): 2D array of pixel longitude
            coordinates in degrees east. Defaults to None.
        ndvi_array (np.ndarray | None, optional): Processed NDVI array if running in
            'ndvi' mode; otherwise None. Defaults to None.

    Raises:
        ValueError: If an unrecognized mode is provided.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_vars = {}
    coords = {}

    if latitudes is not None and longitudes is not None:
        coords["latitude"] = (
            ["y", "x"],
            latitudes,
            {"units": "degrees_north", "standard_name": "latitude"},
        )
        coords["longitude"] = (
            ["y", "x"],
            longitudes,
            {"units": "degrees_east", "standard_name": "longitude"},
        )

    if mode == "ndvi":
        if ndvi_array is None:
            raise ValueError("NDVI mode requires a valid 'ndvi_array'.")

        data_vars["ndvi"] = (
            ["y", "x"],
            ndvi_array.astype(np.float32),
            {
                "standard_name": "normalized_difference_vegetation_index",
                "long_name": "Surface Normalized Difference Vegetation Index",
                "units": "1",
                "_FillValue": np.nan,
                "valid_range": [-1.0, 1.0],
            },
        )
    elif mode in ["rgb", "aerosol_rgb"]:
        channel_names = ["red", "green", "blue"]
        for idx, band_num in enumerate(target_bands[:3]):
            data_vars[channel_names[idx]] = (
                ["y", "x"],
                corrected_arrays[f"band_{band_num}"].astype(np.float32),
                {
                    "standard_name": "surface_bidirectional_reflectance",
                    "long_name": f"Bottom-of-Rayleigh Surface Reflectance Band {band_num}",
                    "units": "1",
                    "_FillValue": np.nan,
                    "valid_range": [0.0, 1.0],
                },
            )
    else:
        raise ValueError(f"Unsupported export mode: '{mode}'")

    global_attrs = {
        "Conventions": "CF-1.8",
        "title": f"Sentinel-3 OLCI Level-2 Surface Reflectance ({mode.upper()})",
        "source": "Sentinel-3 OLCI L1B Radiance",
        "processing_level": "Level-2",
        "comment": "Rayleigh scattering and ozone gaseous absorption corrected",
    }

    ds_out = xr.Dataset(data_vars=data_vars, coords=coords, attrs=global_attrs)
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.variables}

    ds_out.to_netcdf(output_path, encoding=encoding)

def load_coordinates(folder_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract full-resolution latitude and longitude coordinate grids."""
    coords_file = folder_path / "geo_coordinates.nc"
    with xr.open_dataset(coords_file) as ds:
        latitudes = ds["latitude"].values.astype(np.float32)
        longitudes = ds["longitude"].values.astype(np.float32)
    return latitudes, longitudes