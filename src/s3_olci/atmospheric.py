from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from .constants import OZONE_COEFFS, OZONE_KG_M2_TO_ATM_CM, P0_STANDARD
from .interpolation import upscale_tie_grid


def calculate_earth_sun_correction(folder_path: Path) -> float:
    """Calculate the inverse-squared Earth-Sun distance correction factor (1/d^2).

    Extracts the acquisition date from the Sentinel-3 SAFE product folder name

    Args:
        folder_path (Path): Path to the Sentinel-3 OLCI L1B product directory.

    Returns:
        float: Inverse-squared Earth-Sun distance correction factor (u_factor).
    """
    folder_name = folder_path.name
    date_str = folder_name[16:24]
    dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    doy = dt.timetuple().tm_yday

    theta = (2.0 * np.pi / 365.256363) * (doy - 4)
    d = 1.0 - 0.01674 * np.cos(theta)
    return 1.0 / (d**2)


def get_ozone_transmittance(sza_deg: np.ndarray, oza_deg: np.ndarray, ozone_col: np.ndarray, band_num: int) -> np.ndarray:
    """Calculate the two-way atmospheric ozone transmittance (T_O3).

    Args:
        sza_deg (np.ndarray): High-resolution Solar Zenith Angle grid in degrees.
        oza_deg (np.ndarray): High-resolution Observation Zenith Angle grid in degrees.
        ozone_col (np.ndarray): Total column ozone grid in kg/m^2.
        band_num (int): Sentinel-3 OLCI band number (1-21) to select k_O3.

    Returns:
        np.ndarray: Two-way ozone transmittance array in range [0, 1].
    """
    sza_rad = np.radians(sza_deg)
    oza_rad = np.radians(oza_deg)

    cos_sza = np.clip(np.cos(sza_rad), 1e-5, 1.0)
    cos_oza = np.clip(np.cos(oza_rad), 1e-5, 1.0)
    air_mass = (1.0 / cos_sza) + (1.0 / cos_oza)

    ozone_atm_cm = ozone_col * OZONE_KG_M2_TO_ATM_CM
    k_o3 = OZONE_COEFFS.get(band_num, 0.0)

    return np.exp(-k_o3 * ozone_atm_cm * air_mass)


def compute_rayleigh_reflectance(meteo_ds: xr.Dataset, geom_ds: xr.Dataset, wavelength_nm: float) -> np.ndarray:
    """Calculate Rayleigh scattering reflectance on the coarse tie-point grid.

    Computes Rayleigh optical thickness scaled by local sea-level pressure,
    evaluates the scattering phase function, and determines primary reflectance.

    Args:
        meteo_ds (xr.Dataset): Dataset containing meteorological tie-points (`tie_meteo.nc`).
        geom_ds (xr.Dataset): Dataset containing geometry tie-points (`tie_geometries.nc`).
        wavelength_nm (float): Central wavelength of the target channel in nanometers.

    Returns:
        np.ndarray: 2D array of Rayleigh reflectance on the coarse tie-point grid.
    """
    pressure = meteo_ds["sea_level_pressure"].values
    sza = np.radians(geom_ds["SZA"].values)
    oza = np.radians(geom_ds["OZA"].values)
    saa = np.radians(geom_ds["SAA"].values)
    oaa = np.radians(geom_ds["OAA"].values)

    rel_azimuth = np.mod(saa - oaa + np.pi, 2 * np.pi) - np.pi
    cos_theta = -np.cos(sza) * np.cos(oza) + np.sin(sza) * np.sin(oza) * np.cos(
        rel_azimuth
    )
    phase_function = 0.75 * (1.0 + cos_theta**2)

    wl_um = wavelength_nm / 1000.0
    tau_r0 = (
        0.008569 * (wl_um**-4) * (1.0 + 0.0113 * (wl_um**-2) + 0.00013 * (wl_um**-4))
    )
    tau_r = tau_r0 * (pressure / P0_STANDARD)

    return (tau_r * phase_function) / (4.0 * np.cos(sza) * np.cos(oza))


def load_and_upscale_geometries(
    folder_path: Path, reference_band_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load and upscale all angular and meteorological tie-points to image resolution.

    Args:
        folder_path (Path): Path to the unzipped Sentinel-3 OLCI L1B product directory.
        reference_band_path (Path): Path to a full-resolution reference NetCDF band file.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - sza (np.ndarray): High-resolution Solar Zenith Angle grid in degrees.
            - oza (np.ndarray): High-resolution Observation Zenith Angle grid in degrees.
            - ozone (np.ndarray): High-resolution total column ozone grid in kg/m^2.
            - u_factor (float): Inverse-squared Earth-Sun distance correction factor.
    """
    with xr.open_dataset(reference_band_path) as ref_ds:
        first_var = list(ref_ds.data_vars)[0]
        target_shape = ref_ds[first_var].shape

    with xr.open_dataset(folder_path / "tie_geometries.nc") as geom_ds:
        sza = upscale_tie_grid(geom_ds["SZA"].values, target_shape)
        oza = upscale_tie_grid(geom_ds["OZA"].values, target_shape)

    with xr.open_dataset(folder_path / "tie_meteo.nc") as meteo_ds:
        ozone = upscale_tie_grid(meteo_ds["total_ozone"].values, target_shape)

    u_factor = calculate_earth_sun_correction(folder_path)
    return sza, oza, ozone, u_factor


def correct_surface_reflectance(
    folder_path: Path,
    bands: dict[str, dict],
    radiances: dict[str, np.ndarray],
    sza: np.ndarray,
    oza: np.ndarray,
    ozone: np.ndarray,
    u_factor: float,
) -> dict[str, np.ndarray]:
    """Apply Rayleigh scattering and gaseous ozone corrections to target bands.

    Converts Top-of-Atmosphere (TOA) radiance to reflectance, corrects for ozone
    absorption along the viewing path, and subtracts interpolated Rayleigh reflectance.

    Args:
        folder_path (Path): Path to the Sentinel-3 OLCI L1B product directory.
        bands (dict[str, dict]): Dictionary mapping band names to metadata dictionaries
            containing 'wl', 'e0', and 'file'.
        radiances (dict[str, np.ndarray]): Dictionary mapping band names to raw TOA
            radiance arrays.
        sza (np.ndarray): High-resolution Solar Zenith Angle grid in degrees.
        oza (np.ndarray): High-resolution Observation Zenith Angle grid in degrees.
        ozone (np.ndarray): High-resolution total column ozone grid in kg/m^2.
        u_factor (float): Inverse-squared Earth-Sun distance correction factor.

    Returns:
        dict[str, np.ndarray]: Dictionary mapping band keys to bottom-of-Rayleigh
        surface reflectance arrays (clipped at minimum 0.0).
    """
    corrected = {}
    cos_sza = np.clip(np.cos(np.radians(sza)), 1e-5, 1.0)

    meteo_path = folder_path / "tie_meteo.nc"
    geom_path = folder_path / "tie_geometries.nc"

    with xr.open_dataset(meteo_path) as meteo_ds, xr.open_dataset(geom_path) as geom_ds:
        for band_key, props in bands.items():
            band_num = int(band_key.split("_")[1])
            adjusted_e0 = props["e0"] * u_factor

            # Rayleigh reflectance
            coarse_rayleigh = compute_rayleigh_reflectance(
                meteo_ds, geom_ds, props["wl"]
            )
            high_res_rayleigh = upscale_tie_grid(
                coarse_rayleigh, target_shape=sza.shape
            )

            # Ozone transmission
            t_o3 = get_ozone_transmittance(sza, oza, ozone, band_num)

            # Invert to surface reflectance
            rho_toa = (np.pi * radiances[band_key]) / (adjusted_e0 * cos_sza)
            rho_gas_corrected = rho_toa / t_o3
            rho_surface = np.clip(rho_gas_corrected - high_res_rayleigh, 0.0, None)

            corrected[band_key] = rho_surface

    return corrected
