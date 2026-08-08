import numpy as np
import matplotlib.pyplot as plt
import yaml
import xarray as xr
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

OZONE_COEFFS = {

    3: 0.002,   # 442.5 nm
    4: 0.021,   # 490 nm
    5: 0.043,   # 510 nm
    6: 0.105,   # 560 nm
    7: 0.115,   # 620 nm
    8: 0.052,   # 665 nm
    9: 0.043,   # 673.75 nm
    10: 0.035,  # 681.25 nm
    11: 0.020,  # 708.75 nm
    12: 0.009,  # 753.75 nm
    13: 0.007,  # 761.25 nm
    14: 0.008,  # 764.375 nm
    15: 0.008,  # 767.5 nm
    16: 0.006,  # 778.75 nm
}

def get_o3_trans(high_res_sza, high_res_oza, high_res_ozone, band_num):
    """
    Calculates the two-way ozone transmittance T_O3.
    """
    #Convert angles to radians
    sza_rad = np.radians(high_res_sza)
    oza_rad = np.radians(high_res_oza)
    
    # Calculate two-way Air Mass (M)
    cos_sza = np.clip(np.cos(sza_rad), 1e-5, 1.0)
    cos_oza = np.clip(np.cos(oza_rad), 1e-5, 1.0)
    air_mass = (1.0 / cos_sza) + (1.0 / cos_oza)
    
    # Convert ozone from kg/m^2 to atm-cm
    ozone_atm_cm = high_res_ozone * 46.696
    
    # Get the absorption coefficient for this band
    k_o3 = OZONE_COEFFS.get(band_num, 0.0)
    
    # Calculate Transmittance
    t_o3 = np.exp(-k_o3 * ozone_atm_cm * air_mass)
    
    return t_o3
    
def generate_rayleigh(meteo_path, geom_path, target_wavelength_nm):
    """
    Generates a Rayleigh scattering tie-point grid.
    """
    
    meteo = xr.open_dataset(meteo_path)
    geom = xr.open_dataset(geom_path)
    
    # Extract Pressure 
    pressure = meteo['sea_level_pressure']
    
    # Extract Geometry and convert to radians for numpy trig functions
    sza = np.radians(geom['SZA'])  # Solar Zenith Angle
    oza = np.radians(geom['OZA'])  # Observation (Sensor) Zenith Angle
    saa = np.radians(geom['SAA'])  # Solar Azimuth Angle
    oaa = np.radians(geom['OAA'])  # Observation Azimuth Angle
    
    # Calculate Relative Azimuth Angle
    rel_azimuth = saa - oaa
    rel_azimuth = np.mod(rel_azimuth + np.pi, 2*np.pi) - np.pi
    
    # Calculate Scattering Angle (Theta)
    cos_theta = -np.cos(sza) * np.cos(oza) + np.sin(sza) * np.sin(oza) * np.cos(rel_azimuth)
    
    # Calculate the Rayleigh Phase Function P_R(Theta)
    phase_function = 0.75 * (1 + cos_theta**2)

    # Convert wavelength to microns
    wl_um = target_wavelength_nm / 1000.0
    
    # Calculate base optical depth for standard pressure 
    tau_r0 = 0.008569 * (wl_um ** -4) * (1 + 0.0113 * (wl_um ** -2) + 0.00013 * (wl_um ** -4))
    
    # Scale optical depth by the actual pressure at each tie-point
    P0_standard = 1013.25 # hPa
    tau_r = tau_r0 * (pressure / P0_standard)

    # Calculate Rayleigh Reflectance 
    rayleigh_reflectance = (tau_r * phase_function) / (4 * np.cos(sza) * np.cos(oza))
    
    rayleigh = xr.Dataset({
        'rayleigh_reflectance': rayleigh_reflectance,
        'optical_depth': tau_r,
        'phase_function': phase_function
    })
    
    return rayleigh

def upscale_rayleigh(rayleigh_ds, l1b_band_path):
    """
    Loads a coarse Rayleigh calculation and upscales it to the full image resolution. 
    """
    
    coarse_array = rayleigh_ds['rayleigh_reflectance'].values
    num_tie_rows, num_tie_cols = coarse_array.shape
    
    full_res = xr.open_dataset(l1b_band_path)

    var_name = list(full_res.data_vars)[0] 
    num_full_rows, num_full_cols = full_res[var_name].shape
    
    # Map the tie-points across the full index space of the image.
    coarse_y_coords = np.linspace(0, num_full_rows - 1, num_tie_rows)
    coarse_x_coords = np.linspace(0, num_full_cols - 1, num_tie_cols)
    
    # Initialize the interpolator. 
    interpolator = RegularGridInterpolator(
        (coarse_y_coords, coarse_x_coords), 
        coarse_array, 
        method='linear',
        bounds_error=False,
        fill_value=None 
    )
    
    # Create the dense pixel grid 
    full_y_coords = np.arange(num_full_rows)
    full_x_coords = np.arange(num_full_cols)
    
    Y, X = np.meshgrid(full_y_coords, full_x_coords, indexing='ij')
    
    target_points = np.stack([Y.ravel(), X.ravel()], axis=-1)

    # Interpolate, reshape, and return to array
    high_res_values = interpolator(target_points)
    
    high_res_rayleigh = high_res_values.reshape((num_full_rows, num_full_cols))

    da_high_res = xr.DataArray(
        high_res_rayleigh,
        dims=['rows', 'columns'],
        coords={'rows': full_y_coords, 'columns': full_x_coords},
        name='high_res_rayleigh_reflectance'
    )
    
    return da_high_res


def upscale_tie_variable(tie_nc_path, variable_name, l1b_band_path):
    """
    Upscale any coarse tie-point variable to the full image resolution.
    """
    
    # Load the coarse tie-point variable
    tie_ds = xr.open_dataset(tie_nc_path)
    coarse_array = tie_ds[variable_name].values
    num_tie_rows, num_tie_cols = coarse_array.shape
    
    # Get the target dimensions 
    full_res = xr.open_dataset(l1b_band_path)
    target_var = list(full_res.data_vars)[0]
    num_full_rows, num_full_cols = full_res[target_var].shape
    
    # Interpolate
    coarse_y = np.linspace(0, num_full_rows - 1, num_tie_rows)
    coarse_x = np.linspace(0, num_full_cols - 1, num_tie_cols)
    
    interpolator = RegularGridInterpolator(
        (coarse_y, coarse_x), 
        coarse_array, 
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    Y, X = np.meshgrid(np.arange(num_full_rows), np.arange(num_full_cols), indexing='ij')
    target_points = np.stack([Y.ravel(), X.ravel()], axis=-1)
    
    high_res_array = interpolator(target_points).reshape((num_full_rows, num_full_cols))
    
    return high_res_array

def get_earth_sun_correction(folder_path_name):
    """
    Extracts the date and calculates the inverse squared Earth-Sun distance correction factor (1/d^2).
    """

    date_str = folder_path_name.split('/')[-1][16:24]
    
    dt = datetime.strptime(date_str, "%Y%m%d")
    doy = dt.timetuple().tm_yday
    
    # Calculate Earth-Sun distance 
    theta = (2 * np.pi / 365.256363) * (doy - 4)
    d = 1.0 - 0.01674 * np.cos(theta)
    
    # Calculate the correction factor 
    u_factor = 1.0 / (d ** 2)
    
    return u_factor

def apply_rayleigh_correction(radiance_path, high_res_rayleigh, high_res_sza, solar_flux,t_o3):
    """
    Converts Radiance to Reflectance, then subtracts the Rayleigh signal.
    """
    
    ds_rad = xr.open_dataset(radiance_path)
    var_name = list(ds_rad.data_vars)[0]
    L_toa = ds_rad[var_name].values
    
    # Convert SZA to radians
    sza_rad = np.radians(high_res_sza)
    
    rho_toa = (np.pi * L_toa) / (solar_flux * np.cos(sza_rad))

    # Apply Ozone Transmittance Correction
    rho_toa_gas_corrected = rho_toa / t_o3
    
    # Calculate correction
    rayleigh_corrected_reflectance = rho_toa_gas_corrected - high_res_rayleigh
    
    # Clip negative values
    rayleigh_corrected_reflectance = rayleigh_corrected_reflectance.clip(min=0)
    
    return rayleigh_corrected_reflectance

def extract_band_properties(folder_path, band_numbers):
    """
    Dynamically extracts the wavelength and solar flux.
    """
    instrument_file = f"{folder_path}/instrument_data.nc"
    
    ds_instr = xr.open_dataset(instrument_file)
    
    bands_dict = {}
    
    for band_num in band_numbers:

        idx = band_num - 1
        
        # Extract the data store in dict
        wl = ds_instr['lambda0'].values[idx, :].mean()
        e0 = ds_instr['solar_flux'].values[idx, :].mean()
        
        file_name = f"Oa{band_num:02d}_radiance.nc"
        
        bands_dict[f"band_{band_num}"] = {
            'file': file_name,
            'wl': float(wl),
            'e0': float(e0)
        }
        
        print(f"  -> Band {band_num:02d}: Wavelength = {wl:.2f} nm, E0 = {e0:.2f}")
        
    ds_instr.close()
    
    return bands_dict


def main():
    # Load Configuration
    config = load_config("config.yaml")
    folder_path = config['input']['folder_path']
    mode = config['processing']['mode']
    gamma = config['visualization']['gamma']
    
    print(f"Processing in '{mode.upper()}' mode...")

    # Determine Bands based on Mode
    if mode in ['rgb', 'aerosol_rgb']:
        target_bands = config['processing']['rgb_bands']
        
        # If aerosol correction is on, append Near-Infrared band
        if mode == 'aerosol_rgb' and 17 not in target_bands:
            target_bands.append(17) 
            
    elif mode == 'ndvi':
        target_bands = [8, 17]
    else:
        raise ValueError(f"Unknown mode: {mode}. Check your config.yaml.")

    # Setup & Geometry
    bands = extract_band_properties(folder_path, target_bands)
    geom_file = f"{folder_path}/tie_geometries.nc"
    meteo_file = f"{folder_path}/tie_meteo.nc"
    
    reference_file = f"{folder_path}/{bands[f'band_{target_bands[-1]}']['file']}"
    sza_high_res = upscale_tie_variable(geom_file, 'SZA', reference_file)

    oza_high_res = upscale_tie_variable(geom_file, 'OZA', reference_file)
    ozone_high_res = upscale_tie_variable(meteo_file, 'total_ozone', reference_file)

    u_factor = get_earth_sun_correction(folder_path)

    # Generate Rayleigh corrections
    corrected_arrays = {}
    for band_key, props in bands.items():
        
        radiance_path = f"{folder_path}/{props['file']}"

        adjusted_e0 = props['e0'] * u_factor
        
        rayleigh_ds = generate_rayleigh(meteo_file, geom_file, props['wl'])
        high_res_rayleigh = upscale_rayleigh(rayleigh_ds, radiance_path)

        band_num = int(band_key.split('_')[1])
        t_o3 = get_o3_trans(sza_high_res, oza_high_res, ozone_high_res, band_num)

        corrected_image = apply_rayleigh_correction(radiance_path, high_res_rayleigh, sza_high_res, adjusted_e0,t_o3)
        corrected_arrays[band_key] = corrected_image.values

    # Processing
    plt.figure(figsize=(12, 10))

    if mode == 'ndvi':
        
        red = corrected_arrays['band_8']
        nir = corrected_arrays['band_17']
        
        denominator = (nir + red)
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = np.where(denominator != 0, (nir - red) / denominator, np.nan)
        
        plt.imshow(ndvi, cmap=config['visualization']['ndvi_cmap'], vmin=-0.2, vmax=0.9)
        plt.colorbar(label="NDVI")
        plt.title("NDVI Map", fontsize=14)
        
    elif mode in ['rgb', 'aerosol_rgb']:
        
        if mode == 'aerosol_rgb':
            
            haze_map = corrected_arrays['band_17']
            water_mask = haze_map < 0.1
            # Apply dynamic dark object subtraction strictly to the visual bands
            for b in target_bands[:3]:
                corrected_arrays[f'band_{b}'] = np.where(
                    water_mask, 
                    corrected_arrays[f'band_{b}'] - haze_map, 
                    corrected_arrays[f'band_{b}']
                )

        # Stack and Normalize RGB dynamically
        l2_rgb = np.dstack((
            corrected_arrays[f'band_{target_bands[0]}'], 
            corrected_arrays[f'band_{target_bands[1]}'], 
            corrected_arrays[f'band_{target_bands[2]}']
        ))
        
        rgb_min, rgb_max = np.nanpercentile(l2_rgb, 1), np.nanpercentile(l2_rgb, 95)
        l2_normalized = np.clip((l2_rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)
        
        l2_enhanced = l2_normalized ** (1.0 / gamma)
        plt.imshow(l2_enhanced)
        plt.title(f"True Color ({mode})", fontsize=14)

    # Output
    plt.axis('off')
    output_name = f"Output_{mode}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    if mode == 'ndvi':
        # Create a NetCDF dataset
        ds_out = xr.Dataset(
            data_vars={"ndvi": (["y", "x"], ndvi)},
            attrs={"description": "NDVI", "sensor": "Optical"}
        )
    elif mode in ['rgb', 'aerosol_rgb']:
        # Create a NetCDF dataset 
        ds_out = xr.Dataset(
            data_vars={
                "red": (["y", "x"], corrected_arrays[f'band_{target_bands[0]}']),
                "green": (["y", "x"], corrected_arrays[f'band_{target_bands[1]}']),
                "blue": (["y", "x"], corrected_arrays[f'band_{target_bands[2]}'])
            },
            attrs={"description": f"Composite ({mode})", "sensor": "Optical"}
        )
        
    nc_output_name = f"Processed_Data_{mode}.nc"
    ds_out.to_netcdf(nc_output_name)
    plt.show()

if __name__ == "__main__":
    main()