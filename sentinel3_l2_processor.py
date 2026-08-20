import numpy as np
import matplotlib.pyplot as plt
import yaml
import xarray as xr
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import warnings

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_radiance(folder_path: str, band: dict, sza_high_res: np.ndarray, u_factor: float, cloud_config: dict) -> dict:
    """Loads raw radiance arrays and applies a cloud mask if enabled.
    Args:
    folder_path (str): Path to the Sentinel-3 L1B product folder.
        band (dict): Dictionary of band properties (wavelength, E0, filename).
        sza_high_res (numpy.ndarray): High-resolution Solar Zenith Angle grid.
        u_factor (float): Earth-Sun distance correction factor.
        cloud_config (dict): Cloud masking configuration settings.

    Returns:
        dict: Dictionary mapping band names to raw radiance arrays (with clouds masked as NaN).
    """

    if cloud_config.get('enabled',False):
        cloud_mask = get_cloud_mask(folder_path, sza_high_res, u_factor, brightness_threshold=cloud_config.get('brightness_threshold', 0.5), ndvi_cloud_min = cloud_config.get('ndvi_cloud_min',-0.05), ndvi_cloud_max = cloud_config.get('ndvi_cloud_max', 0.15))
    else:
        cloud_mask = None
    
    radiance = {}
    for band_key, props in band.items():
        radiance_path = f"{folder_path}/{props['file']}"

        with xr.open_dataset(radiance_path) as ds:
            var_name = list(ds.data_vars)[0]
            raw_rad = ds[var_name].values


        if cloud_mask is not None:
            raw_rad = np.where(cloud_mask, np.nan, raw_rad)

        radiance[band_key] = raw_rad

    return radiance

def get_o3_trans(high_res_sza: np.ndarray, high_res_oza: np.ndarray, high_res_ozone: np.ndarray, band_num: int) -> np.ndarray:
    """Calculates the two-way ozone transmittance T_O3.

    Args:
        high_res_sza (numpy.ndarray): High-resolution Solar Zenith Angle grid in degrees.
        high_res_oza (numpy.ndarray): High-resolution Observation Zenith Angle grid in degrees.
        high_res_ozone (numpy.ndarray): High-resolution total column ozone grid.
        band_num (int): The Sentinel-3 band number

    Returns:
        numpy.ndarray: Transmittance array for the specified band
    """
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
    
def generate_rayleigh(meteo_path: str, geom_path: str, target_wavelength_nm: float):
    """Generates a Rayleigh scattering tie-point grid.

    Args:
        meteo_path (str): Path to the tie_meteo.nc file.
        geom_path (str): Path to the tie_geometries.nc file.
        target_wavelength_nm (float): Central wavelength of the target band in nm.

    Returns:
        xarray.Dataset: Dataset containing rayleigh reflectance and optical depth.
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

def upscale_rayleigh(rayleigh_ds: xr.Dataset, l1b_band_path: str) -> xr.DataArray:
    """ Loads a coarse Rayleigh calculation and upscales it to the full image resolution. 

    Args:
        rayleigh_ds (xr.Dataset): Dataset containing coarse rayleigh tie-points
        l1b_band_path (str): Path to a full-res reference band file (NetCDF)

    Returns:
        xr.DataArray: High-res Rayleigh reflectance array
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

def load_geometries(folder_path: str, reference_file: str) -> tuple:
    """Loads and upscales all angular and meteorological ti-points.
    
        Args:
            folder_path (str): Path to OCLI L1B folder
            reference_file (str): Path to full-res reference band (NetCDF)
    
        Returns:
            tuple: 
                - sza (numpy.ndarray): High-res Solar Zenith Angle grid (degrees)
                - oza (numpy.ndarray): High-res Observation Zenith Angle grid (degrees)
                - ozone (numpy.ndarray): High-res Ozone grid
                - u_factor (float): Inverse-squared Earth-Sun distance correction factor
        """
    geom_file = f"{folder_path}/tie_geometries.nc"
    meteo_file = f"{folder_path}/tie_meteo.nc"
    
    sza = upscale_tie_variable(geom_file, 'SZA', reference_file)
    oza = upscale_tie_variable(geom_file, 'OZA', reference_file)
    ozone = upscale_tie_variable(meteo_file, 'total_ozone', reference_file)
    u_factor = get_earth_sun_correction(folder_path)
    
    return sza, oza, ozone, u_factor

def get_cloud_mask(folder_path, sza_high_res, u_factor, brightness_threshold = 0.35, ndvi_cloud_min = -0.05, ndvi_cloud_max = 0.15):
    """Detects clouds using TOA reflectance ratio

    Args:
        folder_path (str): Path to unzipped Sentinel-3 OCLI L1B product folder.
        sza_high_res (numpy.ndarray): Full resolution Solar Zenith Angle grid in degrees.
        u_factor (float): Inverse-squared Earth-Sun distance correction factor
        brightness_threshold (float, optional): Absolute TOA reflectance cutoff for NIR band 17. 

        Returns:
            numpy.ndarray: A boolean 2D array of shape (rows, columns) where True indicates a cloud pixel to mask.
    """

    target_bands = [8, 17]
    band_props = extract_band_properties(folder_path, target_bands, verbose = False)
    
    sza_rad = np.radians(sza_high_res)
    cos_sza = np.clip(np.cos(sza_rad), 1e-5, 1.0)

    rho_bands = {}
    for b in target_bands:
        props = band_props[f'band_{b}']
        e0_adjusted = props['e0'] * u_factor
        
        with xr.open_dataset(f"{folder_path}/{props['file']}") as ds:
            var_name = list(ds.data_vars)[0]
            radiance = ds[var_name].values
            
        rho_bands[b] = (np.pi * radiance) / (e0_adjusted * cos_sza)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Low Cloud Check: NIR brightness
        bright_cloud_mask = rho_bands[17] > brightness_threshold

    init_ndvi = calculate_ndvi(rho_bands[17], rho_bands[8])
    ndvi_mask = (init_ndvi > ndvi_cloud_min) & (init_ndvi < ndvi_cloud_max) & (rho_bands[17] > ndvi_cloud_max)
    cloud_mask = bright_cloud_mask | ndvi_mask

    return cloud_mask

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

def apply_rayleigh_correction(L_toa, high_res_rayleigh, high_res_sza, solar_flux,t_o3):
    """
    Converts Radiance to Reflectance, then subtracts the Rayleigh signal.
    """
    
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

def calculate_ndvi(nir_band,red_band):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi

def extract_band_properties(folder_path, band_numbers, verbose = True):
    """Dynamically extracts the wavelength and solar flux.

    Args:
        folder_path (str): Path to the L1 product folder.
        band_numbers (list): List of integer band numbers to extract
        verbose (bool. optional): If True, prints extracted bands. Defaults True

    Returns:
        dict: Nested dictionary containing 'file', 'wl', and 'e0' for each requested band.
    """

    instrument_file = f"{folder_path}/instrument_data.nc"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = UserWarning)
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

        if verbose:
            print(f"  -> Band {band_num:02d}: Wavelength = {wl:.2f} nm, E0 = {e0:.2f}")
        
    ds_instr.close()
    
    return bands_dict


def generate_ndvi(band_arrays: dict) -> np.ndarray:
    """Extracts required bands and calculates NDVI."""
    red = band_arrays['band_8']
    nir = band_arrays['band_17']
    return calculate_ndvi(nir, red)


def process_surface_reflectance(folder_path: str, bands: dict, L_toa: dict, sza: np.ndarray, oza: np.ndarray, ozone: np.ndarray, u_factor:float) -> dict:
    """Applies Rayleigh and Ozone corrections to target bands.

    Args:
        folder_path (str): Path to OCLI L1B folder
        bands (dict): Dictionary of band properties (wavelength, E0, filename)
        L_toa (dict): Dictionary mapping band names to raw TOA radiance numpy arrays
        sza (numpy.ndarray): High-res Solar Zenith Angle grid (degrees)
        oza (numpy.ndarray): High-res Observation Zenith Angle grid (degrees)
        ozone (numpy.ndarray): High-res Ozone grid
        u_factor (float): Inverse-squared Earth-Sun distance correction factor

    Returns:
        dict: Dictionary mapping band names to their corrected surface reflectance numpy arrays.
    """
    geom_file = f"{folder_path}/tie_geometries.nc"
    meteo_file = f"{folder_path}/tie_meteo.nc"
    
    corrected_arrays = {}
    
    for band_key, props in bands.items():
        radiance_path = f"{folder_path}/{props['file']}"
        adjusted_e0 = props['e0'] * u_factor

        rayleigh_ds = generate_rayleigh(meteo_file, geom_file, props['wl'])
        high_res_rayleigh = upscale_rayleigh(rayleigh_ds, radiance_path)

        band_num = int(band_key.split('_')[1])
        t_o3 = get_o3_trans(sza, oza, ozone, band_num)

        corrected_image = apply_rayleigh_correction(
            L_toa[band_key], high_res_rayleigh, sza, adjusted_e0, t_o3
        )
        corrected_arrays[band_key] = corrected_image.values
        
    return corrected_arrays

def generate_rgb(corrected_arrays: dict, target_bands: list, mode: str, gamma: float) -> np.ndarray:
    """Processes Aerosol subtractions and normalizes RGB bands for visualization.

    Args:
        corrected_arrays (dict): Dict of correct surface reflectance arrays.
        target_bands (list): List of band numbers corresponding to RGB channels
        mode (str): Processing mode ('rgb' or 'aerosol_rgb')
        gamma (float): Gamma correction factor applied to the final image.

    Returns:
        np.ndarray: Normalized and gamma-corrected RGB image array
    """
    if mode == 'aerosol_rgb':
        haze_map = corrected_arrays['band_17']
        water_mask = haze_map < 0.1
        for b in target_bands[:3]:
            corrected_arrays[f'band_{b}'] = np.where(
                water_mask, 
                corrected_arrays[f'band_{b}'] - haze_map, 
                corrected_arrays[f'band_{b}']
            )

    l2_rgb = np.dstack((
        corrected_arrays[f'band_{target_bands[0]}'], 
        corrected_arrays[f'band_{target_bands[1]}'], 
        corrected_arrays[f'band_{target_bands[2]}']
    ))
    
    rgb_min, rgb_max = np.nanpercentile(l2_rgb, 1), np.nanpercentile(l2_rgb, 95)
    l2_normalized = np.clip((l2_rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)
    
    return l2_normalized ** (1.0 / gamma)

def plotting(data: np.ndarray, mode: str, config: dict, output_name: str) -> None:
    """Generates and saves a Matplotlib plot of the final product based on config settings.

    Args:
        data (numpy.ndarray): The processed image array (2D for NDVI, 3D for RGB).
        mode (str): Processing mode ('ndvi', 'rgb', or 'aerosol_rgb').
        config (dict): Configuration dictionary containing visualization settings.
        output_name (str): The filename for the output PNG.
    """

    plt.figure(figsize=(12, 10))

    if mode == 'ndvi':
        cmap = config['visualization'].get('ndvi_cmap', 'RdYlGn')
        plt.imshow(data, cmap=cmap, vmin=-0.2, vmax=0.9)
        plt.colorbar(label="NDVI")
        plt.title("NDVI Map", fontsize=14)
        
    elif mode in ['rgb', 'aerosol_rgb']:
        plt.imshow(data)
        plt.title(f"True Color", fontsize=14)

    plt.axis('off')
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    
    # Check config if the plot should be displayed in a pop-up window
    if config['visualization'].get('show_plot', True):
        plt.show()
    else:
        plt.close()

def export_netcdf(mode: str, target_bands: list[int], corrected_arrays: dict[str, np.ndarray], ndvi_array: np.ndarray | None, output_name: str) -> None:
    """Exports processed data to a NetCDF file.
    Args:
        mode (str): Processing mode
        target_bands (List[int]): List of band n umbers used in final product
        corrected_arrays (dict): Dictionary mapping band names to corrected surface reflectance arrays
        ndvi_array (Any): Calculated NDVI array (None if RGB)
        output_name (str): Destination filename
    """
    if mode == 'ndvi':
        ds_out = xr.Dataset(
            data_vars={"ndvi": (["y", "x"], ndvi_array)},
            attrs={"description": "NDVI", "sensor": "Optical"}
        )
    elif mode in ['rgb', 'aerosol_rgb']:
        ds_out = xr.Dataset(
            data_vars={
                "red": (["y", "x"], corrected_arrays[f'band_{target_bands[0]}']),
                "green": (["y", "x"], corrected_arrays[f'band_{target_bands[1]}']),
                "blue": (["y", "x"], corrected_arrays[f'band_{target_bands[2]}'])
            },
            attrs={"description": f"Composite ({mode})", "sensor": "Optical"}
        )
        
    ds_out.to_netcdf(output_name)

def main():
    # Load Configuration
    config = load_config("config.yaml")
    folder_path = config['input']['folder_path']
    mode = config['processing']['mode']
    gamma = config['visualization']['gamma']
    cloud_config = config.get('processing', {}).get('cloud_masking', {})
    
    print(f"Processing in '{mode.upper()}' mode.")

    # Determine Bands based on Mode
    if mode in ['rgb', 'aerosol_rgb']:
        target_bands = list(config['processing']['rgb_bands'])
        
        # If aerosol correction is on, append Near-Infrared band
        if mode == 'aerosol_rgb' and 17 not in target_bands:
            target_bands.append(17) 
            
    elif mode == 'ndvi':
        target_bands = [8, 17]
    else:
        raise ValueError(f"Unknown mode: {mode}. Check your config.yaml.")
                                                         
    # Setup & Geometry
    bands = extract_band_properties(folder_path, target_bands)
    reference_file = f"{folder_path}/{bands[f'band_{target_bands[-1]}']['file']}"
    sza_high_res, oza_high_res, ozone_high_res, u_factor = load_geometries(folder_path,reference_file)

    L_toa = load_radiance(folder_path, bands,sza_high_res, u_factor, cloud_config)

    # Generate Rayleigh corrections
    corrected_arrays = process_surface_reflectance(folder_path, bands, L_toa, sza_high_res, oza_high_res, ozone_high_res, u_factor)

    # Processing and Export
    nc_output_name = f"Processed_Data_{mode}.nc"
    if mode == 'ndvi':
        output_data = generate_ndvi(corrected_arrays)
        export_netcdf(mode, target_bands, corrected_arrays, output_data, nc_output_name)

    elif mode in ['rgb', 'aerosol_rgb']:
        output_data = generate_rgb(corrected_arrays, target_bands, mode, gamma)
        export_netcdf(mode, target_bands, corrected_arrays, None, nc_output_name)   
        
    print(f"Data saved to {nc_output_name}") 
    
    #Visualization
    if config.get('visualization', {}).get('generate_plot', True):
        png_output_name = f"Output_{mode}.png"
        plotting(output_data, mode, config, png_output_name)

if __name__ == "__main__":
    main()