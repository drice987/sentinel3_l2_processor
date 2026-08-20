# Sentinel-3 Level-2 Optical Processor

## Overview
A lightweight Earth Observation data pipeline that processes Sentinel-3 OLCI Top-of-Atmosphere radiance with standard Python packages into Level-2 surface reflectance, aerosol-corrected composites, and Normalized Difference Vegetation Index (NDVI). 

## Example Outputs

<p align="center">
  <img src="truecolor_ndvi.png" width="100%" alt="True Color and NDVI Comparison">
</p>
Showing true color with aerosol correction (left) and the NDVI visualization (right).

## Key Features & Physics Implemented
* **Atmospheric Scattering (Rayleigh):** Calculates real-time Rayleigh optical thickness using the Single-Scattering Approximation (SSA)
* **Orbital Normalization:** Dynamically calculates the Julian day from satellite metadata to apply Earth-Sun inverse-square solar flux corrections.
* **Spatial Interpolation:** Upscales coarse tie-point geometry and meteorological grids to full sensor resolution using bivariate spline interpolation 
* **Aerosol Correction:** Implements a dynamic Near-Infrared (NIR) Dark Object Subtraction (DOS) for quick corrections.
* **Ozone Absorption Correction** Uses ozone data to compute and apply gaseous transmittance corrections
* **Cloud Masking:** Automated masking module to detect clouds and mask pixels using configurable NDVI and near-infrared brightness thresholds.
* **.nc Export:** Outputs both  `.png` composites and analysis-ready `.nc` (NetCDF4) datasets preserving full precision.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone git@github.com:drice987/sentinel-level2-processor.git
cd sentinel-level2-processor
pip install -r requirements.txt
```

## Usage

**1. Configure the Pipeline**
Edit the `config.yaml` file to point to your unzipped Sentinel-3 L1B folder and select your processing mode.

```yaml
input:
  folder_path: "path/to/SEN3"

processing:
  mode: "rgb"  # Options: 'raw_rgb', 'rgb', 'aerosol_rgb', 'ndvi'
  rgb_bands: [7, 6, 4] # Corresponds to OLCI 620nm, 560nm, 490nm
  
  cloud_masking:
    enabled: true
    brightness_threshold: 0.5
    ndvi_cloud_min: -0.05
    ndvi_cloud_max: 0.15

visualization:
  generate_plot: true
  gamma: 1.5
  ndvi_cmap: "RdYlGn"
```

**2. Run the Processor**
Execute the master script from the terminal:
```bash
python sentinel3_l2_processor.py
```

## Processing Modes
* `rgb`: Applies Rayleigh scattering correction based on local solar/viewing geometries and atmospheric pressure.
* `aerosol_rgb`: Applies Rayleigh correction followed by a dynamic NIR Dark Object Subtraction to strip aerosol haze over water bodies.
* `ndvi`: Calculates the Normalized Difference Vegetation Index using Rayleigh-corrected Red and Near-Infrared bands.


---
*Developed for Earth Observation data engineering and optical physics applications.*