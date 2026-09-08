from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import __version__
from .atmospheric import (
    correct_surface_reflectance,
    load_and_upscale_geometries,
)
from .io import (
    export_netcdf,
    extract_band_properties,
    load_config,
    load_radiance_bands,
)
from .masking import compute_cloud_mask
from .products import compute_ndvi, generate_rgb_composite
from .visualization import save_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("s3_olci")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sentinel-3 OLCI Level 1B Surface Reflectance Processor"
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    return parser.parse_args()


def run_pipeline(config_path: Path) -> None:
    """Execute the full Level 1B processing pipeline."""
    config = load_config(config_path)
    folder_path = Path(config["input"]["folder_path"])
    mode = config["processing"]["mode"].lower()
    gamma = config["visualization"].get("gamma", 1.0)
    cloud_cfg = config.get("processing", {}).get("cloud_masking", {})

    logger.info("Processing scene: %s (mode: %s)", folder_path.name, mode.upper())

    # Determine required bands based on operational mode
    if mode in ["rgb", "aerosol_rgb"]:
        target_bands = list(config["processing"]["rgb_bands"])
        if mode == "aerosol_rgb" and 17 not in target_bands:
            target_bands.append(17)
    elif mode == "ndvi":
        target_bands = [8, 17]
    else:
        raise ValueError(
            f"Unsupported processing mode: '{mode}'. Check your configuration."
        )

    bands = extract_band_properties(folder_path, target_bands)
    ref_band_path = folder_path / bands[f"band_{target_bands[-1]}"]["file"]
    sza, oza, ozone, u_factor = load_and_upscale_geometries(folder_path, ref_band_path)

    cloud_mask = None
    if cloud_cfg.get("enabled", False):
        cloud_mask = compute_cloud_mask(
            folder_path=folder_path,
            sza=sza,
            u_factor=u_factor,
            brightness_threshold=cloud_cfg.get("brightness_threshold", 0.35),
            ndvi_cloud_min=cloud_cfg.get("ndvi_cloud_min", -0.05),
            ndvi_cloud_max=cloud_cfg.get("ndvi_cloud_max", 0.15),
        )

    l_toa = load_radiance_bands(folder_path, bands, cloud_mask=cloud_mask)
    corrected_refl = correct_surface_reflectance(
        folder_path, bands, l_toa, sza, oza, ozone, u_factor
    )

    output_dir = Path(config.get("output", {}).get("directory", "outputs"))
    output_nc = output_dir / f"Processed_Data_{mode}.nc"

    if mode == "ndvi":
        product_data = compute_ndvi(corrected_refl["band_17"], corrected_refl["band_8"])
        export_netcdf(
            mode, output_nc, target_bands, corrected_refl, ndvi_array=product_data
        )
    else:
        product_data = generate_rgb_composite(
            corrected_refl, target_bands, mode=mode, gamma=gamma
        )
        export_netcdf(mode, output_nc, target_bands, corrected_refl, ndvi_array=None)

    logger.info("Exported NetCDF to %s", output_nc)

    if config.get("visualization", {}).get("generate_plot", True):
        output_png = output_dir / f"Output_{mode}.png"
        save_plot(
            product_data,
            mode=mode,
            output_path=output_png,
            cmap=config.get("visualization", {}).get("ndvi_cmap", "RdYlGn"),
            show_plot=config.get("visualization", {}).get("show_plot", False),
        )


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
