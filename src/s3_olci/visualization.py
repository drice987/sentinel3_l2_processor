from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_plot(data: np.ndarray,mode: str,output_path: Path,cmap: str = "RdYlGn",show_plot: bool = False,) -> None:
    """Save the output array as a high-resolution PNG image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))

    if mode == "ndvi":
        plt.imshow(data, cmap=cmap, vmin=-0.2, vmax=0.9)
        plt.colorbar(label="Surface NDVI")
        plt.title("Sentinel-3 OLCI Surface NDVI", fontsize=14)
    elif mode in ["rgb", "aerosol_rgb"]:
        plt.imshow(data)
        plt.title(f"True Color Surface Composite ({mode})", fontsize=14)

    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()
