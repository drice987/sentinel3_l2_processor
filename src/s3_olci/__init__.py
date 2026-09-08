from importlib.metadata import PackageNotFoundError, version

from .atmospheric import correct_surface_reflectance, load_and_upscale_geometries
from .masking import compute_cloud_mask
from .products import compute_ndvi, generate_rgb_composite

try:
    __version__ = version("s3-olci-processor")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "compute_cloud_mask",
    "compute_ndvi",
    "correct_surface_reflectance",
    "generate_rgb_composite",
    "load_and_upscale_geometries",
]
