"""Legacy raster runtime setup, isolated from the API process."""
import os
from pathlib import Path


def configure_raster_runtime():
    # Match the Flask application's bundled-PROJ fix. On Windows, PostGIS may
    # otherwise inject an incompatible proj.db into Rasterio's environment.
    import rasterio
    from rasterio.env import set_proj_data_search_path

    bundled = Path(rasterio.__file__).resolve().parent / "proj_data"
    if (bundled / "proj.db").is_file():
        os.environ["PROJ_LIB"] = str(bundled)
        os.environ["PROJ_DATA"] = str(bundled)
        set_proj_data_search_path(str(bundled))
