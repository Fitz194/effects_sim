"""Real-world loaders (optional extra: pip install -e ".[geo]").

- Population: any gridded-count GeoTIFF (WorldPop, GHSL GHS-POP), reprojected onto the local grid
  with sum-preserving resampling.
- Buildings: OpenStreetMap footprints via osmnx, cached as GeoPackage in cache/.

Both are projected to the UTM zone of ground zero, then shifted so ground zero is (0, 0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from effects.data.environment import BUILDING_COLUMNS, Environment, Grid

CACHE_DIR = Path("cache")

# OSM building=* tag -> fragility class. Edit to suit the region (UK housing is mostly masonry).
OSM_CLASS_MAP = {
    "house": "masonry",
    "detached": "masonry",
    "semidetached_house": "masonry",
    "terrace": "masonry",
    "residential": "masonry",
    "bungalow": "masonry",
    "apartments": "reinforced_concrete",
    "office": "reinforced_concrete",
    "hospital": "reinforced_concrete",
    "school": "masonry",
    "university": "reinforced_concrete",
    "hotel": "reinforced_concrete",
    "commercial": "steel_frame",
    "retail": "steel_frame",
    "supermarket": "steel_frame",
    "industrial": "light_industrial",
    "warehouse": "light_industrial",
    "shed": "wood_frame",
    "garage": "light_industrial",
    "garages": "light_industrial",
    "hut": "wood_frame",
    "cabin": "wood_frame",
    "church": "masonry",
}
DEFAULT_OSM_CLASS = "masonry"
METRES_PER_LEVEL = 3.0


def _utm_transformer(lat: float, lon: float):
    from pyproj import CRS, Transformer

    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    crs = CRS.from_epsg(epsg)
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = to_utm.transform(lon, lat)
    return crs, x0, y0


def load_population_raster(path: str | Path, lat: float, lon: float, grid: Grid) -> np.ndarray:
    """Reproject a population-count raster onto the local grid. Total count is preserved."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    crs, x0, y0 = _utm_transformer(lat, lon)
    e = grid.n * grid.cell_m / 2
    dst_transform = from_origin(x0 - e, y0 + e, grid.cell_m, grid.cell_m)
    dst = np.zeros((grid.n, grid.n), dtype=np.float64)
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        if src.nodata is not None:
            data[data == src.nodata] = 0.0
        data[~np.isfinite(data) | (data < 0)] = 0.0
        reproject(
            source=data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=Resampling.sum,
        )
    # Raster rows run north->south; local grid rows run south->north.
    return np.flipud(dst)


def load_osm_buildings(
    lat: float, lon: float, radius_m: float, refresh: bool = False
) -> pd.DataFrame:
    """Fetch (or load cached) OSM building footprints as a local-metre building table."""
    import geopandas as gpd

    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"osm_buildings_{lat:.5f}_{lon:.5f}_{int(radius_m)}.gpkg"
    if cache.exists() and not refresh:
        gdf = gpd.read_file(cache)
    else:
        import osmnx as ox

        gdf = ox.features_from_point((lat, lon), tags={"building": True}, dist=radius_m)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        keep = [c for c in ("building", "building:levels", "height") if c in gdf.columns]
        gdf = gdf[keep + ["geometry"]].reset_index(drop=True)
        for c in keep:
            gdf[c] = gdf[c].astype(str)
        gdf.to_file(cache, driver="GPKG")

    crs, x0, y0 = _utm_transformer(lat, lon)
    gdf = gdf.to_crs(crs)
    cent = gdf.geometry.centroid
    tag = gdf.get("building", pd.Series(["yes"] * len(gdf))).astype(str)
    cls = tag.map(OSM_CLASS_MAP).fillna(DEFAULT_OSM_CLASS)
    height = _height(gdf)
    return pd.DataFrame(
        {
            "x": cent.x.to_numpy() - x0,
            "y": cent.y.to_numpy() - y0,
            "cls": cls.to_numpy(),
            "footprint_m2": gdf.geometry.area.to_numpy(),
            "height_m": height,
        }
    )[BUILDING_COLUMNS]


def _height(gdf) -> np.ndarray:
    n = len(gdf)
    h = np.full(n, np.nan)
    if "height" in gdf.columns:
        h = pd.to_numeric(
            gdf["height"].str.replace("m", "", regex=False), errors="coerce"
        ).to_numpy()
    if "building:levels" in gdf.columns:
        lv = pd.to_numeric(gdf["building:levels"], errors="coerce").to_numpy() * METRES_PER_LEVEL
        h = np.where(np.isnan(h), lv, h)
    return np.where(np.isnan(h), 2 * METRES_PER_LEVEL, h)


def real_environment(
    lat: float,
    lon: float,
    grid: Grid,
    class_names: tuple[str, ...],
    default_mix: np.ndarray,
    population_raster: str | None,
    use_osm_buildings: bool = True,
) -> Environment:
    pop = (
        load_population_raster(population_raster, lat, lon, grid)
        if population_raster
        else np.zeros((grid.n, grid.n))
    )
    if use_osm_buildings:
        b = load_osm_buildings(lat, lon, grid.half_width_m * np.sqrt(2))
        _, _, ok = grid.cell_index(b["x"].to_numpy(), b["y"].to_numpy())
        b = b[ok].reset_index(drop=True)
    else:
        b = pd.DataFrame(columns=BUILDING_COLUMNS)
    return Environment(
        grid=grid,
        population=pop,
        buildings=b,
        class_names=tuple(class_names),
        default_mix=np.asarray(default_mix, dtype=float),
        meta={"kind": "real", "lat": lat, "lon": lon, "population_raster": population_raster},
    )
