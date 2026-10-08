"""Environment data: local grid, population, buildings.

Synthetic environments need only the core install. Real-world loaders live in effects.data.geo
and need the [geo] extra; downloads are cached in cache/ so reruns are offline and repeatable.
"""

from effects.data.environment import BUILDING_COLUMNS, Environment, Grid
from effects.data.synthetic import synthetic_city

__all__ = ["Environment", "Grid", "BUILDING_COLUMNS", "synthetic_city"]
