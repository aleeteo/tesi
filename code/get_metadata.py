import json
import numpy as np
from typing import Tuple


def get_centroids(meta: dict) -> np.ndarray:
    mcc_coords = meta["MCCCoord"]
    # Calcolo della media e divisione per 2
    return np.array([np.mean(coords, axis=0) for coords in mcc_coords.values()]) / 2


def get_centroids_pairs(meta: dict) -> Tuple[list, np.ndarray]:
    """
    Restituisce una lista di coppie di centri dei colorChecker.
    """
    centroids = get_centroids(meta)
    n = len(centroids)
    pairs = [(i, (i + 1) % n) for i in range(n)]
    return (pairs, centroids)


if __name__ == "__main__":
    with open(
        "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon/meta.json", "r"
    ) as f:
        data = json.load(f)
    meta = data["Place0"]

    centroids = get_centroids(meta)
    print("Centroids:\n", centroids)
