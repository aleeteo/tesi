import json
import numpy as np


def get_centroids(meta: dict) -> np.ndarray:
    mcc_coords = meta["MCCCoord"]
    # Convertiamo ogni lista di coordinate in un array numpy e calcoliamo la media lungo l'asse 0 (x e y)
    return np.array([np.mean(coords, axis=0) for coords in mcc_coords.values()])


if __name__ == "__main__":
    with open(
        "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon/meta.json", "r"
    ) as f:
        data = json.load(f)
    meta = data["Place0"]

    centroids = get_centroids(meta)
    print("Centroids:\n", centroids)
