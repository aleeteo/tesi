import json


def get_centroids(meta: dict) -> list:
    centroids = []

    for mcc, coords in meta["MCCCoord"].items():
        # Separiamo le coordinate x e y
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]

        # Calcoliamo il centroide come media aritmetica
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        centroids.append((cx, cy))

    return centroids


if __name__ == "__main__":
    with open(
        "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon/meta.json", "r"
    ) as f:
        data = json.load(f)
    meta = data["Place0"]

    gen = get_centroids(meta)
    print("Centroids:", gen)
