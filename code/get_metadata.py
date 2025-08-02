import json


def get_centroids(image_name: str, json_file_path: str = "sony_meta.json"):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    centroids = []

    for mcc, coords in data[image_name]["MCCCoord"].items():
        # Separiamo le coordinate x e y
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]

        # Calcoliamo il centroide come media aritmetica
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        centroids.append((mcc, (cx, cy)))

    for mcc, centroide in centroids:
        print(f"{mcc}: centroide = {centroide}")


if __name__ == "__main__":
    get_centroids("Place925.jpg", "sony_meta.json")
