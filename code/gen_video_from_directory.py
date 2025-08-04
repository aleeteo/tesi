import json, os, random, cv2
import numpy as np
import get_metadata, artificial_video, tiff_visualization

lin2srgb = tiff_visualization.linear_tiff_to_srgb


def gen_video_from_directory(input_dir: str) -> list:
    # controllo l'esistenza della directory e del file meta.json
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"La directory specificata non esiste: {input_dir}")

    meta_path = os.path.join(input_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"File meta.json non trovato in: {input_dir}")

    # caricemento del file meta.json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # -------------------------------------------------------
    key = list(meta_data.keys())[0]  # prendo il primo key del meta_data
    # TODO: da mettere nel loop

    # -------------------------------------------------------
    # estrazione del nome dell'immagine e della GT
    # con il massimo numero di illuminanti accesi
    num_lights = meta_data[key]["NumOfLights"]
    lights_str = "".join(str(i + 1) for i in range(num_lights))  # es. "12", "123", ecc.
    tiff_path = os.path.join(input_dir, key, f"{key}_{lights_str}.tiff")
    gt_path = os.path.join(input_dir, key, f"{key}_{lights_str}_gt.tiff")

    # -------------------------------------------------------
    # caricamento in memotia delle immagini
    img = cv2.imread(tiff_path)
    gt = cv2.imread(gt_path)

    # impostazione del seed per la riproducibilità
    seed = random.randint(0, 10000)

    # TODO: generazione del video

    return []


if __name__ == "__main__":
    input_dir = (
        "/Users/alessandroteodori/Documents/stage/code/LSMI-dataset/nikon_processed/"
    )
    gen_video_from_directory(input_dir)
