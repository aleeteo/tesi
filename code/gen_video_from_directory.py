import json
import os
import random
import cv2
import numpy as np
import get_metadata
import artificial_video
import my_utils as u

lin2srgb = u.lin2srgb
get_centroids = get_metadata.get_centroids


def gen_video_from_directory(input_dir: str) -> list:
    # controllo l'esistenza della directory e del file meta.json
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"La directory specificata non esiste: {input_dir}")

    meta_path = os.path.join(input_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"File meta.json non trovato in: {input_dir}")

    # definizione sensor_depth base al tipo di telecamera
    depth: int = 10 if "galaxy" in input_dir.lower() else 14

    # caricemento del file meta.json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # -------------------------------------------------------
    for place in meta_data.keys():
        # -------------------------------------------------------
        # estrazione del nome dell'immagine e della GT
        # con il massimo numero di illuminanti accesi
        num_lights = meta_data[place]["NumOfLights"]
        lights_str = "".join(
            str(i + 1) for i in range(num_lights)
        )  # es. "12", "123", ecc.
        tiff_path = os.path.join(input_dir, place, f"{place}_{lights_str}.tiff")
        gt_path = os.path.join(input_dir, place, f"{place}_{lights_str}_gt.tiff")

        # -------------------------------------------------------
        # caricamento in memotia delle immagini
        img = u.imread(tiff_path, depth)
        gt = u.imread(gt_path, depth)

        # impostazione del seed per la riproducibilità e altri parametri del video
        seed = random.randint(0, 10000)
        zoom = 3
        fps = 15
        video_duration = 3  # durata del video in secondi
        # recupero i centroidi dei colorChecker
        centroids = get_centroids(meta_data[place])

        # TODO: applicare le maschere ai video
        # TODO: concatenare i video in un unico video

        video_frames = artificial_video.ken_burns_advanced(
            img,
            start_point=centroids[0],
            end_point=centroids[1],
            start_zoom=zoom,
            end_zoom=zoom,
            interp_pan="linear",
            interp_zoom="ease_in_out",
            noise_strength=0.6,
            noise_speed=0.2,
            noise_seed=seed,
            video_fps=fps,
            video_duration=video_duration,
        )

        gt_video_frames = artificial_video.ken_burns_advanced(
            gt,
            start_point=centroids[0],
            end_point=centroids[1],
            start_zoom=zoom,
            end_zoom=zoom,
            interp_pan="linear",
            interp_zoom="ease_in_out",
            noise_strength=0.5,
            noise_speed=0.2,
            noise_seed=seed,
            video_fps=fps,
            video_duration=video_duration,
        )

        artificial_video.play_video_from_array(video_frames, fps=fps)
        artificial_video.play_video_from_array(gt_video_frames, fps=fps)

    return []


if __name__ == "__main__":
    input_dir = "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon_processed/"
    gen_video_from_directory(input_dir)
