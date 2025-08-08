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

    results = []

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

        # TODO: trasformare tutta sta roba in parametri
        #
        # impostazione del seed per la riproducibilità e altri parametri del video
        seed = random.randint(0, 10000)
        zoom = 3
        nframes = 15
        # TODO: applicare le maschere ai video

        # recupero i centroidi dei colorChecker
        centroids = get_centroids(meta_data[place])
        n = len(centroids)
        if n < 2:
            # niente da fare: un solo centroide non consente transizioni
            continue

        # coppie consecutive con wrap-around: (0->1, 1->2, ..., n-1->0)
        pairs = [(i, (i + 1) % n) for i in range(n)]

        # genera segmenti e accumula
        segments = []
        gt_segments = []

        for a, b in pairs:
            seg = artificial_video.ken_burns_advanced(
                img,
                start_point=centroids[a],
                end_point=centroids[b],
                start_zoom=zoom,
                end_zoom=zoom,
                interp_pan="linear",
                interp_zoom="ease_in_out",
                noise_strength=0.9,
                noise_speed=0.2,
                noise_seed=seed,
                num_frames=nframes,
            )
            segments.append(seg)

            gt_seg = artificial_video.ken_burns_advanced(
                gt,
                start_point=centroids[a],
                end_point=centroids[b],
                start_zoom=zoom,
                end_zoom=zoom,
                interp_pan="linear",
                interp_zoom="ease_in_out",
                noise_strength=0.9,
                noise_speed=0.2,
                noise_seed=seed,
                num_frames=nframes,
            )
            gt_segments.append(gt_seg)

        # concatena tutti i segmenti in un unico video (asse temporale)
        video_frames = np.concatenate(segments, axis=0)
        gt_video_frames = np.concatenate(gt_segments, axis=0)

        # playback (facoltativo)
        artificial_video.play_video_from_array(video_frames, fps=20)
        artificial_video.play_video_from_array(gt_video_frames, fps=20)

        # salva nel risultato
        results.append(
            {
                "key": place,
                "video": video_frames,
                "gt_video": gt_video_frames,
            }
        )

    return []


if __name__ == "__main__":
    input_dir = "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon_processed/"
    gen_video_from_directory(input_dir)
