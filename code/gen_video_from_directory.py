import json
import os
import random
from typing import List
import cv2
import numpy as np
import get_metadata as m
import artificial_video
import my_utils as u

lin2srgb = u.lin2srgb


def load_json(input_dir: str) -> dict:
    """
    Carica il file meta.json dalla directory specificata.
    """
    # controllo l'esistenza della directory e del file meta.json
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"La directory specificata non esiste: {input_dir}")

    meta_path = os.path.join(input_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"File meta.json non trovato in: {input_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def gen_video_from_directory(
    input_dir: str,
    zoom: float = 2,
    totalFrames: int = 60,
    write: bool = False,
    playback: bool = False,
    output_dir: str | None = None,  # <— nuovo
) -> list:
    """
    Si occupa di generare un video da una directory contenente immagini e metadati.
    """

    # caricamento del dict dal file json dei metadati
    meta = load_json(input_dir)

    # definizione sensor_depth base al tipo di telecamera
    depth: int = 10 if "galaxy" in input_dir.lower() else 14

    # cartella di output di default: dentro input_dir (scrivibile)
    if output_dir is None:
        output_dir = os.path.join(input_dir, "output_vids")

    results = []

    for place in meta.keys():
        # -------------------------------------------------------
        # estrazione dei path
        num_lights = meta[place]["NumOfLights"]
        lights_str = "".join(str(i + 1) for i in range(num_lights))
        img_path = os.path.join(input_dir, place, f"{place}_{lights_str}.tiff")
        gt_path = os.path.join(input_dir, place, f"{place}_{lights_str}_gt.tiff")
        mask_path = os.path.join(input_dir, place, f"{place}_mask.png")

        # -------------------------------------------------------
        # caricamento immagini
        img = u.imread(img_path, depth)
        gt = u.imread(gt_path, depth)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.ones(shape=img.shape[:2], dtype=np.float32)
        else:
            mask = mask.astype(np.float32)
            # normalizza se arriva 0–255
            if mask.max() > 1.0:
                mask /= 255.0

        # applica maschera (broadcast su canali)
        img = img * mask[:, :, None]
        gt = gt * mask[:, :, None]

        (pairs, centroids) = m.get_centroids_pairs(meta[place])

        # Parametri generazione video
        seed = random.randint(0, 10**16)
        nframes = max(1, totalFrames // max(1, len(pairs)))  # safety

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

        og_video_frames = np.concatenate(segments, axis=0)
        gt_video_frames = np.concatenate(gt_segments, axis=0)

        if playback:
            artificial_video.play_video_from_array(og_video_frames, fps=20)
            artificial_video.play_video_from_array(gt_video_frames, fps=20)

        result = {
            "key": place,
            "og_video": og_video_frames,
            "gt_video": gt_video_frames,
        }
        results.append(result)

        if write:
            # scrivi solo il corrente, per evitare riscritture ripetute
            write_videos_as_png_single(result, output_dir)

    return results


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_videos_as_png_single(result: dict, output_root: str):
    """
    Scrive i PNG per un singolo 'place'.
    Crea automaticamente: <output_root>/<place>/{og,gt}
    """
    key = result["key"]
    og_video = result["og_video"]
    gt_video = result["gt_video"]

    place_dir = os.path.join(output_root, key)
    og_dir = os.path.join(place_dir, "og")
    gt_dir = os.path.join(place_dir, "gt")

    _ensure_dir(og_dir)
    _ensure_dir(gt_dir)

    num_frames = og_video.shape[0]
    for i in range(num_frames):
        u.imwrite(os.path.join(og_dir, f"{i:04d}"), og_video[i])
        u.imwrite(os.path.join(gt_dir, f"{i:04d}"), gt_video[i])


def write_videos_as_png(results: List[dict], output_root: str):
    """
    Versione batch (se vuoi scrivere tutto alla fine).
    """
    for res in results:
        write_videos_as_png_single(res, output_root)


if __name__ == "__main__":
    input_dir = "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon_processed/"
    results = gen_video_from_directory(
        input_dir,
        zoom=3,
        totalFrames=60,
        write=True,
        playback=False,
        # output_dir=os.path.join(input_dir, "output_vids"),  # <— scrivibile
        output_dir="output_vids/",
    )
