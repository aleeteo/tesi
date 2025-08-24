import json
import os
import random
from typing import List, Tuple

import artificial_video
import cv2
import get_metadata as m
import my_utils as u
import numpy as np


def load_json(input_dir: str) -> dict:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"La directory specificata non esiste: {input_dir}")
    meta_path = os.path.join(input_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"File meta.json non trovato in: {input_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _segment_seeds(pairs: List[Tuple[int, int]], seed: int | None) -> list[int]:
    """
    Genera un seed distinto per ciascun segmento, stabile e riproducibile.
    Usa SeedSequence.spawn per ottenere child indipendenti.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(pairs))
    # un intero per segmento (32 bit bastano per il tuo uso)
    seg_seeds = [
        np.random.default_rng(c).integers(0, 2**31 - 1, dtype=np.uint32).item()
        for c in children
    ]
    return seg_seeds


def gen_video_from_directory(
    input_dir: str,
    zoom: float = 2,
    totalFrames: int = 60,
    write: bool = False,
    playback: bool = False,
    output_dir: str | None = None,
    *,
    seed: int | None = None,  # opzionale: per riproducibilità globale
    save_format: str = "tiff",  # "tiff" | "npy"
) -> list:
    """
    Genera video (og/gt) per ciascun 'place'.
    - Assume immagini/stack RGB float32 in [0,1] (o convertibili a 3 canali in utils).
    - Se write=True, salva per-place durante il loop (nessun batch finale).
    """
    meta = load_json(input_dir)
    depth: int = 10 if "galaxy" in input_dir.lower() else 14

    if output_dir is None:
        output_dir = os.path.join(input_dir, "output_vids")

    results = []

    for place in meta.keys():
        # ----------------------------
        # path
        num_lights = meta[place]["NumOfLights"]
        lights_str = "".join(str(i + 1) for i in range(num_lights))
        img_path = os.path.join(input_dir, place, f"{place}_{lights_str}.tiff")
        gt_path = os.path.join(input_dir, place, f"{place}_{lights_str}_gt.tiff")
        mask_path = os.path.join(input_dir, place, f"{place}_mask.png")

        # ----------------------------
        # load immagini (float32 [0,1])
        img = u.imread(img_path, depth)  # normalizza su sensor_depth
        gt = u.imread(gt_path, depth)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.ones(shape=img.shape[:2], dtype=np.float32)
        else:
            mask = mask.astype(np.float32)
            if mask.max() > 1.0:
                mask /= 255.0

        # applica maschera (broadcast)
        img = img * mask[:, :, None]
        gt = gt * mask[:, :, None]

        # ----------------------------
        # coppie centroidi e semi per segmento
        pairs, centroids = m.get_centroids_pairs(meta[place])
        pairs = list(pairs)  # garantisci indice deterministico
        nframes = max(1, totalFrames // max(1, len(pairs)))
        seg_seeds = _segment_seeds(pairs, seed)

        segments = []
        gt_segments = []

        for (a, b), seg_seed in zip(pairs, seg_seeds):
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
                noise_seed=int(seg_seed),  # seed unico PER SEGMENTO
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
                noise_seed=int(seg_seed),  # stesso seed -> stesso movimento/rumore
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

        # ----------------------------
        # SALVATAGGIO PER-PLACE (on the fly)
        if write:
            u.save_place_videos(result, output_dir, format=save_format)  # type: ignore

    return results


if __name__ == "__main__":
    input_dir = "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon_processed/"
    results = gen_video_from_directory(
        input_dir,
        zoom=3,
        totalFrames=60,
        write=True,  # salva per-place, nel loop
        playback=False,
        # output_dir=os.path.join(input_dir, "output_vids"),
        output_dir="output_vids/",
        seed=123456789,  # opzionale (riproducibile)
        save_format="npy",  # o "tiff"
    )
