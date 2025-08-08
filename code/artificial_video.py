import cv2
import numpy as np
import math
from typing import Optional
from perlin_noise import PerlinNoise

import my_utils as u


# =========================
# UTILITÀ DI SALVATAGGIO
# =========================
def save_video(frames: np.ndarray, output_path: str, fps: int, format: str = "mp4"):
    """
    Salva un video come file MJPEG, MP4 o NPY.
    """
    if frames.ndim != 4:
        raise ValueError("Il video deve essere un array 4D di forma (N, H, W, C).")

    if format == "npy":
        if not output_path.endswith(".npy"):
            output_path += ".npy"
        np.save(output_path, frames)
        print(f"✅ Video salvato come array NumPy: {output_path}")
        return

    n_frames, h, w, _ = frames.shape
    if format == "mjpeg":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore
        if not output_path.endswith(".avi"):
            output_path += ".avi"
    elif format == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        if not output_path.endswith(".mp4"):
            output_path += ".mp4"
    else:
        raise ValueError("Formato non supportato. Usa 'mjpeg', 'mp4' o 'npy'.")

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for i in range(n_frames):
        out.write(frames[i])
    out.release()
    print(f"✅ Video salvato: {output_path}")


def load_video_npy(path: str) -> np.ndarray:
    """
    Carica un video salvato come array NumPy (NPY).
    """
    return np.load(path)


def play_video_from_array(frames: np.ndarray, fps: int = 30):
    """Riproduce un video da un array 4D (N, H, W, C)."""
    frames_srgb = u.lin2srgb(frames)
    for frame in frames_srgb:
        cv2.imshow("Video", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# =========================
# FUNZIONI DI GENERAZIONE
# =========================
def ken_burns_simple(
    image: np.ndarray,
    video_fps: int = 30,
    video_duration: int = 5,
    start_point: tuple = (0.5, 0.5),
    end_point: tuple = (0.5, 0.5),
    start_zoom: float = 1.0,
    end_zoom: float = 1.2,
    video_size: tuple = (1280, 720),
) -> np.ndarray:
    """Effetto Ken Burns semplice: pan lineare + zoom lineare. Restituisce (N, H, W, C)."""
    H, W, _ = image.shape
    total_frames = int(video_fps * video_duration)
    frames = np.zeros((total_frames, video_size[1], video_size[0], 3), dtype=np.uint8)

    for i in range(total_frames):
        t = i / (total_frames - 1)
        cx = (start_point[0] + t * (end_point[0] - start_point[0])) * W
        cy = (start_point[1] + t * (end_point[1] - start_point[1])) * H
        zoom = start_zoom + t * (end_zoom - start_zoom)

        crop_w = int(W / zoom)
        crop_h = int(H / zoom)
        x_offset = max(0, min(int(cx - crop_w / 2), W - crop_w))
        y_offset = max(0, min(int(cy - crop_h / 2), H - crop_h))

        crop = image[y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
        frames[i] = cv2.resize(crop, video_size)

    return frames


def ken_burns_advanced(
    image: np.ndarray,
    num_frames: int = 90,  # nuovo: numero di frame totali
    start_point: np.ndarray = np.array([0.5, 0.5]),
    end_point: np.ndarray = np.array([0.5, 0.5]),
    start_zoom: float = 1.2,
    end_zoom: float = 1.2,
    video_size: tuple = (1280, 720),
    interp_pan: str = "linear",
    interp_zoom: str = "ease_in_out",
    noise_strength: float = 0.0,
    noise_speed: float = 0.5,
    noise_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Effetto Ken Burns avanzato con numero di frame fisso.
    Restituisce un array (N, H, W, C).
    start_point e end_point possono essere in coordinate normalizzate (0-1) o assolute (pixel).
    """

    def linear(t):
        return t

    def ease_in_out(t):
        return (1 - math.cos(math.pi * t)) / 2

    def logarithmic(t):
        return math.log1p(t * 9) / math.log(10)

    def exponential(t):
        return (math.exp(t) - 1) / (math.e - 1)

    interp_map = {
        "linear": linear,
        "ease_in_out": ease_in_out,
        "log": logarithmic,
        "exp": exponential,
    }
    pan_fn = interp_map.get(interp_pan, linear)
    zoom_fn = interp_map.get(interp_zoom, ease_in_out)

    H, W, _ = image.shape

    def normalize_point(p: np.ndarray) -> np.ndarray:
        if p[0] > 1 or p[1] > 1:
            return np.array([p[0] / W, p[1] / H], dtype=np.float32)
        return p.astype(np.float32)

    start_point = normalize_point(start_point)
    end_point = normalize_point(end_point)

    noise_gen_x = PerlinNoise(
        octaves=1,
        seed=noise_seed if noise_seed is not None else np.random.randint(0, 10000),
    )
    noise_gen_y = PerlinNoise(
        octaves=1,
        seed=(noise_seed + 1000)
        if noise_seed is not None
        else np.random.randint(0, 10000),
    )

    frames = np.zeros((num_frames, video_size[1], video_size[0], 3), dtype=np.float32)

    for i in range(num_frames):
        t = i / (num_frames - 1)
        ft_pan = pan_fn(t)
        ft_zoom = zoom_fn(t)

        cx = (start_point[0] + ft_pan * (end_point[0] - start_point[0])) * W
        cy = (start_point[1] + ft_pan * (end_point[1] - start_point[1])) * H

        zoom = start_zoom + ft_zoom * (end_zoom - start_zoom)
        crop_w = int(W / zoom)
        crop_h = int(H / zoom)

        if noise_strength > 0:
            cx += noise_gen_x(i * noise_speed) * noise_strength * W * 0.01
            cy += noise_gen_y(i * noise_speed) * noise_strength * H * 0.01

        x_offset = max(0, min(int(cx - crop_w / 2), W - crop_w))
        y_offset = max(0, min(int(cy - crop_h / 2), H - crop_h))

        crop = image[y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
        frames[i] = cv2.resize(crop, video_size)

    return frames


# =========================
# ESEMPIO DI UTILIZZO
# =========================
if __name__ == "__main__":
    img = u.imread(
        "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon_processed/Place954/Place954_123_gt.tiff"
    )

    # Genera il video (solo in memoria come array NumPy 4D)
    video_frames = ken_burns_advanced(
        image=img,
        start_point=np.array([0.2, 0.2]),
        end_point=np.array([0.8, 0.8]),
        start_zoom=2,
        end_zoom=2.5,
        interp_pan="linear",
        interp_zoom="ease_in_out",
        noise_strength=0.8,
        noise_speed=0.2,
        noise_seed=42,
    )

    # Salvataggio separato
    # save_video(video_frames, "ken_burns_output", fps=30, format="mp4")

    # Riproduzione
    play_video_from_array(video_frames, fps=30)
