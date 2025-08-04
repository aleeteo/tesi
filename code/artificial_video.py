import cv2
import math
import numpy as np
from typing import List, Optional
from perlin_noise import PerlinNoise


def write_video(frames: List[np.ndarray], output_path: str, fps: int):
    """Scrive una lista di frame in un file video."""
    if not frames:
        raise ValueError("Nessun frame fornito per la scrittura del video.")

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        out.write(frame)
    out.release()


def ken_burns_effect_array(
    image: np.ndarray,
    video_fps: int = 30,
    video_duration: int = 5,
    zoom_factor: float = 1.2,
    video_size: tuple = (1280, 720),
    pan_direction: str = "diagonal",  # "horizontal", "vertical", "none"
    output_path: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Effetto Ken Burns base (zoom lineare + pan semplice) compatibile con np.ndarray.
    """
    h, w, _ = image.shape
    total_frames = video_fps * video_duration
    frames = []

    for i in range(total_frames):
        t = i / total_frames  # Interpolazione 0 → 1

        # Zoom progressivo lineare
        scale = 1 + t * (zoom_factor - 1)
        crop_w = int(w / scale)
        crop_h = int(h / scale)

        # Offset pan in base alla direzione scelta
        if pan_direction == "horizontal":
            x_offset = int((w - crop_w) * t)
            y_offset = (h - crop_h) // 2
        elif pan_direction == "vertical":
            x_offset = (w - crop_w) // 2
            y_offset = int((h - crop_h) * t)
        elif pan_direction == "none":
            x_offset = (w - crop_w) // 2
            y_offset = (h - crop_h) // 2
        else:  # "diagonal"
            x_offset = int((w - crop_w) * t)
            y_offset = int((h - crop_h) * t)

        # Crop dinamico e resize
        crop = image[y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
        frame = cv2.resize(crop, video_size)
        frames.append(frame)

    if output_path:
        write_video(frames, output_path, video_fps)

    return frames


def ken_burns_advanced_array(
    image: np.ndarray,
    video_fps: int = 30,
    video_duration: int = 5,
    start_point: tuple = (0.5, 0.5),
    end_point: tuple = (0.5, 0.5),
    start_zoom: float = 1.2,
    end_zoom: float = 1.2,
    video_size: tuple = (1280, 720),
    interp_pan: str = "linear",
    interp_zoom: str = "ease_in_out",
    noise_strength: float = 0.0,
    noise_speed: float = 0.5,
    noise_seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Effetto Ken Burns avanzato con interpolazioni e rumore Perlin deterministico.
    """

    # Interpolazioni disponibili
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

    # Generatore di rumore Perlin deterministico
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

    # Dimensioni immagine
    H, W, _ = image.shape
    total_frames = int(video_fps * video_duration)
    frames = []

    for i in range(total_frames):
        t = i / (total_frames - 1)
        ft_pan = pan_fn(t)
        ft_zoom = zoom_fn(t)

        # Pan
        cx = (start_point[0] + ft_pan * (end_point[0] - start_point[0])) * W
        cy = (start_point[1] + ft_pan * (end_point[1] - start_point[1])) * H

        # Zoom
        zoom = start_zoom + ft_zoom * (end_zoom - start_zoom)
        crop_w = int(W / zoom)
        crop_h = int(H / zoom)

        # Rumore (jitter) deterministico
        if noise_strength > 0:
            cx += noise_gen_x(i * noise_speed) * noise_strength * W * 0.01
            cy += noise_gen_y(i * noise_speed) * noise_strength * H * 0.01

        # Crop centrato
        x_offset = max(0, min(int(cx - crop_w / 2), W - crop_w))
        y_offset = max(0, min(int(cy - crop_h / 2), H - crop_h))

        crop = image[y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
        frame = cv2.resize(crop, video_size)
        frames.append(frame)

    if output_path:
        write_video(frames, output_path, video_fps)

    return frames


def play_video_from_frames(frames: List[np.ndarray], fps: int = 30):
    """Riproduce un video da una lista di frame numpy."""
    for frame in frames:
        cv2.imshow("Riproduzione Video", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread("Place925.jpg")

    # Versione base
    frames_basic = ken_burns_effect_array(
        image=img,
        zoom_factor=1.2,
        pan_direction="diagonal",
        output_path="ken_burns_basic.mp4",
    )

    # Versione avanzata
    frames_advanced = ken_burns_advanced_array(
        image=img,
        start_point=(0.1, 0.1),
        end_point=(0.8, 0.8),
        start_zoom=2,
        end_zoom=2.5,
        interp_pan="linear",
        interp_zoom="ease_in_out",
        noise_strength=0.8,
        noise_speed=0.2,
        noise_seed=42,
        output_path="ken_burns_advanced.mp4",
    )

    # Riproduzione video base
    play_video_from_frames(frames_basic, fps=30)

