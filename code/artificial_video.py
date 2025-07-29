import cv2
import math
import noise
import numpy as np


def ken_burns_effect(
    input_image: str,
    output_video: str = "ken_burns_effect.mp4",
    video_fps: int = 30,
    video_duration: int = 5,
    zoom_factor: float = 1.2,
    video_size: tuple = (1280, 720),
    pan_direction: str = "diagonal",  # "horizontal", "vertical", "none"
):
    """
    Applica un effetto Ken Burns (pan & zoom) su un'immagine e crea un video.

    Args:
        input_image (str): Percorso all'immagine di input.
        output_video (str): Nome file del video di output.
        video_fps (int): Frame per secondo del video.
        video_duration (int): Durata del video in secondi.
        zoom_factor (float): Fattore di zoom finale (es. 1.2 = 20% in più).
        video_size (tuple): Risoluzione di output (larghezza, altezza).
        pan_direction (str): Direzione pan: "diagonal", "horizontal", "vertical", "none".
    """
    img = cv2.imread(input_image)
    if img is None:
        raise ValueError(f"Errore: Immagine '{input_image}' non trovata!")

    h, w, _ = img.shape
    total_frames = video_fps * video_duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, video_fps, video_size)

    for i in range(total_frames):
        t = i / total_frames  # Interpolazione 0 → 1

        # Zoom progressivo (lineare)
        scale = 1 + t * (zoom_factor - 1)
        crop_w = int(w / scale)
        crop_h = int(h / scale)

        # Offset pan
        if pan_direction == "horizontal":
            x_offset = int((w - crop_w) * t)
            y_offset = (h - crop_h) // 2
        elif pan_direction == "vertical":
            x_offset = (w - crop_w) // 2
            y_offset = int((h - crop_h) * t)
        elif pan_direction == "none":
            x_offset = (w - crop_w) // 2
            y_offset = (h - crop_h) // 2
        else:  # diagonale
            x_offset = int((w - crop_w) * t)
            y_offset = int((h - crop_h) * t)

        # Crop dinamico e ridimensionamento
        crop = img[y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
        frame = cv2.resize(crop, video_size)
        out.write(frame)

    out.release()
    print(f"✅ Video creato: {output_video}")


def ken_burns_custom_pan(
    input_image: str,
    output_video: str = "ken_burns_custom.mp4",
    video_fps: int = 30,
    video_duration: int = 5,
    start_zoom: float = 1.0,
    end_zoom: float = 1.2,
    start_point: tuple = (
        0.2,
        0.2,
    ),  # (x,y) normalizzati (0=sinistra/alto, 1=destra/basso)
    end_point: tuple = (0.8, 0.8),  # idem
    video_size: tuple = (1280, 720),
):
    """
    Effetto Ken Burns personalizzato con pan tra due punti definiti.

    Args:
        input_image (str): Percorso all'immagine.
        output_video (str): File video di output.
        video_fps (int): Frame per secondo.
        video_duration (int): Durata video in secondi.
        start_zoom (float): Zoom iniziale (1.0 = nessuno).
        end_zoom (float): Zoom finale (>1 per zoom in, <1 per zoom out).
        start_point (tuple): Punto iniziale (x,y) normalizzati [0,1].
        end_point (tuple): Punto finale (x,y) normalizzati [0,1].
        video_size (tuple): Risoluzione del video (w,h).
    """
    img = cv2.imread(input_image)
    if img is None:
        raise ValueError(f"Errore: Immagine '{input_image}' non trovata!")

    h, w, _ = img.shape
    total_frames = video_fps * video_duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, video_fps, video_size)

    for i in range(total_frames):
        t = i / total_frames  # Interpolazione 0→1

        # Interpolazione zoom
        scale = start_zoom + t * (end_zoom - start_zoom)
        crop_w = int(w / scale)
        crop_h = int(h / scale)

        # Interpolazione pan (da punto iniziale a punto finale)
        cx = (start_point[0] + t * (end_point[0] - start_point[0])) * w
        cy = (start_point[1] + t * (end_point[1] - start_point[1])) * h

        # Calcolo rettangolo di crop centrato su (cx,cy)
        x_offset = int(cx - crop_w / 2)
        y_offset = int(cy - crop_h / 2)

        # Assicuriamoci che il crop resti dentro l'immagine
        x_offset = max(0, min(x_offset, w - crop_w))
        y_offset = max(0, min(y_offset, h - crop_h))

        # Crop dinamico
        crop = img[y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
        frame = cv2.resize(crop, video_size)
        out.write(frame)

    out.release()
    print(f"✅ Video creato: {output_video}")


def ken_burns_advanced(
    input_image: str,
    output_video: str = "ken_burns_advanced.mp4",
    video_fps: int = 30,
    video_duration: int = 5,
    start_point: tuple = (0.5, 0.5),
    end_point: tuple = (0.5, 0.5),
    start_zoom: float = 1.2,
    end_zoom: float = 1.2,
    video_size: tuple = (1280, 720),
    interp_pan: str = "linear",  # Interpolazione pan
    interp_zoom: str = "ease_in_out",  # Interpolazione zoom
    noise_strength: float = 0.0,
    noise_speed: float = 0.5,
):
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

    # Carica immagine
    img = cv2.imread(input_image)
    if img is None:
        raise ValueError(f"Errore: Immagine '{input_image}' non trovata!")
    H, W, _ = img.shape

    total_frames = int(video_fps * video_duration)
    out = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, video_size
    )

    for i in range(total_frames):
        t = i / (total_frames - 1)

        # Interpolazioni separate
        ft_pan = pan_fn(t)  # pan costante
        ft_zoom = zoom_fn(t)  # zoom fluido

        # Centro camera (pan)
        cx = (start_point[0] + ft_pan * (end_point[0] - start_point[0])) * W
        cy = (start_point[1] + ft_pan * (end_point[1] - start_point[1])) * H

        # Zoom interpolato
        zoom = start_zoom + ft_zoom * (end_zoom - start_zoom)
        crop_w = int(W / zoom)
        crop_h = int(H / zoom)

        # Rumore jitter (camera shake)
        if noise_strength > 0:
            cx += noise.pnoise1(i * noise_speed) * noise_strength * W * 0.01
            cy += noise.pnoise1((i + 1000) * noise_speed) * noise_strength * H * 0.01

        # Crop centrato
        x_offset = max(0, min(int(cx - crop_w / 2), W - crop_w))
        y_offset = max(0, min(int(cy - crop_h / 2), H - crop_h))

        crop = img[y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
        frame = cv2.resize(crop, video_size)
        out.write(frame)

    out.release()
    print(f"✅ Video creato: {output_video}")


if __name__ == "__main__":
    # Genera video con effetto Ken Burns
    input_path = "Place925.jpg"
    output_path = "ken_burns_output.mp4"

    ken_burns_advanced(
        input_image=input_path,
        output_video=output_path,
        start_point=(0.1, 0.1),
        end_point=(0.9, 0.9),
        start_zoom=1.5,
        end_zoom=2,
        interp_pan="linear",
        interp_zoom="ease_in_out",
    )

    # --- Riproduzione automatica del video ---
    cap = cv2.VideoCapture(output_path)
    if not cap.isOpened():
        raise IOError("Errore: impossibile aprire il video per la riproduzione.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Riproduzione Video", frame)
        # Avanza con un delay in base al framerate
        if cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
