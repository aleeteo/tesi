import cv2
import numpy as np
import os
from typing import Union


def linear_tiff_to_srgb(
    img: Union[str, np.ndarray],
    apply_auto_wb: bool = True,
    wb_strength: float = 1.0,
    exposure: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """
    Converte un'immagine LSMI (16-bit lineare) in sRGB per visualizzazione.
    Accetta sia un percorso file (.tiff) che un array numpy.

    Parametri:
        img (str | np.ndarray): Path TIFF o immagine numpy già caricata.
        apply_auto_wb (bool): Applica auto white balance per canale.
        wb_strength (float): Intensità del white balance (0=off, 1=full).
        exposure (float): Fattore di esposizione globale.
        clip (bool): Se True, clippa i valori a [0,1] prima di convertire.

    Ritorna:
        np.ndarray: Immagine in sRGB uint8 (H,W,3).
    """
    # Se è un path, carica il file
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"File non trovato: {os.path.abspath(img)}")
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Impossibile leggere il TIFF: {os.path.abspath(img)}")

    if not isinstance(img, np.ndarray):
        raise TypeError("Input deve essere un np.ndarray o un path valido a un TIFF")

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    # Normalizza in base alla profondità a 16 bit (se necessario)
    if img.max() > 1.0:
        img = img / 65535.0

    # Auto White Balance controllato
    if apply_auto_wb:
        mean_per_channel = np.mean(img, axis=(0, 1), keepdims=True)
        scale = 1.0 / (mean_per_channel + 1e-8)
        scale = 1.0 + wb_strength * (scale - 1.0)
        img = img * scale

    # Correzione esposizione globale
    img = np.clip(img * exposure, 0, 1)

    # Conversione gamma sRGB
    def linear_to_srgb(x):
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)

    img_srgb = linear_to_srgb(img)
    if clip:
        img_srgb = np.clip(img_srgb, 0, 1)

    return (img_srgb * 255).astype(np.uint8)


def visualize_image(img: np.ndarray, title: str = "Immagine"):
    """Mostra a schermo un'immagine numpy già convertita in sRGB."""
    if img.dtype != np.uint8:
        raise ValueError(
            "L'immagine deve essere in formato uint8 per la visualizzazione."
        )
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Esempio 1: Caricamento da file TIFF
    img_srgb = linear_tiff_to_srgb(
        "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon/Place954/Place954_123.tiff"
    )
    visualize_image(img_srgb, "Da file TIFF")

    # Esempio 2: Input come array numpy (già in memoria)
    img_array = cv2.imread(
        "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon/Place954/Place954_123.tiff",
        cv2.IMREAD_UNCHANGED,
    )
    img_srgb_from_array = linear_tiff_to_srgb(img_array)
    visualize_image(img_srgb_from_array, "Da array numpy")
