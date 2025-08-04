import cv2
import numpy as np


def linear_tiff_to_srgb(
    img: np.ndarray,
    apply_auto_wb: bool = True,
    wb_strength: float = 1.0,
    exposure: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """
    Converte un'immagine LSMI (16-bit lineare) in sRGB per visualizzazione.

    Parametri:
        img (np.ndarray): Immagine in input (16-bit lineare).
        apply_auto_wb (bool): Applica auto white balance per canale.
        wb_strength (float): Intensità del white balance (0=off, 1=full).
        exposure (float): Fattore di esposizione globale.
        clip (bool): Se True, clippa i valori a [0,1].

    Ritorna:
        np.ndarray: Immagine in sRGB uint8 (H,W,3).
    """
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


if __name__ == "__main__":
    # Esempio d'uso:
    img = cv2.imread(
        "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon_processed/Place954/Place954_123.tiff",
        cv2.IMREAD_UNCHANGED,
    )
    img_srgb = linear_tiff_to_srgb(img)
    cv2.imshow("TIFF LSMI", img_srgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

