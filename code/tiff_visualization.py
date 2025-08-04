import cv2
import numpy as np
import os


# TODO: input np.ndarray (immagine gia' caricata da file)
# TODO: output tipizzato
def linear_tiff_to_srgb(
    tiff_path, apply_auto_wb=True, wb_strength=1, exposure=1, clip=True
):
    """
    Legge un TIFF LSMI a 16 bit lineare e lo converte in sRGB per visualizzazione.
    """
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"File non trovato: {os.path.abspath(tiff_path)}")

    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Impossibile leggere il TIFF: {os.path.abspath(tiff_path)}")

    img = img.astype(np.float32)

    # Normalizza in base alla profondità a 16 bit
    img_norm = img / 65535.0

    # Auto White Balance controllato
    if apply_auto_wb:
        mean_per_channel = np.mean(img_norm, axis=(0, 1), keepdims=True)
        scale = 1.0 / (mean_per_channel + 1e-8)
        scale = 1.0 + wb_strength * (scale - 1.0)  # bilanciamento parziale
        img_norm = img_norm * scale

    # Correzione esposizione globale
    img_norm = np.clip(img_norm * exposure, 0, 1)

    # Conversione gamma sRGB
    def linear_to_srgb(x):
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)

    img_srgb = linear_to_srgb(img_norm)
    if clip:
        img_srgb = np.clip(img_srgb, 0, 1)

    return (img_srgb * 255).astype(np.uint8)


def visualize_tiff(tiff_path):
    img_srgb = linear_tiff_to_srgb(tiff_path)
    cv2.imshow(f"TIFF LSMI - {os.path.basename(tiff_path)}", img_srgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # ESEMPIO:
    visualize_tiff(
        "/Users/alessandroteodori/Documents/stage/code/LSMI-dataset/nikon_processed/Place954/Place954_123.tiff"
    )
