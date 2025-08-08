import cv2
import numpy as np


def imread(path: str, sensor_depth: int = 14) -> np.ndarray:
    """
    Legge un TIFF 16-bit lineare e restituisce un'immagine float32 normalizzata [0,1].
    Funziona anche se il TIFF originale ha una profondità effettiva inferiore.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere il file: {path}")

    # Normalizza in [0,1] in base al massimo valore trovato
    max_val = img.max()
    if max_val > 0:
        img /= 2**sensor_depth - 1  # invece di 2**sensor_depth

    return img  # float32 lineare [0,1]


def imwrite(path: str, img: np.ndarray):
    """
    Salva un'immagine float32 lineare [0,1] come TIFF 16-bit lineare.
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    # Clippa e converte in uint16
    img_16bit = np.clip(img * (2**16 - 1), 0, 2**16 - 1).astype(np.uint16)

    if not path.lower().endswith(".tiff") and not path.lower().endswith(".tif"):
        path += ".tiff"

    cv2.imwrite(path, img_16bit)


def lin2srgb(
    arr: np.ndarray,
    clip: bool = True,
    out_dtype: np.dtype = np.uint8,  # type: ignore
    copy: bool = True,
) -> np.ndarray:
    """
    Converte un'immagine o video in gamma sRGB.
    - Input: float32 in [0,1], shape (..., C) con C=1/3 (2D/3D/4D ok).
    - Output: uint8 [0,255] di default (o float se out_dtype=np.float32).
    - Se copy=False e arr è float32 writeable, modifica in-place (attenzione!).
    """
    a = np.asarray(arr, dtype=np.float32)
    if copy:
        a = a.copy()

    if clip:
        np.clip(a, 0.0, 1.0, out=a)

    out_f = np.empty_like(a, dtype=np.float32)
    m = a <= 0.0031308

    # ramo lineare
    out_f[m] = 12.92 * a[m]

    # ramo non-lineare — NIENTE out= su slicing booleano
    nonlin = np.power(a[~m], 1 / 2.4)
    out_f[~m] = 1.055 * nonlin - 0.055

    if out_dtype == np.uint8:
        return np.clip(out_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    elif out_dtype == np.float32:
        return out_f
    else:
        return out_f.astype(out_dtype)


def imshow(img: np.ndarray, winname: str = "Image", wait: bool = True):
    """
    Mostra un'immagine float32 lineare (convertendola a sRGB 8-bit per preview).
    """
    preview = lin2srgb(img) if img.dtype == np.float32 else img
    cv2.imshow(winname, preview)
    if wait:
        cv2.waitKey(0)
        cv2.destroyWindow(winname)


if __name__ == "__main__":
    # Lettura in lineare
    img = imread(
        "/Users/alessandroteodori/stage/code/LSMI-dataset/nikon_processed/Place954/Place954_123_gt.tiff"
    )  # float32 [0,1]

    # Visualizzazione
    imshow(img, "preview")  # converte temporaneamente a sRGB 8-bit

    # Elaborazioni varie in float lineare...
    # ...

    # Salvataggio come TIFF lineare 16-bit
    # imwrite("output.tiff", img)
