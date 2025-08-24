import numpy as np
import cv2
from typing import Literal, Optional

NormMode = Literal["l2", "sum", "max", "none"]


def _check_img(img: np.ndarray) -> None:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img deve essere HxWx3 (RGB).")
    if img.dtype != np.float32:
        raise TypeError("img deve essere np.float32.")


def _check_vid(vid: np.ndarray) -> None:
    if vid.ndim != 4 or vid.shape[-1] != 3:
        raise ValueError("video deve essere TxHxWx3 (RGB).")
    if vid.dtype != np.float32:
        raise TypeError("video deve essere np.float32.")


def _normalize(vec: np.ndarray, mode: NormMode = "l2", eps: float = 1e-8) -> np.ndarray:
    v = vec.astype(np.float32, copy=False)
    if mode == "l2":
        n = np.linalg.norm(v) + eps
        return v / n
    if mode == "sum":
        s = float(np.sum(v)) + eps
        return v / s
    if mode == "max":
        m = float(np.max(v)) + eps
        return v / m
    return v


def _valid_mask_from(
    img: np.ndarray, mask: Optional[np.ndarray], ignore_black: bool
) -> Optional[np.ndarray]:
    """
    Restituisce una maschera booleana valida per img (HxW) o None.
    - Se mask è fornita: usa quella.
    - Altrimenti, se ignore_black: True -> mask = any(img>0) sui 3 canali.
    """
    if mask is not None:
        return mask.astype(bool) if mask.dtype != np.bool_ else mask
    if ignore_black:
        return np.any(img > 0.0, axis=-1)
    return None


# --- Derivate per Gray-Edge ---
def _gaussian_blur_channels(I: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return I
    k = int(np.ceil(3 * sigma) * 2 + 1)
    out = np.empty_like(I)
    for c in range(3):
        out[..., c] = cv2.GaussianBlur(
            I[..., c],
            (k, k),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
    return out


def _grad_mag(img_ch: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img_ch, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Sobel(img_ch, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    return np.hypot(gx, gy)


def _laplacian_mag(img_ch: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(img_ch, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT101)
    return np.abs(lap)


# --- Stimatori immagine singola (aggiunto ignore_black) ---
def illuminant_gray_world(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalize: NormMode = "l2",
    ignore_black: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    _check_img(img)
    m = _valid_mask_from(img, mask, ignore_black)
    flat = img.reshape(-1, 3) if m is None else img[m].reshape(-1, 3)
    if flat.size == 0:
        raise ValueError("Nessun pixel valido per la stima.")
    e = flat.mean(axis=0)
    return _normalize(e, normalize, eps)


def illuminant_white_patch(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    percentile: float = 100.0,
    normalize: NormMode = "l2",
    ignore_black: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    _check_img(img)
    m = _valid_mask_from(img, mask, ignore_black)
    flat = img.reshape(-1, 3) if m is None else img[m].reshape(-1, 3)
    if flat.size == 0:
        raise ValueError("Nessun pixel valido per la stima.")
    e = np.percentile(flat, percentile, axis=0).astype(np.float32)
    return _normalize(e, normalize, eps)


def illuminant_shades_of_gray(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    p: float = 6.0,
    normalize: NormMode = "l2",
    ignore_black: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    if p <= 0:
        raise ValueError("p dev'essere > 0")
    _check_img(img)
    m = _valid_mask_from(img, mask, ignore_black)
    flat = img.reshape(-1, 3) if m is None else img[m].reshape(-1, 3)
    if flat.size == 0:
        raise ValueError("Nessun pixel valido per la stima.")
    m_p = np.mean(np.power(flat, p, dtype=np.float32), axis=0)
    e = np.power(m_p + eps, 1.0 / p, dtype=np.float32)
    return _normalize(e, normalize, eps)


def illuminant_gray_edge(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    p: float = 6.0,
    order: Literal[1, 2] = 1,
    sigma: float = 1.0,
    normalize: NormMode = "l2",
    ignore_black: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    if p <= 0:
        raise ValueError("p dev'essere > 0")
    _check_img(img)
    sm = _gaussian_blur_channels(img, sigma)
    # maschera dei pixel validi determinata sull'immagine originale (non sfocata)
    m = _valid_mask_from(img, mask, ignore_black)
    feats = np.empty(3, dtype=np.float32)
    for c in range(3):
        ch = sm[..., c]
        F = _grad_mag(ch) if order == 1 else _laplacian_mag(ch)
        vals = F if m is None else F[m]
        if vals.size == 0:
            raise ValueError("Nessun pixel valido per la stima.")
        m_p = float(np.mean(np.power(vals, p)))
        feats[c] = (m_p + eps) ** (1.0 / p)
    return _normalize(feats, normalize, eps)


# --- Router immagine ---
def estimate_illuminant(
    img: np.ndarray,
    method: Literal["grayworld", "whitepatch", "shadesofgray", "grayedge"],
    **kwargs,
) -> np.ndarray:
    method = method.lower()
    if method == "grayworld":
        return illuminant_gray_world(img, **kwargs)
    if method == "whitepatch":
        return illuminant_white_patch(img, **kwargs)
    if method == "shadesofgray":
        return illuminant_shades_of_gray(img, **kwargs)
    if method == "grayedge":
        return illuminant_gray_edge(img, **kwargs)
    raise ValueError(f"Metodo sconosciuto: {method}")


# --- API video: T x H x W x 3 -> T x 3 ---
def estimate_illuminant_video(
    video: np.ndarray,
    method: Literal["grayworld", "whitepatch", "shadesofgray", "grayedge"],
    **kwargs,
) -> np.ndarray:
    """
    Calcola la stima per ogni frame. Restituisce array (T,3) float32.
    kwargs passati allo stimatore (es. p, sigma, percentile, normalize, ignore_black).
    """
    _check_vid(video)
    T = video.shape[0]
    out = np.empty((T, 3), dtype=np.float32)
    for t in range(T):
        out[t] = estimate_illuminant(video[t], method=method, **kwargs)
    return out
