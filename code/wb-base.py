import cv22
import numpy as np
import matplotlib.pyplot as plt


def gray_world(img, mask):
    means = [np.mean(img[:, :, c][mask]) for c in range(3)]
    avg = sum(means) / 3
    scale = [avg / m if m > 0 else 1.0 for m in means]
    img_out = img.copy().astype(np.float32)
    for c in range(3):
        img_out[:, :, c] *= scale[c]
    return np.clip(img_out, 0, 255).astype(np.uint8)


def white_patch(img, mask):
    max_vals = [np.max(img[:, :, c][mask]) for c in range(3)]
    scale = [255 / m if m > 0 else 1.0 for m in max_vals]
    img_out = img.copy().astype(np.float32)
    for c in range(3):
        img_out[:, :, c] *= scale[c]
    return np.clip(img_out, 0, 255).astype(np.uint8)


def gray_edge(img, mask, order=1, sigma=1.0):
    img_blur = cv22.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    grads = [
        cv22.Sobel(img_blur[:, :, c], cv22.CV_32F, 1, 0, ksize=3) ** 2
        + cv22.Sobel(img_blur[:, :, c], cv22.CV_32F, 0, 1, ksize=3) ** 2
        for c in range(3)
    ]
    means = [np.mean(g[mask]) for g in grads]
    avg = sum(means) / 3
    scale = [np.sqrt(avg / m) if m > 0 else 1.0 for m in means]
    img_out = img.copy().astype(np.float32)
    for c in range(3):
        img_out[:, :, c] *= scale[c]
    return np.clip(img_out, 0, 255).astype(np.uint8)


def get_non_saturated_mask(img, thresh=250):
    return np.all(img < thresh, axis=2)


def apply_mask_visualization(img, mask):
    """Rende nero tutto ciò che è fuori dalla maschera"""
    masked_img = np.zeros_like(img)
    for c in range(3):
        masked_img[:, :, c][mask] = img[:, :, c][mask]
    return masked_img


def show_side_by_side_results(
    original, corrected_images, masks_used, titles_corr, titles_masks
):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    all_corr = [original] + corrected_images
    all_masks = [original] + [apply_mask_visualization(original, m) for m in masks_used]

    for ax, img, title in zip(axs[0], all_corr, titles_corr):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    for ax, img, title in zip(axs[1], all_masks, titles_masks):
        ax.imshow(img)
        ax.set_title(title + " - Masked Area")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python awb_comparison.py path_immagine")
        sys.exit(1)

    path = sys.argv[1]
    bgr = cv22.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Impossibile aprire '{path}'")

    rgb = cv22.cv2tColor(bgr, cv22.COLOR_BGR2RGB).astype(np.uint8)
    mask = get_non_saturated_mask(rgb)

    gw = gray_world(rgb, mask)
    wp = white_patch(rgb, mask)
    ge = gray_edge(rgb, mask)

    show_side_by_side_results(
        original=rgb,
        corrected_images=[gw, wp, ge],
        masks_used=[mask, mask, mask],
        titles_corr=["Originale", "Gray-World", "White-Patch", "Gray-Edge"],
        titles_masks=["Originale", "Gray-World", "White-Patch", "Gray-Edge"],
    )
