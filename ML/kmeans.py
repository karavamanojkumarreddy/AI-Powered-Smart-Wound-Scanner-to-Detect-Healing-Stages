"""
Wound K-Means Segmentation — Full Code
========================================
Step 3 of your pipeline:
  Autoencoder output → K-Means Segmentation → Wound region mask

What this file does:
  1. Loads denoised wound images
  2. Converts to Lab color space
  3. Runs K-Means clustering (k=5)
  4. Selects wound cluster using centre-proximity + colour scoring
  5. Cleans mask with morphological open + close
  6. Saves visualisation grid to results folder

Outputs:
  C:\\AI WoundScanner Project\\results\\kmeans_masks.png

Usage:
  python wound_kmeans.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR   = r'C:\AI WoundScanner Project\dataset'
FOLDER_NAMES  = ['inflammation', 'proliferation', 'maturation']
CLASS_NAMES   = ['Inflammation', 'Proliferation', 'Maturation']
OUTPUT_DIR    = r'C:\AI WoundScanner Project\results'
IMG_SIZE      = 256        # resize all images to this
KMEANS_K      = 5          # number of clusters
SAMPLES_SHOW  = 3          # images per class in visualisation
RANDOM_STATE  = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)

# Colours for the plot
BG     = '#F8F8F6'
DARK   = '#2C2C2A'
GRAY   = '#888780'
TEAL   = '#1D9E75'
CORAL  = '#D85A30'
PURPLE = '#7F77DD'
AMBER  = '#EF9F27'

CLASS_BORDER = {
    'Inflammation':  CORAL,
    'Proliferation': TEAL,
    'Maturation':    PURPLE,
}

# ─────────────────────────────────────────────
# STEP 1 — PREPROCESSING
# (same as in your main pipeline)
# ─────────────────────────────────────────────
def preprocess(image_bgr):
    """
    Resize → CLAHE on L channel → Bilateral filter → Normalize [0,1]
    Returns float32 array (IMG_SIZE, IMG_SIZE, 3) in BGR order.
    """
    img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))

    # CLAHE — improves contrast in wound region
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Bilateral filter — denoises while keeping wound edges sharp
    img   = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# STEP 2 — CONVERT TO Lab COLOR SPACE
# ─────────────────────────────────────────────
def to_lab(image_float_bgr):
    """
    Convert float32 BGR image [0,1] → Lab uint8.

    Why Lab?
      - L*  channel = lightness (brightness)
      - a*  channel = red-green axis  ← KEY for wound redness
      - b*  channel = blue-yellow axis

    Lab separates colour from brightness, making wound tissue
    (which is redder and darker than surrounding skin) much
    easier to isolate with K-Means than RGB.
    """
    uint8 = (image_float_bgr * 255).astype(np.uint8)
    lab   = cv2.cvtColor(uint8, cv2.COLOR_BGR2LAB)
    return lab


# ─────────────────────────────────────────────
# STEP 3 — K-MEANS CLUSTERING
# ─────────────────────────────────────────────
def run_kmeans(lab_image, k=KMEANS_K):
    """
    Run K-Means on Lab pixel values.

    Inputs:
      lab_image : uint8 Lab image (H, W, 3)
      k         : number of clusters (default 5)

    Returns:
      label_map : (H, W) int array  — cluster index per pixel
      centers   : (k, 3) float32   — Lab centroid of each cluster
    """
    h, w    = lab_image.shape[:2]
    pixels  = lab_image.reshape(-1, 3).astype(np.float32)

    # Termination: stop when change < 0.5 OR after 30 iterations
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,    # max iterations
        0.5    # epsilon
    )

    # cv2.KMEANS_PP_CENTERS = K-Means++ initialisation
    # (smarter starting points → faster convergence, better clusters)
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        attempts=10,                     # run 10 times, keep best
        flags=cv2.KMEANS_PP_CENTERS
    )

    label_map = labels.flatten().reshape(h, w)
    return label_map, centers


# ─────────────────────────────────────────────
# STEP 4 — SELECT WOUND CLUSTER
# ─────────────────────────────────────────────
def select_wound_cluster(label_map, centers):
    """
    Selects which K-Means cluster contains the wound region.

    Scoring system (each cluster gets a score):

    Score = (centre_proximity × 2.0) + (colour_score × 1.0)

    ① Centre-proximity score
       Wound images are usually photographed with the wound
       in the centre of the frame. We build a 2-D Gaussian
       weight map that peaks at the image centre and falls
       off towards the edges. Clusters whose pixels are
       concentrated near the centre score higher.

    ② Colour score (Lab-based)
       Wound tissue (all 3 stages) is:
         - Darker than surrounding skin  → lower L*
         - Redder or more saturated      → higher a*
       colour_score = (a* / 127) + (1 - L*/255) × 0.5
                     - brightness_penalty
       A brightness_penalty is subtracted for very bright
       clusters (L* > 160) which are almost certainly skin
       background or glare, not wound tissue.

    Coverage filter:
       Skip clusters covering < 5% or > 80% of the image.
       Too small = noise; too large = background.

    Returns:
       wound_cluster : int — index of the selected cluster
    """
    h, w   = label_map.shape
    cy, cx = h / 2.0, w / 2.0

    # ── Gaussian centre-weight map ──────────────────────────────
    ys, xs = np.mgrid[0:h, 0:w]
    sigma  = min(h, w) * 0.35           # spread covers ~35% of image
    centre_weight = np.exp(
        -((ys - cy)**2 + (xs - cx)**2) / (2 * sigma**2)
    )

    # ── Score each cluster ──────────────────────────────────────
    best_cluster = 0
    best_score   = -1e9

    for ci in range(len(centers)):
        cluster_mask = (label_map == ci).astype(np.float32)
        coverage     = cluster_mask.mean()

        # Skip too-small or too-large clusters
        if coverage < 0.05 or coverage > 0.80:
            continue

        # ① Centre-proximity score
        centre_score = (
            (cluster_mask * centre_weight).sum()
            / (cluster_mask.sum() + 1e-6)
        )

        # ② Colour score
        L_mean = float(centers[ci, 0])   # lightness
        a_mean = float(centers[ci, 1])   # red-green (a*)

        # Penalise very bright clusters (likely skin background)
        brightness_penalty = max(0.0, (L_mean - 160) / 95.0)

        colour_score = (
            (a_mean / 127.0)
            + (1.0 - L_mean / 255.0) * 0.5
            - brightness_penalty
        )

        # Combined score
        score = centre_score * 2.0 + colour_score * 1.0

        if score > best_score:
            best_score   = score
            best_cluster = ci

    return best_cluster


# ─────────────────────────────────────────────
# STEP 5 — BUILD BINARY MASK
# ─────────────────────────────────────────────
def build_mask(label_map, wound_cluster, img_h, img_w):
    """
    Creates a binary mask from the selected wound cluster.

    White (1) = wound region
    Black (0) = background / skin

    Morphological cleanup:
      OPEN  (erosion → dilation) : removes small noise specks
      CLOSE (dilation → erosion) : fills small holes inside wound
    """
    mask = (label_map == wound_cluster).astype(np.uint8)

    # Elliptical structuring element — smoother than a square kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Remove noise outside the wound region
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Fill gaps inside the wound region
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ── Fallback: if mask is too small, use centre 50% crop ──
    # (happens when K-Means cannot find a clear wound cluster)
    min_pixels = img_h * img_w * 0.05   # at least 5% of image
    if mask.sum() < min_pixels:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[img_h//4 : 3*img_h//4,
             img_w//4 : 3*img_w//4] = 1

    return mask


# ─────────────────────────────────────────────
# FULL PIPELINE — one image
# ─────────────────────────────────────────────
def kmeans_segment(image_bgr_raw, k=KMEANS_K):
    """
    Full K-Means segmentation pipeline for ONE image.

    Input  : raw BGR image (any size)
    Output : (preprocessed_float, lab_uint8, binary_mask, wound_pct)

    Steps:
      1. Preprocess   (resize, CLAHE, bilateral, normalize)
      2. Convert to Lab
      3. Run K-Means
      4. Select wound cluster
      5. Build binary mask
    """
    # Step 1 — preprocess
    pre = preprocess(image_bgr_raw)           # float32 BGR [0,1]

    # Step 2 — Lab conversion
    lab = to_lab(pre)                          # uint8 Lab

    # Step 3 — K-Means clustering
    label_map, centers = run_kmeans(lab, k=k)

    # Step 4 — select wound cluster
    wound_cluster = select_wound_cluster(label_map, centers)

    # Step 5 — binary mask
    h, w  = pre.shape[:2]
    mask  = build_mask(label_map, wound_cluster, h, w)

    # Wound coverage percentage
    wound_pct = float(mask.mean() * 100)

    return pre, lab, mask, wound_pct


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
def plot_kmeans_results(samples, out_path):
    """
    Plots a grid:  Original | Lab | Binary Mask
    One row per sample image.
    """
    n_rows = len(samples)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(11, n_rows * 2.8),
        facecolor=BG
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles  = [
        'Original (preprocessed)',
        'Lab conversion  (L* · a* · b*)',
        'Binary mask  (white = wound)'
    ]
    col_colours = [TEAL, PURPLE, CORAL]

    for ci_col, (title, colour) in enumerate(zip(col_titles, col_colours)):
        axes[0, ci_col].set_title(
            title, fontsize=10, fontweight='bold', color=colour, pad=7)

    for ri, s in enumerate(samples):
        cls        = s['class']
        fname      = s['fname']
        pre        = s['pre']      # float32 BGR
        lab        = s['lab']      # uint8 Lab
        mask       = s['mask']
        wound_pct  = s['wound_pct']
        border_col = CLASS_BORDER[cls]

        # ── Col 0: Original (convert BGR→RGB for display) ──
        ax = axes[ri, 0]
        rgb = cv2.cvtColor(
            (pre * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_ylabel(
            f'{cls}\n{fname[:16]}',
            fontsize=8, color=DARK,
            rotation=0, labelpad=86, va='center')
        for sp in ax.spines.values():
            sp.set_edgecolor(border_col); sp.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])

        # ── Col 1: Lab (normalise each channel to 0-255 for display) ──
        ax = axes[ri, 1]
        lab_disp = np.zeros_like(lab)
        for c in range(3):
            lab_disp[:, :, c] = cv2.normalize(
                lab[:, :, c], None, 0, 255, cv2.NORM_MINMAX)
        ax.imshow(lab_disp)
        # Channel labels
        for txt, xp, yp in [('L*', 0.12, 0.89),
                             ('a*', 0.50, 0.89),
                             ('b*', 0.88, 0.89)]:
            ax.text(xp, yp, txt,
                    transform=ax.transAxes,
                    fontsize=8, color='white', fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='black', alpha=0.45, lw=0))
        for sp in ax.spines.values():
            sp.set_edgecolor(PURPLE); sp.set_linewidth(1.5)
        ax.set_xticks([]); ax.set_yticks([])

        # ── Col 2: Binary mask (B&W) ──
        ax = axes[ri, 2]
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1)

        # Wound % badge
        ax.text(0.97, 0.04,
                f'wound area: {wound_pct:.1f}%',
                transform=ax.transAxes,
                fontsize=7.5, color='white', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='black', alpha=0.5, lw=0))
        # Mask quality hint
        quality = ('Good' if 10 <= wound_pct <= 70
                   else 'Check — too small' if wound_pct < 10
                   else 'Check — too large')
        q_col = '#1D9E75' if quality == 'Good' else '#D85A30'
        ax.text(0.03, 0.04, quality,
                transform=ax.transAxes,
                fontsize=7.5, color=q_col, ha='left', va='bottom',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', alpha=0.6, lw=0))
        for sp in ax.spines.values():
            sp.set_edgecolor(CORAL); sp.set_linewidth(1.5)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        'K-Means Segmentation — Original | Lab | Binary Mask',
        fontsize=13, fontweight='bold', color=DARK, y=1.005)

    # Legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(facecolor='#FCEBEB',
                       edgecolor=CORAL,   label='Inflammation'),
        mpatches.Patch(facecolor='#E1F5EE',
                       edgecolor=TEAL,    label='Proliferation'),
        mpatches.Patch(facecolor='#EEEDFE',
                       edgecolor=PURPLE,  label='Maturation'),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=3, fontsize=9, frameon=True,
               facecolor=BG, edgecolor='#D3D1C7',
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────
# LOAD SAMPLES + RUN
# ─────────────────────────────────────────────
def load_samples():
    """Load SAMPLES_SHOW images per class and run K-Means on each."""
    samples = []

    for label, folder in enumerate(FOLDER_NAMES):
        cls_dir = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(cls_dir):
            print(f"  [WARNING] Not found: {cls_dir}")
            continue

        files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])[:SAMPLES_SHOW]

        print(f"  {CLASS_NAMES[label]:20s}: processing {len(files)} images")

        for fname in files:
            path = os.path.join(cls_dir, fname)
            raw  = cv2.imread(path)
            if raw is None:
                print(f"    [SKIP] Cannot read: {fname}")
                continue

            pre, lab, mask, wound_pct = kmeans_segment(raw)

            samples.append({
                'class':     CLASS_NAMES[label],
                'fname':     fname,
                'pre':       pre,
                'lab':       lab,
                'mask':      mask,
                'wound_pct': wound_pct,
            })
            print(f"    {fname[:30]:30s}  →  wound area: {wound_pct:.1f}%")

    return samples


# ─────────────────────────────────────────────
# BATCH SEGMENTATION (use in your main pipeline)
# ─────────────────────────────────────────────
def segment_all_images(data_dir=DATASET_DIR, k=KMEANS_K):
    """
    Runs K-Means segmentation on every image in your dataset.
    Returns lists of masks and feature-ready images.

    Use this function from your main pipeline:

        from wound_kmeans import kmeans_segment, segment_all_images
        masks, images, labels = segment_all_images()
    """
    masks, images, labels = [], [], []

    for label, folder in enumerate(FOLDER_NAMES):
        cls_dir = os.path.join(data_dir, folder)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        for fname in files:
            raw = cv2.imread(os.path.join(cls_dir, fname))
            if raw is None:
                continue
            pre, _, mask, _ = kmeans_segment(raw, k=k)
            masks.append(mask)
            images.append(pre)
            labels.append(label)

    return masks, images, labels


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*58)
    print("  K-MEANS WOUND SEGMENTATION")
    print("  Original  →  Lab conversion  →  Binary Mask")
    print("="*58)
    print(f"\n  Dataset  : {DATASET_DIR}")
    print(f"  Clusters : k = {KMEANS_K}")
    print(f"  Img size : {IMG_SIZE} × {IMG_SIZE}")
    print(f"  Samples  : {SAMPLES_SHOW} per class "
          f"({SAMPLES_SHOW * 3} total)\n")

    # ── Load + segment sample images ──
    samples = load_samples()

    if not samples:
        print("  [ERROR] No images loaded. Check DATASET_DIR.")
        return

    # ── Plot visualisation ──
    out_path = os.path.join(OUTPUT_DIR, 'kmeans_masks.png')
    print(f"\n  Generating visualisation ({len(samples)} images × 3 columns)...")
    plot_kmeans_results(samples, out_path)

    # ── Summary ──
    print("\n" + "="*58)
    print("  DONE")
    print("="*58)
    print(f"\n  Mask grid saved → {out_path}")
    print()
    print("  How to read the mask quality:")
    print("  ─────────────────────────────────────────────────")
    print("  wound area 10–70%  →  Good mask")
    print("  wound area  < 10%  →  Mask too small "
          "(fallback used)")
    print("  wound area  > 70%  →  Mask too large "
          "(background included)")
    print()
    print("  Cluster scoring formula:")
    print("  ─────────────────────────────────────────────────")
    print("  score = centre_proximity × 2.0")
    print("        + colour_score     × 1.0")
    print()
    print("  colour_score = (a*/127)")
    print("               + (1 - L*/255) × 0.5")
    print("               - brightness_penalty")
    print("="*58 + "\n")

    print("  To use in your main pipeline:")
    print("  ─────────────────────────────────────────────────")
    print("  from wound_kmeans import kmeans_segment")
    print()
    print("  pre, lab, mask, pct = kmeans_segment(image_bgr)")
    print("  # pre  = preprocessed float32 image")
    print("  # lab  = Lab color space image (uint8)")
    print("  # mask = binary wound mask (0/1)")
    print("  # pct  = wound coverage % ")
    print("="*58 + "\n")


if __name__ == '__main__':
    main()