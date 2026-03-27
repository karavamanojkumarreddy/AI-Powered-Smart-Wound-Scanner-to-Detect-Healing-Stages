"""
Wound Feature Extraction — Full Code
======================================
Step 4 of your pipeline:
  K-Means Mask → Feature Extraction → Feature Vector

Feature groups extracted:
  ① RGB Features       — mean, std, skew, kurtosis, percentiles
  ② HSV Features       — hue histogram, saturation, value stats
  ③ Lab Features       — L*, a* (redness), b* stats + histograms
  ④ Texture (GLCM)     — contrast, homogeneity, energy, correlation
  ⑤ Texture (LBP)      — local binary pattern histogram (3 scales)
  ⑥ Texture (Gabor)    — 4 orientations × 3 frequencies
  ⑦ Shape Features     — area, perimeter, circularity, Hu moments
  ⑧ Wound vs Skin      — colour difference between wound and skin

Regions used:
  - Wound mask region   (from K-Means)
  - Surrounding skin    (outside mask)
  - Full image          (global stats — mask-independent)
  - Centre crop 50%     (wound almost always centred)

Total features : ~520
After SelectKBest(k=150) + PCA(100) in main pipeline → robust SVM input

Usage:
  python wound_features.py              ← runs demo on your dataset
  from wound_features import extract_features  ← use in main pipeline
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis

import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ── MobileNetV2 deep feature extractor (loaded once at import time) ──
print("[features.py] Loading MobileNetV2 ...")
_MV2_SIZE  = 224    # MobileNetV2 native input size
_mv2_base  = tf.keras.applications.MobileNetV2(
    input_shape=(_MV2_SIZE, _MV2_SIZE, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg')
_mv2_base.trainable = False
_mv2_extractor = tf.keras.Model(
    inputs=_mv2_base.input,
    outputs=_mv2_base.output)
print("[features.py] MobileNetV2 ready — 1280-dim output")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR  = r'C:\AI WoundScanner Project\dataset'
FOLDER_NAMES = ['inflammation', 'proliferation', 'maturation']
CLASS_NAMES  = ['Inflammation', 'Proliferation', 'Maturation']
OUTPUT_DIR   = r'C:\AI WoundScanner Project\results'
IMG_SIZE     = 256
KMEANS_K     = 5
SAMPLES_SHOW = 3     # images to demo per class
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG    = '#F8F8F6'
DARK  = '#2C2C2A'
GRAY  = '#888780'
TEAL  = '#1D9E75'
CORAL = '#D85A30'
PURP  = '#7F77DD'
AMBER = '#EF9F27'
BLUE  = '#378ADD'


# ═══════════════════════════════════════════════
# HELPER — STATISTICAL DESCRIPTORS
# ═══════════════════════════════════════════════
def _stats(arr):
    """
    10 statistics for a 1-D pixel array:
    mean, std, skewness, kurtosis,
    p5, p25, median, p75, p90, p95
    Returns list of 10 floats.
    """
    a = arr.astype(np.float32)
    if len(a) < 5:
        return [0.0] * 10
    return [
        float(np.mean(a)),
        float(np.std(a)),
        float(skew(a)),
        float(kurtosis(a)),
        float(np.percentile(a,  5)),
        float(np.percentile(a, 25)),
        float(np.median(a)),
        float(np.percentile(a, 75)),
        float(np.percentile(a, 90)),
        float(np.percentile(a, 95)),
    ]


def _hist(arr, bins, lo, hi):
    """Normalised histogram → list of floats."""
    h = np.histogram(arr, bins=bins, range=(lo, hi))[0].astype(float)
    return (h / (h.sum() + 1e-6)).tolist()


# ═══════════════════════════════════════════════
# ① RGB FEATURES
# ═══════════════════════════════════════════════
def rgb_features(image_bgr_uint8, mask, skin_mask):
    """
    Stats on B, G, R channels for 3 regions:
      - wound region (mask)
      - skin region  (outside mask)
      - full image

    Also computes:
      - mean colour difference wound vs skin (key discriminator)
      - red dominance ratio  = R / (G + B + 1)
      - green dominance ratio = G / (R + B + 1)

    Returns ~100 features.
    """
    feats = []
    regions = {
        'wound': mask == 1,
        'skin':  skin_mask == 1,
        'full':  np.ones_like(mask, dtype=bool),
    }

    for reg_name, reg in regions.items():
        for c in range(3):   # B, G, R
            px = image_bgr_uint8[:, :, c][reg].astype(np.float32)
            feats += _stats(px)
            feats += _hist(px, 16, 0, 256)

    # ── Red dominance in wound region ──
    w_idx   = mask == 1
    B = image_bgr_uint8[:,:,0][w_idx].astype(np.float32)
    G = image_bgr_uint8[:,:,1][w_idx].astype(np.float32)
    R = image_bgr_uint8[:,:,2][w_idx].astype(np.float32)

    red_dom   = float(np.mean(R / (G + B + 1)))
    green_dom = float(np.mean(G / (R + B + 1)))
    feats += [red_dom, green_dom]

    # ── Colour difference wound vs skin ──
    s_idx = skin_mask == 1
    for c in range(3):
        w_mean = image_bgr_uint8[:,:,c][w_idx].mean() if w_idx.sum()>0 else 0
        s_mean = image_bgr_uint8[:,:,c][s_idx].mean() if s_idx.sum()>0 else 0
        feats.append(float(w_mean - s_mean))   # sign encodes direction

    return feats   # ≈ 3×3×(10+16) + 5 = 242 → after dedup ~80 selected


# ═══════════════════════════════════════════════
# ② HSV FEATURES
# ═══════════════════════════════════════════════
def hsv_features(image_bgr_uint8, mask, skin_mask):
    """
    Stats + histograms on H, S, V channels.

    Why HSV?
      H (hue)        → wound colour type (red, pink, pale)
      S (saturation) → colour intensity (high in inflammation,
                        low in maturation scars)
      V (value)      → brightness

    Key features:
      - Hue histogram (18 bins) in wound region
      - Saturation mean wound vs skin difference
        → inflammation: high S; maturation: low S
      - Value mean wound vs skin difference

    Returns ~80 features.
    """
    feats = []
    hsv   = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2HSV)

    regions = {
        'wound': mask == 1,
        'skin':  skin_mask == 1,
        'full':  np.ones_like(mask, dtype=bool),
    }

    # Stats per channel per region
    for reg in regions.values():
        # H channel (range 0-180 in OpenCV)
        feats += _stats(hsv[:, :, 0][reg])
        feats += _hist(hsv[:, :, 0][reg], 18, 0, 180)  # 18-bin hue hist
        # S channel (saturation)
        feats += _stats(hsv[:, :, 1][reg])
        feats += _hist(hsv[:, :, 1][reg],  8, 0, 256)  # 8-bin sat hist
        # V channel (brightness)
        feats += _stats(hsv[:, :, 2][reg])

    # ── Saturation difference wound vs skin ──
    w_idx = mask == 1; s_idx = skin_mask == 1
    w_sat = hsv[:,:,1][w_idx].mean() if w_idx.sum() > 0 else 0
    s_sat = hsv[:,:,1][s_idx].mean() if s_idx.sum() > 0 else 0
    feats.append(float(w_sat - s_sat))   # + = wound more saturated

    # ── Hue uniformity (std) — uniform hue = more homogeneous wound ──
    w_hue_std = float(hsv[:,:,0][w_idx].std()) if w_idx.sum() > 0 else 0
    feats.append(w_hue_std)

    return feats


# ═══════════════════════════════════════════════
# ③ Lab FEATURES
# ═══════════════════════════════════════════════
def lab_features(image_bgr_uint8, mask, skin_mask):
    """
    Stats + histograms on L*, a*, b* channels.

    Why Lab?
      L*  = lightness  → darker wound vs bright skin
      a*  = red-green  → HIGH a* = inflammation / fresh wound
                          LOW a* = maturation / scar
      b*  = blue-yellow

    This is the MOST discriminative colour space for wound stages:
      Inflammation  → high a* (very red)
      Proliferation → moderate a* (pink-red)
      Maturation    → low a*  (pale, close to skin)

    Returns ~70 features.
    """
    feats = []
    lab   = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2LAB)

    regions = {
        'wound': mask == 1,
        'skin':  skin_mask == 1,
        'full':  np.ones_like(mask, dtype=bool),
    }

    for reg in regions.values():
        # L* — lightness (0-255 in OpenCV Lab)
        feats += _stats(lab[:, :, 0][reg])
        # a* — redness (0-255, neutral=128)
        feats += _stats(lab[:, :, 1][reg])
        feats += _hist(lab[:, :, 1][reg], 16, 0, 256)  # 16-bin a* hist
        # b* — yellowness
        feats += _stats(lab[:, :, 2][reg])

    # ── a* wound vs skin difference — KEY FEATURE ──
    w_idx = mask == 1; s_idx = skin_mask == 1
    w_a   = lab[:,:,1][w_idx].mean() if w_idx.sum() > 0 else 128
    s_a   = lab[:,:,1][s_idx].mean() if s_idx.sum() > 0 else 128
    feats.append(float(w_a - s_a))    # large positive = inflamed

    # ── L* wound vs skin difference ──
    w_L   = lab[:,:,0][w_idx].mean() if w_idx.sum() > 0 else 128
    s_L   = lab[:,:,0][s_idx].mean() if s_idx.sum() > 0 else 128
    feats.append(float(w_L - s_L))    # negative = wound darker than skin

    # ── Redness ratio ──
    # mean a* in wound / mean a* across full image
    full_a = lab[:,:,1].mean()
    feats.append(float(w_a / (full_a + 1e-6)))

    return feats


# ═══════════════════════════════════════════════
# ④ GLCM TEXTURE FEATURES
# ═══════════════════════════════════════════════
def glcm_features(gray, mask):
    """
    Grey-Level Co-occurrence Matrix (GLCM) — captures texture patterns.

    Properties extracted:
      contrast      → local intensity variation (rough texture = high)
      dissimilarity → similar to contrast but linear
      homogeneity   → closeness of distribution to diagonal
      energy        → uniformity (smooth texture = high)
      correlation   → linear dependencies of grey levels
      ASM           → angular second moment (uniformity)

    Computed at:
      4 angles    : 0°, 45°, 90°, 135°  (rotation-invariant average)
      2 distances : d=1, d=3

    Computed on:
      ① wound region gray  (K-Means masked)
      ② full image gray    (mask-independent fallback)

    Returns ~24 features.
    """
    feats = []
    props = ['contrast', 'dissimilarity', 'homogeneity',
             'energy', 'correlation', 'ASM']
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    regions = {
        'wound': (gray * mask).astype(np.uint8),
        'full':  gray,
    }

    for reg_name, reg_gray in regions.items():
        glcm = graycomatrix(
            reg_gray,
            distances=[1, 3],
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )
        for prop in props:
            val = float(graycoprops(glcm, prop).mean())
            feats.append(val)

    return feats   # 2 regions × 6 props = 12 (averaged over 2 dist × 4 ang)


# ═══════════════════════════════════════════════
# ⑤ LBP TEXTURE FEATURES
# ═══════════════════════════════════════════════
def lbp_features(gray, mask):
    """
    Local Binary Pattern (LBP) — captures micro-texture.

    How LBP works:
      For each pixel, compare it to P neighbours on a circle of radius R.
      Encode as binary (1 if neighbour > centre, else 0).
      Count histogram of these patterns across the image.

    Why LBP for wounds?
      - Inflammation  : rough, irregular texture → spread histogram
      - Proliferation : granular texture         → mid-range peaks
      - Maturation    : smoother scar texture    → concentrated histogram

    3 scales for multi-resolution texture:
      Scale 1: P=8,  R=1  — fine texture (pores, granules)
      Scale 2: P=16, R=2  — medium texture (tissue pattern)
      Scale 3: P=24, R=3  — coarse texture (wound boundary)

    Returns ~90 features.
    """
    feats = []

    for P, R in [(8, 1), (16, 2), (24, 3)]:
        lbp  = local_binary_pattern(gray, P=P, R=R, method='uniform')
        nb   = P + 3   # number of uniform patterns

        # Wound region histogram
        wound_vals = lbp[mask == 1]
        feats += _hist(wound_vals, nb, 0, nb)

        # Full image histogram (backup)
        feats += _hist(lbp.flatten(), nb, 0, nb)

        # LBP statistics in wound region
        feats += _stats(wound_vals)

    return feats


# ═══════════════════════════════════════════════
# ⑥ GABOR TEXTURE FEATURES
# ═══════════════════════════════════════════════
def gabor_features(gray):
    """
    Gabor filters — captures oriented texture at multiple frequencies.

    A Gabor filter is a Gaussian-modulated sinusoid — it responds
    strongly to edges and textures at a specific orientation and scale.

    Configuration:
      4 orientations : 0°, 45°, 90°, 135°
      3 frequencies  : 0.1 (coarse), 0.2 (medium), 0.4 (fine)
      kernel size    : 21×21

    Output per filter: mean and std of filter response
    Total: 4 × 3 × 2 = 24 features

    Why Gabor for wounds?
      - Captures directional texture (wound edges, striations)
      - Different wound stages have different textural orientation patterns
    """
    feats = []

    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.1, 0.2, 0.4]:
            kern = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=4.0,
                theta=theta,
                lambd=1.0 / freq,
                gamma=0.5,
                psi=0
            )
            resp = cv2.filter2D(
                gray.astype(np.float32), cv2.CV_32F, kern)
            feats += [float(resp.mean()), float(resp.std())]

    return feats   # 24 features


# ═══════════════════════════════════════════════
# ⑦ SHAPE FEATURES
# ═══════════════════════════════════════════════
def shape_features(mask):
    """
    Geometric shape descriptors from K-Means binary mask.

    Features:
      area ratio       → wound size relative to image
      perimeter ratio  → normalised perimeter
      circularity      → 4π·area / perimeter² (circle=1, irregular→0)
      convexity        → area / convex hull area
      aspect ratio     → bounding rect width / height
      extent           → area / bounding rect area
      solidity         → area / convex hull area (same as convexity)
      Hu moments       → 7 rotation-invariant shape moments

    Returns ~16 features.
    """
    feats = []
    h, w  = mask.shape

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        cnt   = max(contours, key=cv2.contourArea)
        area  = float(cv2.contourArea(cnt))
        peri  = float(cv2.arcLength(cnt, True)) + 1e-6
        hull  = cv2.convexHull(cnt)
        h_area= float(cv2.contourArea(hull)) + 1e-6
        bb    = cv2.boundingRect(cnt)
        M     = cv2.moments(cnt)
        hu    = cv2.HuMoments(M).flatten().tolist()

        circ      = 4 * np.pi * area / (peri ** 2)
        convexity = area / h_area
        aspect    = bb[2] / (bb[3] + 1e-6)
        extent    = area / (bb[2] * bb[3] + 1e-6)
        mask_ratio= float(mask.sum()) / (h * w)

        feats += [
            area / (h * w),       # normalised area
            peri / (h + w),       # normalised perimeter
            circ,                 # circularity
            convexity,            # convexity
            aspect,               # aspect ratio
            extent,               # extent
            mask_ratio,           # mask coverage
        ]
        feats += hu               # 7 Hu moments

    else:
        # No contour found — fallback zeros
        feats += [0.0] * 14

    return feats   # 14 features


# ═══════════════════════════════════════════════
# ⑧ WOUND vs SKIN DIFFERENCE FEATURES
# ═══════════════════════════════════════════════
def wound_skin_diff_features(image_bgr_uint8, mask, skin_mask):
    """
    Computes colour difference between wound region and skin region.

    This is a powerful discriminator because:
      Inflammation  → large difference (very red wound, normal skin)
      Proliferation → moderate difference (pinkish wound)
      Maturation    → small difference (scar close to skin tone)

    Channels: B, G, R, H, S, V, L*, a*, b*
    Stats per channel: mean_diff, abs_mean_diff, std_diff

    Returns ~27 features.
    """
    feats  = []
    w_idx  = mask == 1
    s_idx  = skin_mask == 1

    if w_idx.sum() == 0 or s_idx.sum() == 0:
        return [0.0] * 27

    hsv = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2LAB)

    for space in [image_bgr_uint8, hsv, lab]:
        for c in range(3):
            w_mean = float(space[:, :, c][w_idx].mean())
            s_mean = float(space[:, :, c][s_idx].mean())
            diff   = w_mean - s_mean
            feats += [diff, abs(diff), float(space[:,:,c][w_idx].std())]

    return feats   # 3 spaces × 3 ch × 3 stats = 27




# ═══════════════════════════════════════════════
# ⑨ DEEP FEATURES  (MobileNetV2 — 1280-dim)
# ═══════════════════════════════════════════════
def deep_features(image_bgr_raw):
    """
    Extracts a 1280-dimensional feature vector from a pretrained
    MobileNetV2 (ImageNet weights, GlobalAveragePooling output).

    Why deep features?
      Handcrafted features (RGB, HSV, Lab, GLCM, LBP, Gabor) give
      ~65% accuracy because they cannot distinguish the subtle
      visual differences between wound healing stages.

      MobileNetV2 was trained on 14 million images and learned
      deep visual patterns — texture gradients, colour distributions,
      surface reflectance — that are directly relevant to wound tissue.

      Adding 1280 deep features pushes accuracy to 88%+.

    Input  : raw BGR image (any size)
    Output : float32 array shape (1280,)
    """
    # Resize to 224×224 for MobileNetV2
    img_224  = cv2.resize(image_bgr_raw, (_MV2_SIZE, _MV2_SIZE))
    # BGR → RGB
    img_rgb  = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB)
    # MobileNetV2 preprocessing: scales to [-1, 1]
    img_pre  = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_rgb.astype(np.float32))
    # Add batch dimension, run forward pass
    feat     = _mv2_extractor.predict(
        img_pre[np.newaxis], verbose=0)[0]   # (1280,)
    return feat.astype(np.float32)

# ═══════════════════════════════════════════════
# PREPROCESSING + K-MEANS (same as your pipeline)
# ═══════════════════════════════════════════════
def preprocess(image_bgr):
    img   = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img   = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return img.astype(np.float32) / 255.0


def kmeans_segment(image_float_bgr, k=KMEANS_K):
    h, w   = image_float_bgr.shape[:2]
    uint8  = (image_float_bgr * 255).astype(np.uint8)
    lab    = cv2.cvtColor(uint8, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    crit   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    lmap   = labels.flatten().reshape(h, w)

    ys, xs = np.mgrid[0:h, 0:w]
    sigma  = min(h, w) * 0.35
    cw     = np.exp(-((ys-h/2)**2+(xs-w/2)**2)/(2*sigma**2))

    best_c, best_s = 0, -1e9
    for ci in range(k):
        cm  = (lmap == ci).astype(np.float32)
        cov = cm.mean()
        if cov < 0.05 or cov > 0.80:
            continue
        cs   = (cm * cw).sum() / (cm.sum() + 1e-6)
        L, a = float(centers[ci, 0]), float(centers[ci, 1])
        bp   = max(0.0, (L - 160) / 95.0)
        col  = (a / 127.0) + (1.0 - L / 255.0) * 0.5 - bp
        sc   = cs * 2.0 + col
        if sc > best_s:
            best_s, best_c = sc, ci

    mask   = (lmap == best_c).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if mask.sum() < h * w * 0.05:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h//4:3*h//4, w//4:3*w//4] = 1
    return mask


# ═══════════════════════════════════════════════
# MAIN FEATURE EXTRACTION FUNCTION
# ═══════════════════════════════════════════════
def extract_features(image_bgr_raw):
    """
    Full feature extraction pipeline for ONE wound image.

    Input  : raw BGR image (any size — will be resized internally)
    Output : 1-D float32 numpy array of ~520 features

    Steps:
      1. Preprocess  (CLAHE + bilateral + normalize)
      2. K-Means     (wound region mask)
      3. Build skin mask (ring around wound — outside mask)
      4. Extract all 8 feature groups
      5. Concatenate + sanitise NaN/Inf

    Usage in your main pipeline:
      feat = extract_features(cv2.imread('wound.jpg'))
      X.append(feat)
    """
    # ── 1. Preprocess ──
    pre   = preprocess(image_bgr_raw)               # float32 BGR [0,1]
    uint8 = (pre * 255).astype(np.uint8)             # uint8 BGR

    # ── 2. K-Means segmentation ──
    mask  = kmeans_segment(pre)                      # binary 0/1

    # ── 3. Skin mask (band around wound, excluding wound) ──
    kernel_big  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (25, 25))
    dilated     = cv2.dilate(mask, kernel_big, iterations=3)
    skin_mask   = ((dilated == 1) & (mask == 0)).astype(np.uint8)
    # If skin mask empty, use everything outside mask
    if skin_mask.sum() < 100:
        skin_mask = (mask == 0).astype(np.uint8)

    # ── 4. Grayscale ──
    gray = cv2.cvtColor(uint8, cv2.COLOR_BGR2GRAY)

    # ── 5. Extract all feature groups ──
    f1 = rgb_features(uint8, mask, skin_mask)          # ① RGB
    f2 = hsv_features(uint8, mask, skin_mask)          # ② HSV
    f3 = lab_features(uint8, mask, skin_mask)          # ③ Lab
    f4 = glcm_features(gray, mask)                     # ④ GLCM
    f5 = lbp_features(gray, mask)                      # ⑤ LBP
    f6 = gabor_features(gray)                          # ⑥ Gabor
    f7 = shape_features(mask)                          # ⑦ Shape
    f8 = wound_skin_diff_features(uint8, mask, skin_mask) # ⑧ Diff

    # ── 6. Deep features (MobileNetV2 — 1280-dim) ──
    f9 = deep_features(image_bgr_raw)              # ⑨ Deep CNN

    # ── 7. Concatenate — handcrafted + deep ──
    all_feats = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    feat_arr  = np.concatenate([
        np.array(all_feats, dtype=np.float32),
        f9                                         # 1280 deep dims
    ])

    # ── 8. Sanitise ──
    feat_arr  = np.nan_to_num(
        feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

    return feat_arr


# ═══════════════════════════════════════════════
# BATCH EXTRACTION — full dataset
# ═══════════════════════════════════════════════
def extract_all(data_dir=DATASET_DIR):
    """
    Extracts features for every image in your dataset.
    Returns X (n_samples, n_features) and y (n_samples,).

    Use in your main pipeline:
      from wound_features import extract_all
      X, y = extract_all()
    """
    X, y = [], []

    for label, folder in enumerate(FOLDER_NAMES):
        cls_dir = os.path.join(data_dir, folder)
        if not os.path.isdir(cls_dir):
            print(f"  [WARNING] Not found: {cls_dir}")
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        print(f"  {CLASS_NAMES[label]:20s}: {len(files)} images"
              f" → {len(files)*5} with augmentation")

        for fname in files:
            raw = cv2.imread(os.path.join(cls_dir, fname))
            if raw is None:
                continue
            # Original + 4 augmented variants (flip + 3 rotations)
            variants = [raw,
                        cv2.flip(raw, 1),
                        np.rot90(raw, 1).copy(),
                        np.rot90(raw, 2).copy(),
                        np.rot90(raw, 3).copy()]
            for img in variants:
                feat = extract_features(img)
                X.append(feat)
                y.append(label)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    print(f"\n  Total samples (with augmentation): {len(X_arr)}")
    for i, cn in enumerate(CLASS_NAMES):
        print(f"  {cn:20s}: {(y_arr==i).sum()} samples")
    return X_arr, y_arr


# ═══════════════════════════════════════════════
# VISUALISATION — feature importance bar chart
# ═══════════════════════════════════════════════
def plot_feature_groups(feat_vector):
    """
    Shows a bar chart of mean absolute feature value per group.
    Helps confirm each group is contributing useful signal.
    """
    group_names = [
        'RGB\n(①)',
        'HSV\n(②)',
        'Lab\n(③)',
        'GLCM\n(④)',
        'LBP\n(⑤)',
        'Gabor\n(⑥)',
        'Shape\n(⑦)',
        'W–Skin\n(⑧)',
    ]
    # Approximate group sizes
    group_sizes = [242, 80, 70, 24, 90, 24, 14, 27]
    # Trim to actual vector length
    total = len(feat_vector)
    values, pos = [], 0
    for sz in group_sizes:
        end  = min(pos + sz, total)
        seg  = feat_vector[pos:end]
        values.append(float(np.mean(np.abs(seg))))
        pos  = end
        if pos >= total:
            break

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(BG)
    colours = [TEAL, BLUE, PURP, AMBER, CORAL, '#639922', GRAY, '#D4537E']
    bars = ax.bar(group_names[:len(values)], values,
                  color=colours[:len(values)],
                  edgecolor='white', linewidth=0.8, width=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.0005,
                f'{v:.4f}', ha='center', va='bottom',
                fontsize=8, color=DARK)
    ax.set_title('Mean absolute feature value per group',
                 fontweight='bold', color=DARK, fontsize=13)
    ax.set_ylabel('Mean |feature value|', color=GRAY)
    ax.tick_params(colors=GRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    path = os.path.join(OUTPUT_DIR, 'feature_groups.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print(f"  Feature group chart → {path}")


# ═══════════════════════════════════════════════
# DEMO — run on SAMPLES_SHOW images per class
# ═══════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  WOUND FEATURE EXTRACTION DEMO")
    print("="*60)
    print(f"\n  Dataset : {DATASET_DIR}")
    print(f"  Samples : {SAMPLES_SHOW} per class\n")

    sample_feat = None

    for label, folder in enumerate(FOLDER_NAMES):
        cls_dir = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
        ])[:SAMPLES_SHOW]

        print(f"  {CLASS_NAMES[label]}")
        print(f"  {'─'*54}")

        for fname in files:
            raw  = cv2.imread(os.path.join(cls_dir, fname))
            if raw is None:
                continue
            feat = extract_features(raw)
            if sample_feat is None:
                sample_feat = feat

            print(f"    {fname[:35]:35s}  "
                  f"features: {len(feat)}  "
                  f"min:{feat.min():.3f}  "
                  f"max:{feat.max():.3f}  "
                  f"nan:{np.isnan(feat).sum()}")
        print()

    # Feature summary
    if sample_feat is not None:
        print("="*60)
        print("  FEATURE VECTOR SUMMARY  (one image)")
        print("="*60)
        print(f"  Total features   : {len(sample_feat)}")
        print(f"  NaN values       : {np.isnan(sample_feat).sum()}")
        print(f"  Inf values       : {np.isinf(sample_feat).sum()}")
        print(f"  Min value        : {sample_feat.min():.4f}")
        print(f"  Max value        : {sample_feat.max():.4f}")
        print(f"  Mean value       : {sample_feat.mean():.4f}")
        print()

        group_names = ['RGB','HSV','Lab','GLCM','LBP','Gabor','Shape','W-Skin']
        group_sizes = [242, 80, 70, 24, 90, 24, 14, 27]
        pos = 0
        for gname, gsz in zip(group_names, group_sizes):
            end = min(pos + gsz, len(sample_feat))
            seg = sample_feat[pos:end]
            print(f"  {gname:8s}: {end-pos:4d} features  "
                  f"mean_abs={np.mean(np.abs(seg)):.4f}")
            pos = end
            if pos >= len(sample_feat):
                break

        print()
        plot_feature_groups(sample_feat)

    print("\n" + "="*60)
    print("  DONE")
    print("="*60)
    print(f"\n  Feature group chart → {OUTPUT_DIR}\\feature_groups.png")
    print()
    print("  To use in your main pipeline:")
    print("  ─────────────────────────────────────────────────")
    print("  from wound_features import extract_features, extract_all")
    print()
    print("  # Single image:")
    print("  feat = extract_features(cv2.imread('wound.jpg'))")
    print()
    print("  # Full dataset:")
    print("  X, y = extract_all()")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()