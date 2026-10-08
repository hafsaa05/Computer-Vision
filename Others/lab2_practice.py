# =============================================================================
# LAB 02 — POINT TRANSFORMATIONS, HISTOGRAMS & MEDICAL IMAGING PIPELINE
# =============================================================================
# Topics Covered:
#   - Image Negative
#   - Log Transformation
#   - Power-Law (Gamma) Transformation
#   - Contrast Stretching
#   - Intensity-Level (Gray-Level) Slicing
#   - Bit-Plane Slicing
#   - Thresholding (Binary / ToZero / Otsu)
#   - Histogram Computation & Reading
#   - Histogram Equalization
#   - CLAHE (Contrast Limited Adaptive HE)
#   - Histogram Matching
#   - Color Mapping (False Color / Heatmap)
#   - Color Balance (Gray World)
#   - Weighted Fusion (addWeighted)
#   - Video Processing Loop
#
# Tasks:
#   - Task 1: Chest X-Ray Enhancement
#   - Task 2: Multi-Modal Cardiac Image Fusion (CT + MRI)
#   - Task 3: Real-Time Echocardiogram Video Analysis
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# cv2       → OpenCV: image & video processing
# numpy     → matrix and math operations
# matplotlib.pyplot → display images/plots (Jupyter-friendly)
# IPython.display   → for real-time frame animation in Jupyter
# skimage.exposure  → histogram matching
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from skimage.exposure import match_histograms


# =============================================================================
# =============================================================================
# SECTION A — CORE POINT TRANSFORMATIONS (THEORY FUNCTIONS)
# =============================================================================
# =============================================================================


# -----------------------------------------------------------------------------
# A.1 — IMAGE NEGATIVE
# -----------------------------------------------------------------------------
# FORMULA : s = (L - 1) - r = 255 - r
# WHY     : Reveals white/gray details inside dark regions.
# WHERE   : Mammograms, X-rays, dark-region analysis.
# PROS    : Extremely fast, single line.
# CONS    : Not adaptive; only inverts.
# PATTERN : 255 - img  (no arguments)
# -----------------------------------------------------------------------------
def image_negative(gray):
    """
    Input : gray  → uint8 grayscale image
    Output: negative image (uint8)
    """
    # STEP 1: Subtract each pixel from 255
    negative = 255 - gray
    return negative


# -----------------------------------------------------------------------------
# A.2 — LOG TRANSFORMATION
# -----------------------------------------------------------------------------
# FORMULA : s = c * log(1 + r)
#           c = 255 / log(1 + max(r))
# WHY     : Expands dark pixels, compresses bright → reveals faint dark details.
# WHERE   : Underexposed X-rays, Fourier spectrum, dark ultrasound chambers.
# PROS    : Great for dynamic range compression.
# CONS    : Requires float conversion; slow on large images.
# TRAP    : If r is uint8, (1 + 255) wraps to 0 → log(0) = -inf. ALWAYS float first.
# -----------------------------------------------------------------------------
def log_transform(gray):
    """
    Input : gray  → uint8 grayscale image
    Output: log-transformed image (uint8)
    """
    # STEP 1: Convert to float BEFORE any math (avoid uint8 overflow)
    img_float = gray.astype(np.float64)

    # STEP 2: Compute scaling constant c
    #         c = 255 / log(1 + max(r))
    c = 255 / np.log(1 + np.max(img_float))

    # STEP 3: Apply log formula: s = c * log(1 + r)
    log_img = c * np.log(1 + img_float)

    # STEP 4: Clip to valid range (safety) and convert back to uint8
    log_img = np.clip(log_img, 0, 255).astype(np.uint8)
    return log_img


# -----------------------------------------------------------------------------
# A.3 — POWER-LAW (GAMMA) TRANSFORMATION
# -----------------------------------------------------------------------------
# FORMULA : s = 255 * (r / 255) ^ gamma      (normalized form)
# WHY     : Tune brightness in a controllable way via gamma.
# WHERE   : Monitor correction, medical image tuning.
# PROS    : Flexible (whole family of curves via gamma).
# CONS    : Wrong gamma value ruins the image.
# RULE    : gamma < 1  → brightens (expands dark/mid tones)
#           gamma = 1  → no change
#           gamma > 1  → darkens  (expands bright tones)
# -----------------------------------------------------------------------------
def gamma_transform(gray, gamma):
    """
    Input : gray  → uint8 grayscale image
            gamma → float (e.g., 0.4, 0.8, 1.5)
    Output: gamma-corrected image (uint8)
    """
    # STEP 1: Normalize (divide by 255) → values 0..1
    # STEP 2: Apply power: (r/255) ^ gamma
    # STEP 3: Multiply back by 255
    # STEP 4: Clip and convert to uint8
    gamma_img = np.array(255 * (gray / 255.0) ** gamma, dtype=np.uint8)
    return gamma_img


# -----------------------------------------------------------------------------
# A.4 — CONTRAST STRETCHING (MIN-MAX)
# -----------------------------------------------------------------------------
# FORMULA : s = (r - r_min) / (r_max - r_min) * 255
# WHY     : Spreads pixel values from narrow range → full 0-255.
# WHERE   : Low-contrast satellite/medical images.
# PATTERN : cv2.normalize(src, dst, alpha, beta, norm_type)
#   - src         → input image
#   - dst         → output (None = allocate new)
#   - alpha       → lower bound of output range (0)
#   - beta        → upper bound of output range (255)
#   - norm_type   → cv2.NORM_MINMAX
# -----------------------------------------------------------------------------
def contrast_stretch(gray):
    """
    Input : gray → uint8 grayscale image
    Output: stretched image (uint8) spanning 0-255
    """
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


# -----------------------------------------------------------------------------
# A.5 — INTENSITY-LEVEL (GRAY-LEVEL) SLICING
# -----------------------------------------------------------------------------
# WHY    : Highlights a specific intensity range.
# WHERE  : Highlight tissues, water bodies, blood vessels.
# TWO MODES:
#   Mode 1: range → white (255), else → black (0)    → binary output
#   Mode 2: range → white (255), else → unchanged    → keeps background
# -----------------------------------------------------------------------------
def intensity_slice(gray, A, B, keep_background=True):
    """
    Input : gray             → uint8 grayscale image
            A                → lower intensity bound
            B                → upper intensity bound
            keep_background  → bool; True = Mode 2, False = Mode 1
    Output: sliced image (uint8)
    """
    # STEP 1: Create boolean mask of pixels within [A, B]
    mask = (gray >= A) & (gray <= B)

    if keep_background:
        # Mode 2: within range → 255, else keep original
        return np.where(mask, 255, gray).astype(np.uint8)
    else:
        # Mode 1: within range → 255, else → 0
        return np.where(mask, 255, 0).astype(np.uint8)


# -----------------------------------------------------------------------------
# A.6 — BIT-PLANE SLICING
# -----------------------------------------------------------------------------
# WHY    : Split 8-bit image into 8 binary images (bit planes).
#          Bit 7 (MSB) → shape; Bit 0 (LSB) → noise.
# WHERE  : Compression, watermarking, steganography.
# PATTERN: plane = ((gray >> k) & 1) * 255
# -----------------------------------------------------------------------------
def bit_plane_slice(gray, bit_index):
    """
    Input : gray      → uint8 grayscale image
            bit_index → 0..7 (0 = LSB, 7 = MSB)
    Output: binary image showing that bit plane (uint8)
    """
    # STEP 1: Shift right by bit_index, mask last bit, scale to 0/255
    plane = ((gray >> bit_index) & 1) * 255
    return plane.astype(np.uint8)


# -----------------------------------------------------------------------------
# A.7 — THRESHOLDING
# -----------------------------------------------------------------------------
# PATTERN : cv2.threshold(src, thresh, maxval, type)
#   - src     → input grayscale (uint8)
#   - thresh  → cutoff value T
#   - maxval  → value assigned when pixel > T (usually 255)
#   - type    → cv2.THRESH_BINARY | THRESH_TOZERO | THRESH_BINARY + THRESH_OTSU
#
# TYPE COMPARISON:
#   THRESH_BINARY   : below T → 0,       above T → 255
#   THRESH_TOZERO   : below T → 0,       above T → keeps value
#   THRESH_OTSU     : auto-picks T (must be combined with BINARY)
#
# RETURNS : (threshold_used, thresholded_image)
# -----------------------------------------------------------------------------
def apply_threshold(gray, T=200, mode="binary"):
    """
    Input : gray → uint8 grayscale image
            T    → threshold value (0-255)
            mode → 'binary' | 'tozero' | 'otsu'
    Output: thresholded image (uint8)
    """
    if mode == "binary":
        _, out = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    elif mode == "tozero":
        _, out = cv2.threshold(gray, T, 255, cv2.THRESH_TOZERO)
    elif mode == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("mode must be 'binary', 'tozero', or 'otsu'")
    return out


# =============================================================================
# =============================================================================
# SECTION B — HISTOGRAM FUNCTIONS
# =============================================================================
# =============================================================================


# -----------------------------------------------------------------------------
# B.1 — COMPUTE HISTOGRAM
# -----------------------------------------------------------------------------
# PATTERN : cv2.calcHist(images, channels, mask, histSize, ranges)
#   - images    → [gray]              ⚠ MUST be inside a list
#   - channels  → [0]                 (0 for gray; 0/1/2 for B/G/R)
#   - mask      → None                (None = whole image)
#   - histSize  → [256]               (number of bins)
#   - ranges    → [0, 256]            (upper exclusive)
# RETURNS : hist → shape (256, 1) array
# -----------------------------------------------------------------------------
def compute_histogram(gray):
    """
    Input : gray → uint8 grayscale image
    Output: 256-bin histogram as numpy array
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist


# -----------------------------------------------------------------------------
# B.2 — HISTOGRAM EQUALIZATION
# -----------------------------------------------------------------------------
# FORMULA : s_k = (L - 1) * sum(p(r_j))  =  255 * CDF
# WHY     : Auto-boost contrast; spreads pixels uniformly.
# WHERE   : Underexposed X-rays, murky ultrasound.
# PATTERN : cv2.equalizeHist(gray)
#   - gray → uint8 grayscale ONLY (no color, no float)
# -----------------------------------------------------------------------------
def histogram_equalization(gray):
    """
    Input : gray → uint8 grayscale image
    Output: equalized image (uint8)
    """
    return cv2.equalizeHist(gray)


# -----------------------------------------------------------------------------
# B.3 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
# -----------------------------------------------------------------------------
# WHY    : Divides image into tiles → equalizes each → clips to limit noise.
# WHERE  : Medical images (better than global equalization).
# PATTERN: cv2.createCLAHE(clipLimit, tileGridSize)
#   - clipLimit     → float (e.g., 2.0) controls contrast limiting
#   - tileGridSize  → (rows, cols) tile grid (e.g., (8, 8))
# -----------------------------------------------------------------------------
def apply_clahe(gray, clip=2.0, tile=(8, 8)):
    """
    Input : gray → uint8 grayscale image
            clip → clip limit (float)
            tile → tile grid size (tuple)
    Output: CLAHE-enhanced image (uint8)
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(gray)


# -----------------------------------------------------------------------------
# B.4 — HISTOGRAM MATCHING (SPECIFICATION)
# -----------------------------------------------------------------------------
# WHY    : Match src's histogram to reference's histogram.
# PATTERN: match_histograms(src, reference)
# -----------------------------------------------------------------------------
def histogram_match(src, reference):
    """
    Input : src       → image whose histogram will be modified
            reference → image whose histogram will be matched
    Output: matched image
    """
    return match_histograms(src, reference)


# =============================================================================
# =============================================================================
# SECTION C — COLOR MAP, COLOR BALANCE, FUSION
# =============================================================================
# =============================================================================


# -----------------------------------------------------------------------------
# C.1 — COLOR MAPPING (FALSE COLOR / HEATMAP)
# -----------------------------------------------------------------------------
# WHY    : Human eye sees thousands of colors but only ~dozen grays.
# WHERE  : Fluid boundaries, blood flow, medical visualization.
# PATTERN: cv2.applyColorMap(gray_uint8, colormap)
#   - gray_uint8  → 8-bit grayscale image
#   - colormap    → cv2.COLORMAP_JET | COLORMAP_BONE | etc.
# OUTPUT : 3-channel BGR image
# -----------------------------------------------------------------------------
def apply_heatmap(gray, colormap=cv2.COLORMAP_JET):
    """
    Input : gray     → uint8 grayscale
            colormap → cv2 colormap constant
    Output: 3-channel BGR heatmap
    """
    # STEP 1: Apply colormap
    heatmap_bgr = cv2.applyColorMap(gray, colormap)
    return heatmap_bgr


# -----------------------------------------------------------------------------
# C.2 — COLOR BALANCE (GRAY WORLD)
# -----------------------------------------------------------------------------
# WHY    : Removes unwanted color casts.
# ASSUMPTION: average scene should be gray → each channel's mean equal.
# PATTERN:
#   1) Convert to float
#   2) Split B, G, R
#   3) Compute average of channel means
#   4) Scale each channel by (avg / channel_mean)
#   5) Clip to 0-255 and merge
# -----------------------------------------------------------------------------
def gray_world_balance(img_bgr):
    """
    Input : img_bgr → 3-channel BGR image (uint8)
    Output: color-balanced image (uint8, BGR)
    """
    # STEP 1: Convert to float to allow multiplication
    img = img_bgr.astype(np.float32)

    # STEP 2: Split channels (B, G, R order in OpenCV)
    b, g, r = cv2.split(img)

    # STEP 3: Compute average of the three channel means
    avg = (b.mean() + g.mean() + r.mean()) / 3

    # STEP 4: Scale each channel by ratio avg / channel_mean
    b *= avg / b.mean()
    g *= avg / g.mean()
    r *= avg / r.mean()

    # STEP 5: Merge back, clip to 0-255, cast to uint8
    merged = cv2.merge([b, g, r])
    return np.clip(merged, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# C.3 — WEIGHTED FUSION (addWeighted)
# -----------------------------------------------------------------------------
# FORMULA : dst = alpha * img1 + beta * img2 + gamma
# WHY     : Combine complementary info from two images.
# WHERE   : CT + MRI fusion, multi-modal imaging.
# PATTERN : cv2.addWeighted(img1, alpha, img2, beta, gamma)
#   - img1   → first image
#   - alpha  → weight of img1 (float)
#   - img2   → second image
#   - beta   → weight of img2 (float)
#   - gamma  → brightness offset (usually 0)
# RULE    : alpha + beta = 1   → keeps brightness balanced.
# -----------------------------------------------------------------------------
def weighted_fusion(img1, alpha, img2, beta, gamma=0):
    """
    Input : img1, img2 → same size + same channels
            alpha, beta → weights (float)
            gamma       → brightness offset
    Output: fused image (uint8)
    """
    return cv2.addWeighted(img1, alpha, img2, beta, gamma)


# =============================================================================
# =============================================================================
# TASK 1 — CHEST X-RAY ENHANCEMENT
# =============================================================================
# SCENARIO : Underexposed X-rays → lung cavities invisible.
# FLOW     : Load → Display → Equalize → Heatmap → Balance
#            → Threshold → Log → Gamma → Display.
# MNEMONIC : "Lazy Dogs Eat Hot Burgers Then Logout Gaming"
# =============================================================================


def task1_chest_xray(image_path='data/sample_xray.png'):
    # -------------------------------------------------------------------------
    # STEP 1: LOAD GRAYSCALE IMAGE
    # -------------------------------------------------------------------------
    # cv2.imread(path, flag)
    #   - path = file path
    #   - flag = cv2.IMREAD_GRAYSCALE → forces single-channel (0-255) uint8
    xray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # -------------------------------------------------------------------------
    # STEP 2: DISPLAY RAW X-RAY
    # -------------------------------------------------------------------------
    # plt.imshow(img, cmap='gray')  → for 1-channel images
    # plt.axis('off')                → hide tick marks
    plt.figure(figsize=(6, 6))
    plt.imshow(xray, cmap='gray')
    plt.title('Raw X-Ray')
    plt.axis('off')
    plt.show()

    # -------------------------------------------------------------------------
    # STEP 3: HISTOGRAM EQUALIZATION
    # -------------------------------------------------------------------------
    # cv2.equalizeHist(gray) → spreads intensities, boosts contrast
    equalized = cv2.equalizeHist(xray)

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(xray, cmap='gray')
    axes[0].set_title('Raw X-Ray'); axes[0].axis('off')
    axes[1].imshow(equalized, cmap='gray')
    axes[1].set_title('After Histogram Equalization'); axes[1].axis('off')
    plt.tight_layout(); plt.show()

    # -------------------------------------------------------------------------
    # STEP 4: FALSE-COLOR HEATMAP (JET)
    # -------------------------------------------------------------------------
    # cv2.applyColorMap(gray, COLORMAP_JET) → 3-channel BGR heatmap
    heatmap = cv2.applyColorMap(equalized, cv2.COLORMAP_JET)

    # Convert BGR → RGB because matplotlib expects RGB
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap_rgb)
    plt.title('JET Heatmap (Color Mapped)')
    plt.axis('off')
    plt.show()

    # -------------------------------------------------------------------------
    # STEP 5: COLOR BALANCE CORRECTION
    # -------------------------------------------------------------------------
    # 1) Convert to float (so multiplication doesn't overflow uint8)
    # 2) Channel 0 = Red, Channel 2 = Blue (RGB order)
    # 3) Multiply and clip
    balanced = heatmap_rgb.copy().astype(np.float32)
    balanced[:, :, 0] *= 1.1   # slight red boost
    balanced[:, :, 2] *= 0.9   # slight blue reduction
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(balanced)
    plt.title('After Color Balance Correction')
    plt.axis('off')
    plt.show()

    # -------------------------------------------------------------------------
    # STEP 6: THRESHOLDING (DENSE TISSUE)
    # -------------------------------------------------------------------------
    # cv2.threshold(src, T, maxval, type)
    #   - src    = equalized
    #   - T      = 200 (dense tissue is bright)
    #   - maxval = 255
    #   - type   = cv2.THRESH_BINARY → pixel > 200 → 255; else 0
    _, dense_mask = cv2.threshold(equalized, 200, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(6, 6))
    plt.imshow(dense_mask, cmap='gray')
    plt.title('Dense Tissue Mask (Threshold > 200)')
    plt.axis('off')
    plt.show()

    # -------------------------------------------------------------------------
    # STEP 7: LOG TRANSFORMATION (REVEAL DARK BACKGROUND)
    # -------------------------------------------------------------------------
    # Formula: s = c * log(1 + r),   c = 255 / log(1 + max(r))
    # IMPORTANT: convert to float BEFORE log to avoid uint8 overflow
    c = 255 / np.log(1 + np.max(xray.astype(np.float64)))
    log_transformed = c * np.log(1 + xray.astype(np.float64))
    log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(log_transformed, cmap='gray')
    plt.title('Log Transformation (Dark Regions Enhanced)')
    plt.axis('off')
    plt.show()

    # -------------------------------------------------------------------------
    # STEP 8: GAMMA CORRECTION (γ < 1 BRIGHTENS MIDTONES)
    # -------------------------------------------------------------------------
    # Formula: s = 255 * (r / 255) ^ gamma
    # γ < 1 → brightens, expands dark/mid tones
    gamma = 0.4
    gamma_corrected = np.array(255 * (xray / 255.0) ** gamma, dtype=np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(xray, cmap='gray')
    axes[0].set_title('Raw X-Ray'); axes[0].axis('off')
    axes[1].imshow(gamma_corrected, cmap='gray')
    axes[1].set_title(f'Gamma Corrected (γ = {gamma})'); axes[1].axis('off')
    plt.tight_layout(); plt.show()


# =============================================================================
# =============================================================================
# TASK 2 — MULTI-MODAL CARDIAC IMAGE FUSION (CT + MRI)
# =============================================================================
# SCENARIO : CT = structure, MRI = soft tissue → fuse into one view.
# FLOW     : Load → Resize → Equalize → ColorMap → BGR→RGB
#            → Fuse → Log → Gamma → Show.
# KEY RULE : alpha + beta = 1   (balanced brightness)
# =============================================================================


def task2_cardiac_fusion(ct_path='data/ct.jpg', mri_path='data/mri.jpg'):
    # -------------------------------------------------------------------------
    # STEP 1: LOAD BOTH MODALITIES AS GRAYSCALE
    # -------------------------------------------------------------------------
    ct  = cv2.imread(ct_path,  cv2.IMREAD_GRAYSCALE)
    mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)

    # -------------------------------------------------------------------------
    # STEP 2: RESIZE TO SAME SIZE (required for fusion)
    # -------------------------------------------------------------------------
    # cv2.resize(src, dsize)   → dsize = (width, height)  ⚠ width first!
    ct  = cv2.resize(ct,  (400, 400))
    mri = cv2.resize(mri, (400, 400))

    # -------------------------------------------------------------------------
    # STEP 3: HISTOGRAM EQUALIZATION (BOTH INDEPENDENTLY)
    # -------------------------------------------------------------------------
    # Each modality has a different intensity distribution → equalize separately
    ct_eq  = cv2.equalizeHist(ct)
    mri_eq = cv2.equalizeHist(mri)

    # -------------------------------------------------------------------------
    # STEP 4: COLOR MAP (DIFFERENT MAP FOR EACH MODALITY)
    # -------------------------------------------------------------------------
    # CT  → COLORMAP_BONE (white-blue tones: keeps structural look)
    # MRI → COLORMAP_JET  (colorful: highlights tissue variation)
    ct_color  = cv2.applyColorMap(ct_eq,  cv2.COLORMAP_BONE)
    mri_color = cv2.applyColorMap(mri_eq, cv2.COLORMAP_JET)

    # -------------------------------------------------------------------------
    # STEP 5: BGR → RGB CONVERSION (for matplotlib display & fusion)
    # -------------------------------------------------------------------------
    ct_color_rgb  = cv2.cvtColor(ct_color,  cv2.COLOR_BGR2RGB)
    mri_color_rgb = cv2.cvtColor(mri_color, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------------------
    # STEP 6: WEIGHTED FUSION
    # -------------------------------------------------------------------------
    # cv2.addWeighted(img1, alpha, img2, beta, gamma)
    #   - CT heavier (0.7) → preserve sharp edges
    #   - MRI lighter (0.3) → highlight soft tissue
    #   - alpha + beta = 1  → brightness balanced
    fused = cv2.addWeighted(ct_color_rgb, 0.7, mri_color_rgb, 0.3, 0)

    # -------------------------------------------------------------------------
    # STEP 7: LOG TRANSFORMATION (RECOVER DARK AREAS)
    # -------------------------------------------------------------------------
    # Convert to float BEFORE max() → avoids uint8 overflow
    fused_float = fused.astype(np.float64)
    c = 255 / np.log(1 + np.max(fused_float))
    log_fused = c * np.log(1 + fused_float)
    log_fused = np.clip(log_fused, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # STEP 8: GAMMA CORRECTION (MILD BRIGHTENING)
    # -------------------------------------------------------------------------
    # gamma = 0.8 → slightly brighten midtones; not blowing out highlights
    gamma = 0.8
    gamma_fused = np.array(255 * (log_fused / 255.0) ** gamma, dtype=np.uint8)

    # -------------------------------------------------------------------------
    # STEP 9: SIDE-BY-SIDE COMPARISON (CT | MRI | FUSED)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ct_color_rgb)
    axes[0].set_title('CT (Equalized + BONE)'); axes[0].axis('off')

    axes[1].imshow(mri_color_rgb)
    axes[1].set_title('MRI (Equalized + JET)'); axes[1].axis('off')

    axes[2].imshow(gamma_fused)
    axes[2].set_title('Fused Output (CT + MRI)'); axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# =============================================================================
# =============================================================================
# TASK 3 — REAL-TIME ECHOCARDIOGRAM VIDEO ANALYSIS
# =============================================================================
# SCENARIO : Ultrasound video = noisy, murky, low-contrast.
# FLOW     : Open → Check → Figure → Loop(Read → Check → Gray → Equalize
#            → Heatmap → Balance → Log → Gamma → Display) → Release.
# MNEMONIC : "Read → Check → Process → Break → Release"
# =============================================================================


def task3_echo_video(video_path='data/echo.mp4'):
    # -------------------------------------------------------------------------
    # STEP 1: INITIALIZE VIDEO CAPTURE
    # -------------------------------------------------------------------------
    # cv2.VideoCapture(source)
    #   - source = file path ('echo.mp4')  OR  0 (webcam)
    cap = cv2.VideoCapture(video_path)

    # -------------------------------------------------------------------------
    # STEP 2: CHECK IF VIDEO OPENED SUCCESSFULLY
    # -------------------------------------------------------------------------
    if not cap.isOpened():
        print("Error: Video not found.")
        return

    # -------------------------------------------------------------------------
    # STEP 3: SETUP FIGURE (ONCE — outside the loop)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -------------------------------------------------------------------------
    # STEP 4: FRAME-BY-FRAME PROCESSING LOOP
    # -------------------------------------------------------------------------
    while True:
        # cap.read() → returns (ret, frame)
        #   ret   = True if frame read successfully
        #   frame = the image (BGR, 3-channel)
        ret, frame = cap.read()

        # STEP 4a: Break at end of video (frame would be None)
        if not ret:
            break

        # ---------------------------------------------------------------------
        # STEP 4b: CONVERT FRAME BGR → GRAYSCALE
        # ---------------------------------------------------------------------
        # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) → single-channel uint8
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------------------------
        # STEP 4c: HISTOGRAM EQUALIZATION
        # ---------------------------------------------------------------------
        # Combat murky ultrasound contrast
        equalized = cv2.equalizeHist(gray)

        # ---------------------------------------------------------------------
        # STEP 4d: COLOR MAPPING (BLOOD FLOW HEATMAP)
        # ---------------------------------------------------------------------
        heatmap = cv2.applyColorMap(equalized, cv2.COLORMAP_JET)

        # ---------------------------------------------------------------------
        # STEP 4e: COLOR BALANCE ADJUSTMENT
        # ---------------------------------------------------------------------
        # Convert to float → tweak channels → clip → uint8
        # BGR order: channel 0 = Blue, channel 2 = Red
        balanced = heatmap.astype(np.float32)
        balanced[:, :, 0] *= 0.9   # reduce blue
        balanced[:, :, 2] *= 1.1   # boost red
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)

        # ---------------------------------------------------------------------
        # STEP 4f: LOG TRANSFORMATION (REVEAL DARK CHAMBERS)
        # ---------------------------------------------------------------------
        # Float conversion first → avoids uint8 overflow
        balanced_float = balanced.astype(np.float64)
        c = 255 / np.log(1 + np.max(balanced_float))
        log_img = c * np.log(1 + balanced_float)
        log_img = np.clip(log_img, 0, 255).astype(np.uint8)

        # ---------------------------------------------------------------------
        # STEP 4g: GAMMA CORRECTION (SUPPRESS BRIGHT BACKSCATTER NOISE)
        # ---------------------------------------------------------------------
        # For SUPPRESSING bright noise → use gamma > 1 (darkens bright regions)
        gamma = 1.5
        gamma_img = np.array(255 * (log_img / 255.0) ** gamma, dtype=np.uint8)

        # ---------------------------------------------------------------------
        # STEP 4h: FINAL COLOR MAP + BGR → RGB
        # ---------------------------------------------------------------------
        enhanced_final = cv2.applyColorMap(gamma_img, cv2.COLORMAP_JET)
        enhanced_rgb   = cv2.cvtColor(enhanced_final, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------------------
        # STEP 4i: UPDATE DISPLAY (ANIMATION)
        # ---------------------------------------------------------------------
        # Clear previous frame → draw new one → refresh Jupyter cell output
        axes[0].clear()
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title('Raw Feed'); axes[0].axis('off')

        axes[1].clear()
        axes[1].imshow(enhanced_rgb)
        axes[1].set_title('Enhanced Feed'); axes[1].axis('off')

        clear_output(wait=True)
        display(fig)

    # -------------------------------------------------------------------------
    # STEP 5: CLEANUP
    # -------------------------------------------------------------------------
    # cap.release() → frees the video file handle (MUST)
    # plt.close()   → closes matplotlib figure
    cap.release()
    plt.close()
    print("Video finished.")


# =============================================================================
# =============================================================================
# OPTIONAL — COLOR IMAGE HISTOGRAM EQUALIZATION (TRAP DEMO)
# =============================================================================
# ❌ Do NOT equalize B, G, R separately → colors distort.
# ✅ Convert to YCrCb → equalize Y (brightness) only → convert back.
# =============================================================================

def equalize_color_image(img_bgr):
    """
    Input : img_bgr → 3-channel BGR image
    Output: brightness-equalized BGR image (uint8)
    """
    # STEP 1: BGR → YCrCb
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # STEP 2: Equalize Y (brightness) channel only
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    # STEP 3: Convert back to BGR
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# =============================================================================
# =============================================================================
# MAIN — RUN TASKS
# =============================================================================
# Uncomment the task you want to run.
# =============================================================================
if __name__ == "__main__":
    # task1_chest_xray('data/sample_xray.png')
    # task2_cardiac_fusion('data/ct.jpg', 'data/mri.jpg')
    # task3_echo_video('data/echo.mp4')
    pass
