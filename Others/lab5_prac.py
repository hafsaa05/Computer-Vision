# =============================================================================
#  LAB 05 — IMAGE SEGMENTATION  (COMPLETE REFERENCE SCRIPT)
# =============================================================================
#  Libraries : OpenCV (cv2), NumPy (np), Matplotlib (plt)
#  Topics    : Thresholding (Global, Adaptive, Otsu), HSV Color, Canny,
#              Region Growing, Watershed, K-Means
#  Tasks     : 1 to 10 (mapped below)
# =============================================================================

# -----------------------------------------------------------------------------
#  IMPORT SECTION
# -----------------------------------------------------------------------------
import cv2                          # image I/O, processing, segmentation
import numpy as np                  # arrays, reshaping, float32, kernels
import matplotlib.pyplot as plt     # display (needs RGB, not BGR)


# =============================================================================
#  GLOBAL HELPER: DISPLAY FUNCTION
# =============================================================================
#  Function : show(title, image, cmap)
#  Args     : title  → string heading for the plot
#             image  → the array to display
#             cmap   → "gray" for single-channel images, None for color
#  Order    : title must come first, image second, cmap third.
#  Mandatory: title and image are mandatory; cmap is optional.
# =============================================================================
def show(title, image, cmap=None):
    plt.figure(figsize=(6, 4))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 1 — GLOBAL vs ADAPTIVE THRESHOLDING (Uneven Lighting)
# =============================================================================
#  GOAL: Show that a single global threshold fails when lighting varies,
#        and adaptive thresholding fixes it by using local neighbourhoods.
# =============================================================================

# STEP 1 : READ IMAGE directly as grayscale.
#   Function : cv2.imread(filename, flag)
#   Args     : filename → path (mandatory)
#              flag     → cv2.IMREAD_GRAYSCALE loads as 1-channel (optional)
#   Returns  : NumPy array of shape (H, W) for grayscale.
img1 = cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE)

# STEP 2 : SAFETY CHECK (image missing → None)
if img1 is None:
    print("Error: sudoku.png not found")
else:
    # STEP 3 : GLOBAL THRESHOLD at three different T values.
    #   Function : cv2.threshold(src, thresh, maxval, type)
    #   Args     : src    → grayscale (mandatory)
    #              thresh → threshold value T (mandatory)
    #              maxval → value assigned to passing pixels, usually 255 (mandatory)
    #              type   → THRESH_BINARY : pixel > T → maxval, else 0 (mandatory)
    #   Returns  : (used_threshold, binary_image)
    _, g80  = cv2.threshold(img1, 80,  255, cv2.THRESH_BINARY)
    _, g127 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    _, g180 = cv2.threshold(img1, 180, 255, cv2.THRESH_BINARY)

    # STEP 4 : ADAPTIVE THRESHOLD.
    #   Function : cv2.adaptiveThreshold(src, maxValue, adaptiveMethod,
    #                                    thresholdType, blockSize, C)
    #   Args     : src            → grayscale (mandatory)
    #              maxValue       → 255 (mandatory)
    #              adaptiveMethod → MEAN_C or GAUSSIAN_C (mandatory)
    #              thresholdType  → THRESH_BINARY (mandatory)
    #              blockSize      → odd neighbourhood size (mandatory)
    #              C              → constant subtracted from local mean (mandatory)
    #   Returns  : binary image (NOT a tuple).
    adaptive1 = cv2.adaptiveThreshold(
        img1, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,     # Gaussian-weighted local mean
        cv2.THRESH_BINARY,
        11,                                  # blockSize (must be odd)
        2                                    # C
    )

    # STEP 5 : DISPLAY PIPELINE : Original → T=80 → T=127 → T=180 → Adaptive
    titles1 = ["Original", "Global T=80", "Global T=127", "Global T=180", "Adaptive"]
    images1 = [img1, g80, g127, g180, adaptive1]

    plt.figure(figsize=(18, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images1[i], cmap="gray")     # single-channel → cmap="gray"
        plt.title(titles1[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 2 — TUNING ADAPTIVE THRESHOLD (blockSize + C)
# =============================================================================
#  GOAL: Show the effect of blockSize and C, and compare MEAN vs GAUSSIAN.
# =============================================================================

img2 = cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE)

if img2 is None:
    print("Error: sudoku.png not found")
else:
    results2 = []
    titles2  = []

    # --- Loop 1 : Vary blockSize with MEAN_C (C fixed at 2) ---
    #   Step order : read → loop over blockSize → adaptiveThreshold → store
    for b in [5, 15, 31]:
        out = cv2.adaptiveThreshold(
            img2, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,     # MEAN: every neighbour equal weight
            cv2.THRESH_BINARY,
            b,                              # blockSize varies
            2                               # C fixed
        )
        results2.append(out)
        titles2.append("Mean, block=" + str(b) + ", C=2")

    # --- Loop 2 : Vary C with GAUSSIAN_C (blockSize fixed at 15) ---
    for c in [2, 5, 10]:
        out = cv2.adaptiveThreshold(
            img2, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # GAUSSIAN: centre pixel weighted more
            cv2.THRESH_BINARY,
            15,                              # blockSize fixed
            c                                # C varies
        )
        results2.append(out)
        titles2.append("Gaussian, block=15, C=" + str(c))

    # --- Display : 3x3 grid, Original in centre top ---
    plt.figure(figsize=(15, 9))

    plt.subplot(3, 3, 2)                    # original at top-centre
    plt.imshow(img2, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    for i in range(6):
        plt.subplot(3, 3, i + 4)            # positions 4..9 for the 6 results
        plt.imshow(results2[i], cmap="gray")
        plt.title(titles2[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 3 — OTSU'S THRESHOLDING (Automatic Threshold)
# =============================================================================
#  GOAL: Let Otsu auto-pick the best threshold; verify via histogram.
# =============================================================================

img3 = cv2.imread("water_coins.jpg", cv2.IMREAD_GRAYSCALE)

if img3 is None:
    print("Error: water_coins.jpg not found")
else:
    # STEP 1 : OTSU threshold.
    #   Combine THRESH_BINARY + THRESH_OTSU in the 'type' argument.
    #   The 'thresh' value (0 here) is IGNORED when OTSU is used.
    t1, otsu1 = cv2.threshold(img3, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu chose threshold:", t1)

    # STEP 2 : Change CONTRAST with convertScaleAbs.
    #   Function : cv2.convertScaleAbs(src, alpha, beta)
    #   Args     : alpha → contrast multiplier (>1 → more contrast)
    #              beta  → brightness offset (added after multiply)
    #   Formula  : output = |alpha * input + beta|
    changed = cv2.convertScaleAbs(img3, alpha=1.5, beta=20)

    # STEP 3 : Re-run Otsu on the altered image.
    t2, otsu2 = cv2.threshold(changed, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("After contrast change, Otsu chose:", t2)

    # STEP 4 : DISPLAY : Original → Histogram → Otsu mask (and repeat row)
    plt.figure(figsize=(16, 8))

    # Row 1
    plt.subplot(2, 3, 1)
    plt.imshow(img3, cmap="gray");  plt.title("Original");           plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.hist(img3.ravel(), 256, [0, 256])   # ravel() flattens for histogram
    plt.axvline(t1, color="red")            # red vertical line = Otsu threshold
    plt.title("Histogram (red = Otsu T=" + str(int(t1)) + ")")

    plt.subplot(2, 3, 3)
    plt.imshow(otsu1, cmap="gray"); plt.title("Otsu Binary Mask");   plt.axis("off")

    # Row 2 (after contrast change)
    plt.subplot(2, 3, 4)
    plt.imshow(changed, cmap="gray"); plt.title("Contrast Changed"); plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.hist(changed.ravel(), 256, [0, 256])
    plt.axvline(t2, color="red")
    plt.title("Histogram (red = Otsu T=" + str(int(t2)) + ")")

    plt.subplot(2, 3, 6)
    plt.imshow(otsu2, cmap="gray")
    plt.title("Otsu After Contrast Change"); plt.axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 4 — COLOR-BASED SEGMENTATION (HSV Masking)
# =============================================================================
#  GOAL: Extract a colored object using HSV range thresholding.
# =============================================================================

img4 = cv2.imread("smarties.png")               # color load (BGR)

if img4 is None:
    print("Error: smarties.png not found")
else:
    # STEP 1 : BGR → HSV.
    #   Function : cv2.cvtColor(src, code)
    #   Args     : src  → BGR image (mandatory)
    #              code → cv2.COLOR_BGR2HSV (mandatory)
    #   Note     : OpenCV Hue range is 0–179 (NOT 0–360).
    hsv = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)

    # STEP 2 : TIGHT mask (too restrictive).
    #   Function : cv2.inRange(src, lowerb, upperb)
    #   Args     : src    → HSV image (mandatory)
    #              lowerb → NumPy array [H, S, V] minimum (mandatory)
    #              upperb → NumPy array [H, S, V] maximum (mandatory)
    #   Returns  : binary mask (255 inside range, 0 outside).
    tight_low  = np.array([0, 200, 200])
    tight_high = np.array([5, 255, 255])
    mask_tight = cv2.inRange(hsv, tight_low, tight_high)

    # STEP 3 : WIDE mask (accepts shadows/highlights).
    wide_low  = np.array([0, 100, 80])
    wide_high = np.array([10, 255, 255])
    mask_wide = cv2.inRange(hsv, wide_low, wide_high)

    # STEP 4 : EXTRACT using the wide mask.
    #   Function : cv2.bitwise_and(src1, src2, mask=m)
    #   Args     : src1, src2 → same image twice (identity)
    #              mask       → only pixels where mask=255 survive
    extracted = cv2.bitwise_and(img4, img4, mask=mask_wide)

    # STEP 5 : DISPLAY — convert BGR → RGB before plt.imshow.
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    plt.title("Original"); plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask_tight, cmap="gray")
    plt.title("Mask 1: Too Restrictive"); plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(mask_wide, cmap="gray")
    plt.title("Mask 2: Better Range"); plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Red"); plt.axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 5 — CANNY EDGE DETECTION
# =============================================================================
#  GOAL: Show effect of the high threshold on edge count.
#  Pipeline: Read → grayscale → Gaussian blur → Canny (3 pairs) → Display.
# =============================================================================

img5 = cv2.imread("smarties.png", cv2.IMREAD_GRAYSCALE)

if img5 is None:
    print("Error: smarties.png not found")
else:
    # STEP 1 : GAUSSIAN BLUR (mandatory before Canny — kills noise edges).
    #   Function : cv2.GaussianBlur(src, ksize, sigmaX)
    #   Args     : ksize  → (width, height), both ODD (e.g., (5,5))
    #              sigmaX → 0 means auto-derive from ksize.
    blur = cv2.GaussianBlur(img5, (5, 5), 0)

    # STEP 2 : CANNY with 3 threshold pairs (low fixed at 50).
    #   Function : cv2.Canny(image, threshold1, threshold2)
    #   Args     : image      → blurred grayscale
    #              threshold1 → LOW threshold  (weak edge candidate)
    #              threshold2 → HIGH threshold (strong edge)
    #   Rule     : low : high ≈ 1 : 2 or 1 : 3
    e1 = cv2.Canny(blur, 50, 100)     # high low  → many edges (noisy)
    e2 = cv2.Canny(blur, 50, 150)     # balanced
    e3 = cv2.Canny(blur, 50, 250)     # high high → only strongest edges

    # STEP 3 : DISPLAY — Original → Edge1 → Edge2 → Edge3
    titles5 = ["Original", "Canny 50/100", "Canny 50/150", "Canny 50/250"]
    images5 = [img5, e1, e2, e3]

    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images5[i], cmap="gray")
        plt.title(titles5[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 6 — REGION GROWING (MRI segmentation)
# =============================================================================
#  GOAL: Grow a region from a seed using intensity similarity.
#  Pipeline: Define function → load grayscale → run 4 experiments → display.
# =============================================================================

# --- Function definition ---
#  region_grow(img, seed, threshold)
#  Args in order:
#      img       → grayscale image (2D NumPy array)
#      seed      → (x, y) tuple — starting pixel
#      threshold → max |pixel - seed_intensity| allowed for inclusion
#  Returns : binary uint8 mask (255 = in region, 0 = elsewhere)
def region_grow(img, seed, threshold):
    h, w = img.shape
    mask = np.zeros((h, w), np.uint8)          # start all black
    seed_value = int(img[seed[1], seed[0]])    # NOTE: (y, x) indexing in cv2
    stack = [seed]
    mask[seed[1], seed[0]] = 255               # mark seed

    while stack:
        x, y = stack.pop()                     # DFS pop
        # 4-neighbour offsets : left, right, up, down
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            # Bounds check + unvisited check + similarity check
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 0:
                if abs(int(img[ny, nx]) - seed_value) <= threshold:
                    mask[ny, nx] = 255
                    stack.append((nx, ny))
    return mask


img6 = cv2.imread("brain.jpg", cv2.IMREAD_GRAYSCALE)

if img6 is None:
    print("Error: brain.jpg not found")
else:
    h6, w6 = img6.shape
    seed_a = (w6 // 2, h6 // 2)         # centre
    seed_b = (w6 // 3, h6 // 3)         # upper-left area

    # 3 thresholds with same seed  +  1 experiment with different seed
    r1 = region_grow(img6, seed_a, 10)  # tight  → small region
    r2 = region_grow(img6, seed_a, 25)  # medium → balanced
    r3 = region_grow(img6, seed_a, 40)  # loose  → large region
    r4 = region_grow(img6, seed_b, 25)  # different seed, same threshold

    titles6 = ["Original",
               "Seed A, thresh=10", "Seed A, thresh=25",
               "Seed A, thresh=40", "Seed B, thresh=25"]
    images6 = [img6, r1, r2, r3, r4]

    plt.figure(figsize=(18, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images6[i], cmap="gray")
        plt.title(titles6[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 7 — WATERSHED SEGMENTATION (Separating Touching Coins)
# =============================================================================
#  Pipeline (10 stages):
#     1. Preprocess (grayscale)
#     2. Threshold  (INV + Otsu)
#     3. Noise removal (morphological opening)
#     4. Sure background (dilate)
#     5. Distance transform
#     6. Sure foreground (threshold on distance)
#     7. Unknown region  (sure_bg − sure_fg)
#     8. Marker labelling (connectedComponents)
#     9. Watershed transformation
#    10. Boundary visualization
# =============================================================================

img7 = cv2.imread("water_coins.jpg")            # color (watershed needs color)

if img7 is None:
    print("Error: water_coins.jpg not found")
else:
    # Stage 1 : BGR → gray
    gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)

    # Stage 2 : THRESHOLD — INV + OTSU (coins become white)
    _, thresh7 = cv2.threshold(gray7, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Stage 3 : NOISE REMOVAL — Morphological Opening (erosion → dilation).
    #   Function : cv2.morphologyEx(src, op, kernel, iterations)
    #   Args     : op         → cv2.MORPH_OPEN
    #              kernel     → np.ones((3,3), np.uint8)
    #              iterations → apply N times
    kernel7 = np.ones((3, 3), np.uint8)
    opening7 = cv2.morphologyEx(thresh7, cv2.MORPH_OPEN, kernel7, iterations=2)

    # Stage 4 : SURE BACKGROUND — dilate expands white outward.
    #   Function : cv2.dilate(src, kernel, iterations)
    sure_bg7 = cv2.dilate(opening7, kernel7, iterations=3)

    # Stage 5 : DISTANCE TRANSFORM — every white pixel → distance to nearest black.
    #   Function : cv2.distanceTransform(src, distanceType, maskSize)
    #   Args     : distanceType → DIST_L2 (Euclidean), DIST_L1, DIST_C
    #              maskSize     → 3, 5, or DIST_MASK_PRECISE
    #   Effect   : coin centres become bright peaks.
    dist7 = cv2.distanceTransform(opening7, cv2.DIST_L2, 5)

    # Stage 6 : SURE FOREGROUND — keep only bright peaks (coin centres).
    #   Function : cv2.threshold(dist, 0.7*dist.max(), 255, 0)
    #   Note     : type 0 == THRESH_BINARY
    _, sure_fg7 = cv2.threshold(dist7, 0.7 * dist7.max(), 255, 0)
    sure_fg7 = np.uint8(sure_fg7)          # distanceTransform returns float32

    # Stage 7 : UNKNOWN REGION = sure_bg − sure_fg (in-between ring).
    unknown7 = cv2.subtract(sure_bg7, sure_fg7)

    # Stage 8 : MARKER LABELLING.
    #   Function : cv2.connectedComponents(image)
    #   Returns  : (num_labels, label_map)
    #              label_map → each connected region gets unique id; bg = 0
    _, markers7 = cv2.connectedComponents(sure_fg7)
    markers7 = markers7 + 1                # shift so background becomes 1
    markers7[unknown7 == 255] = 0          # mark unknown pixels as 0

    print("Coins detected:", markers7.max() - 1)   # subtract bg label

    # Stage 9 : WATERSHED.
    #   Function : cv2.watershed(image, markers)
    #   Args     : image   → COLOR (3-channel BGR) image
    #              markers → label map (modified in place; -1 at boundaries)
    markers7 = cv2.watershed(img7, markers7)

    # Stage 10 : BOUNDARY VISUALIZATION.
    result7 = img7.copy()
    result7[markers7 == -1] = [0, 0, 255]  # red boundary (BGR order)

    # --- Display pipeline (7 images in 2x4 grid) ---
    titles7 = ["Original", "Threshold", "Sure Background",
               "Distance Transform", "Sure Foreground",
               "Unknown Region", "Final Watershed"]
    images7 = [cv2.cvtColor(img7, cv2.COLOR_BGR2RGB), thresh7, sure_bg7,
               dist7, sure_fg7, unknown7,
               cv2.cvtColor(result7, cv2.COLOR_BGR2RGB)]

    plt.figure(figsize=(18, 8))
    for i in range(7):
        plt.subplot(2, 4, i + 1)
        if i == 0 or i == 6:               # color images
            plt.imshow(images7[i])
        else:                              # single-channel images
            plt.imshow(images7[i], cmap="gray")
        plt.title(titles7[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 8 — TUNING WATERSHED (Distance-Transform Threshold)
# =============================================================================
#  GOAL: Vary 0.3 / 0.5 / 0.7 / 0.9 of dist.max() and compare markers & regions.
# =============================================================================

img8 = cv2.imread("water_coins.jpg")

if img8 is None:
    print("Error: water_coins.jpg not found")
else:
    # --- Reuse pre-processing from Task 7 ---
    gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
    _, thresh8 = cv2.threshold(gray8, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel8 = np.ones((3, 3), np.uint8)
    opening8 = cv2.morphologyEx(thresh8, cv2.MORPH_OPEN, kernel8, iterations=2)
    sure_bg8 = cv2.dilate(opening8, kernel8, iterations=3)
    dist8 = cv2.distanceTransform(opening8, cv2.DIST_L2, 5)

    # --- Sweep over 4 distance thresholds ---
    values8 = [0.3, 0.5, 0.7, 0.9]
    results8 = []
    titles8  = []

    for v in values8:
        # Step 6 : sure foreground with current v
        _, sure_fg8 = cv2.threshold(dist8, v * dist8.max(), 255, 0)
        sure_fg8 = np.uint8(sure_fg8)

        # Step 7 : unknown region
        unknown8 = cv2.subtract(sure_bg8, sure_fg8)

        # Step 8 : marker labelling
        count, markers8 = cv2.connectedComponents(sure_fg8)
        markers8 = markers8 + 1
        markers8[unknown8 == 255] = 0

        found = count - 1                  # foreground markers

        # Step 9 : watershed
        markers8 = cv2.watershed(img8, markers8)

        # Count unique labels, remove bg (1) and boundary (-1)
        regions = len(np.unique(markers8)) - 2

        # Step 10 : visualize boundaries in red
        out8 = img8.copy()
        out8[markers8 == -1] = [0, 0, 255]

        results8.append(cv2.cvtColor(out8, cv2.COLOR_BGR2RGB))
        titles8.append("thresh=" + str(v) + " | markers=" + str(found))

        print("Threshold:", v,
              "| Foreground markers:", found,
              "| Separated regions:", regions)

    # --- 2x2 grid display ---
    plt.figure(figsize=(16, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(results8[i])
        plt.title(titles8[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 9 — K-MEANS SEGMENTATION
# =============================================================================
#  Workflow : reshape → float32 → criteria → kmeans → centers → rebuild
# =============================================================================

img9 = cv2.imread("dog.jpeg")

if img9 is None:
    print("Error: dog.jpeg not found")
else:
    # STEP 1 : Flatten image to (N_pixels, 3) feature vectors.
    #   reshape((-1, 3)) → (-1) auto-computes H*W.
    pixels = img9.reshape((-1, 3))

    # STEP 2 : Convert to float32 (cv2.kmeans ONLY accepts float32).
    pixels = np.float32(pixels)

    # STEP 3 : Stopping criteria.
    #   Tuple = (type, max_iter, epsilon)
    #   type = TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER → stop on EITHER
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    results9 = []
    titles9  = []

    # STEP 4 : Loop over K values.
    for k in [2, 4, 6]:
        # Function : cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)
        #   data        → N×features float32 array (mandatory)
        #   K           → number of clusters (mandatory)
        #   bestLabels  → None to compute fresh (mandatory)
        #   criteria    → stopping tuple (mandatory)
        #   attempts    → number of independent runs (mandatory)
        #   flags       → KMEANS_RANDOM_CENTERS or KMEANS_PP_CENTERS (mandatory)
        # Returns  : (compactness, labels, centers)
        #   labels  shape (N, 1) → cluster id per pixel
        #   centers shape (K, 3) → mean BGR of each cluster
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # STEP 5 : Rebuild image using cluster centres.
        centers = np.uint8(centers)                    # back to 0–255
        segmented = centers[labels.flatten()]          # replace each pixel
        segmented = segmented.reshape(img9.shape)      # back to (H, W, 3)

        # STEP 6 : BGR → RGB for display.
        results9.append(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        titles9.append("K = " + str(k))

        print("K =", k, "| cluster colours (BGR):\n", centers, "\n")

    # STEP 7 : DISPLAY — Original → K=2 → K=4 → K=6
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img9, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    for i in range(3):
        plt.subplot(1, 4, i + 2)
        plt.imshow(results9[i])
        plt.title(titles9[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  TASK 10 — SEGMENTATION SYSTEM FOR UNKNOWN IMAGE
# =============================================================================
#  Choose 3 methods → run on same image → compare → justify.
#  Selected : Otsu, HSV Color, K-Means.
# =============================================================================

img10 = cv2.imread("smarties.png")

if img10 is None:
    print("Error: smarties.png not found")
else:
    # STEP 1 : Convert to grayscale for Otsu.
    gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)

    # METHOD 1 — Otsu
    #   Function : cv2.threshold(src, thresh, maxval, type)
    #   Combine  : THRESH_BINARY_INV + THRESH_OTSU
    _, m1 = cv2.threshold(gray10, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # METHOD 2 — HSV Color (target: red objects)
    #   Step A : BGR → HSV
    hsv10 = cv2.cvtColor(img10, cv2.COLOR_BGR2HSV)
    #   Step B : inRange(src, lowerb, upperb)
    m2 = cv2.inRange(hsv10,
                     np.array([0, 100, 80]),        # lower bound [H, S, V]
                     np.array([10, 255, 255]))      # upper bound

    # METHOD 3 — K-Means (K=4)
    #   Step A : reshape → (-1, 3)  +  float32
    pixels10 = np.float32(img10.reshape((-1, 3)))
    #   Step B : criteria tuple
    criteria10 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #   Step C : kmeans(data, K, bestLabels, criteria, attempts, flags)
    _, labels10, centers10 = cv2.kmeans(pixels10, 4, None, criteria10,
                                        10, cv2.KMEANS_RANDOM_CENTERS)
    #   Step D : rebuild
    centers10 = np.uint8(centers10)
    m3 = centers10[labels10.flatten()].reshape(img10.shape)
    m3 = cv2.cvtColor(m3, cv2.COLOR_BGR2RGB)

    # --- DISPLAY : Original | Method 1 | Method 2 | Method 3 ---
    titles10 = ["Original", "Method 1: Otsu",
                "Method 2: HSV Colour", "Method 3: K-Means (K=4)"]
    images10 = [cv2.cvtColor(img10, cv2.COLOR_BGR2RGB), m1, m2, m3]

    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        if i == 1 or i == 2:                    # binary masks → gray
            plt.imshow(images10[i], cmap="gray")
        else:                                   # color images
            plt.imshow(images10[i])
        plt.title(titles10[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  END OF LAB 05 SCRIPT
# =============================================================================
