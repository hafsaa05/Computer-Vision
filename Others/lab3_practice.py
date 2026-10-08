# =====================================================================
#  LAB 03 — GEOMETRIC TRANSFORMATIONS
#  Complete Code File for All 10 Tasks
#  ---------------------------------------------------------------------
#  ONE-LINE VIBE: This lab is about moving pixels around, not changing
#                 their values. (Geometric ≠ Photometric)
#
#  HIERARCHY: Projective (3×3) → Affine (2×3) → Linear (2×2) → Rigid
# =====================================================================


# =====================================================================
#  IMPORTS
# ---------------------------------------------------------------------
#  cv2   → OpenCV for image loading & transformations
#  np    → NumPy for matrix math (np.float32 arrays)
#  plt   → Matplotlib for displaying images
# =====================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
#  UNIVERSAL DISPLAY HELPER
# ---------------------------------------------------------------------
#  Purpose : Convert BGR (OpenCV) → RGB (Matplotlib) and display.
#  Why     : OpenCV loads images in BGR. Matplotlib expects RGB.
#  Args    :
#      img    → image to display (mandatory)
#      title  → string title (mandatory)
#  Order   : (img, title)
# =====================================================================
def show_image(img, title="Image"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")


# #####################################################################
#  TASK 1 — SCALING WITH CENTERING  (fingerprint.png)
# ---------------------------------------------------------------------
#  GOAL   : Scale image 3× while keeping it centered.
#  CONCEPT: Linear scaling (2×2) shifts image because origin is
#           anchored at top-left. Add translation to re-center.
#  KEY EQ : tx = (1 - sx) * cx      ty = (1 - sy) * cy
# #####################################################################
print("\n========== TASK 1: SCALING WITH CENTERING ==========")

# STEP 1: Load image
image = cv2.imread("fingerprint.png")

# STEP 2: Safety check
if image is None:
    print("Error: fingerprint.png not found")
else:
    # STEP 3: Get dimensions
    #   image.shape       → (height, width, channels)
    #   image.shape[:2]   → (height, width)
    height, width = image.shape[:2]
    print("Original size:", width, "x", height)

    # STEP 4: Set scale factors
    sx = 3   # horizontal scale (3× bigger)
    sy = 3   # vertical scale   (3× bigger)

    # STEP 5: Build 2×2 scaling matrix
    #   Pattern: [[sx, 0], [0, sy]]
    #   - Top-left     = x scaling
    #   - Bottom-right = y scaling
    #   - Off-diagonal = shear (0 here)
    S = np.float32([
        [sx, 0],
        [0, sy]
    ])
    print("\n2x2 scaling matrix:\n", S)

    # STEP 6: Find center of image
    cx = width / 2
    cy = height / 2

    # STEP 7: Where center moves after scaling (origin-anchored)
    new_cx = sx * cx
    new_cy = sy * cy

    # STEP 8: Calculate translation to bring center back
    tx = cx - new_cx     # same as (1 - sx) * cx
    ty = cy - new_cy     # same as (1 - sy) * cy

    # STEP 9: Build 2×3 matrix (add translation column)
    M = np.float32([
        [S[0, 0], S[0, 1], tx],
        [S[1, 0], S[1, 1], ty]
    ])

    # STEP 10: "Wrong" version — no centering
    M_naive = np.float32([
        [sx, 0, 0],
        [0, sy, 0]
    ])
    wrong = cv2.warpAffine(image, M_naive, (width, height))

    # STEP 11: "Correct" version — with centering
    centered = cv2.warpAffine(image, M, (width, height))

    # STEP 12: Display all three side by side
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); show_image(image,    "Original")
    plt.subplot(1, 3, 2); show_image(wrong,    "3x (origin-anchored)")
    plt.subplot(1, 3, 3); show_image(centered, "3x (centered)")
    plt.tight_layout(); plt.show()


# #####################################################################
#  TASK 2 — ROTATION WITH EXPANDED CANVAS  (city.jpg)
# ---------------------------------------------------------------------
#  GOAL   : Rotate manually by -45° on an expanded canvas.
#  CONCEPT: Rotation matrix from cos/sin; canvas grows to fit corners.
#  KEY EQ : new_w = w·|cos| + h·|sin|
#           new_h = w·|sin| + h·|cos|
# #####################################################################
print("\n========== TASK 2: ROTATION WITH EXPANDED CANVAS ==========")

image = cv2.imread("city.jpg")

if image is None:
    print("Error: city.jpg not found")
else:
    height, width = image.shape[:2]

    # STEP 1: Set angle and convert to radians
    angle = -45
    theta = np.radians(angle)   # cos/sin need radians

    # STEP 2: Compute cos and sin
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # STEP 3: Build 2×2 rotation matrix
    #   Pattern: [[cos, -sin], [sin, cos]]
    R = np.float32([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])

    # STEP 4: Calculate expanded canvas size
    new_w = int(width * abs(cos_t) + height * abs(sin_t))
    new_h = int(width * abs(sin_t) + height * abs(cos_t))

    # STEP 5: Old center and new center
    cx, cy = width / 2, height / 2
    new_cx, new_cy = new_w / 2, new_h / 2

    # STEP 6: Translation to re-center rotated image
    tx = new_cx - (cos_t * cx - sin_t * cy)
    ty = new_cy - (sin_t * cx + cos_t * cy)

    # STEP 7: Build 2×3 matrix (rotation + translation)
    M = np.float32([
        [R[0, 0], R[0, 1], tx],
        [R[1, 0], R[1, 1], ty]
    ])

    # STEP 8: Same-canvas version (corners cut)
    M_cut = np.float32([
        [R[0, 0], R[0, 1], cx - (cos_t * cx - sin_t * cy)],
        [R[1, 0], R[1, 1], cy - (sin_t * cx + cos_t * cy)]
    ])
    cut = cv2.warpAffine(image, M_cut, (width, height))

    # STEP 9: Expanded-canvas version (nothing lost)
    full = cv2.warpAffine(image, M, (new_w, new_h))

    # STEP 10: Display
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); show_image(image, "Original")
    plt.subplot(1, 3, 2); show_image(cut,   "Rotated (corners chopped)")
    plt.subplot(1, 3, 3); show_image(full,  "Rotated (canvas expanded)")
    plt.tight_layout(); plt.show()


# #####################################################################
#  TASK 3 — SHEARING & CORRECTING SHEAR  (barcode.jpg)
# ---------------------------------------------------------------------
#  GOAL   : Slant image (shear) then undo it (inverse shear).
#  CONCEPT: Horizontal shear matrix; wider canvas prevents cut-off.
#  MATRIX : [[1, k], [0, 1]]   ← shear
#           [[1,-k], [0, 1]]   ← correcting shear
# #####################################################################
print("\n========== TASK 3: SHEARING & CORRECTING SHEAR ==========")

image = cv2.imread("barcode.jpg")

if image is None:
    print("Error: barcode.jpg not found")
else:
    height, width = image.shape[:2]

    # STEP 1: Shear factor
    k = 0.5

    # STEP 2: 2×2 shear matrix
    S_slant = np.float32([
        [1, k],
        [0, 1]
    ])

    # STEP 3: Wider canvas (pixels move sideways)
    slant_w = int(width + abs(k) * height)

    # STEP 4: Build 2×3 matrix with zero translation
    M_slant = np.float32([
        [1, k, 0],
        [0, 1, 0]
    ])
    slanted = cv2.warpAffine(image, M_slant, (slant_w, height))

    # STEP 5: Correcting shear matrix (negative k)
    S_fix = np.float32([
        [1, -k],
        [0,  1]
    ])

    # STEP 6: Build 2×3 correcting matrix
    M_fix = np.float32([
        [S_fix[0, 0], S_fix[0, 1], 0],
        [S_fix[1, 0], S_fix[1, 1], 0]
    ])

    # STEP 7: Apply correction on slanted image
    fixed = cv2.warpAffine(slanted, M_fix, (width, height))

    # STEP 8: Display
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); show_image(image,   "Original Barcode")
    plt.subplot(1, 3, 2); show_image(slanted, "Slanted (k = +0.5)")
    plt.subplot(1, 3, 3); show_image(fixed,   "Corrected (k = -0.5)")
    plt.tight_layout(); plt.show()


# #####################################################################
#  TASK 4 — TRANSLATION & HOMOGENEOUS COORDINATES  (map.jpg)
# ---------------------------------------------------------------------
#  GOAL   : Prove 2×2 cannot translate → use 3×3 homogeneous.
#  PROOF  : A · 0 = 0  → origin fixed → no translation.
#  MATRIX : [[1, 0, tx], [0, 1, ty], [0, 0, 1]]
# #####################################################################
print("\n========== TASK 4: TRANSLATION & HOMOGENEOUS ==========")

image = cv2.imread("map.jpg")

if image is None:
    print("Error: map.jpg not found")
else:
    height, width = image.shape[:2]

    # STEP 1: PROOF — 2×2 identity × origin = origin
    A = np.float32([
        [1, 0],
        [0, 1]
    ])
    origin = np.float32([0, 0])
    print("2x2 matrix × origin:", A.dot(origin))
    print("→ always (0,0), so 2×2 cannot translate\n")

    # STEP 2: Translation amounts
    tx = 150   # 150 px right
    ty = 80    # 80 px down (y grows downward)

    # STEP 3: Build 3×3 homogeneous translation matrix
    H = np.float32([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

    # STEP 4: Verify on sample point (100, 100)
    point = np.float32([100, 100, 1])   # dummy 1 makes it work
    print("Point (100,100) → ", H.dot(point)[:2])

    # STEP 5: Slice top 2 rows for warpAffine (needs 2×3)
    M = H[:2, :]

    # STEP 6: Apply
    translated = cv2.warpAffine(image, M, (width, height))

    # STEP 7: Display
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); show_image(image,      "Before: Misaligned")
    plt.subplot(1, 2, 2); show_image(translated, "After: Translated (150, 80)")
    plt.tight_layout(); plt.show()


# #####################################################################
#  TASK 5 — RIGID TRANSFORMATION  (chip.jpg)
# ---------------------------------------------------------------------
#  GOAL   : Rigid = rotate + translate (no scale, no shear).
#  CONCEPT: Center-based rotation via matrix composition.
#  ORDER  : M = T3 @ T2 @ R @ T1  (rightmost applied first)
# #####################################################################
print("\n========== TASK 5: RIGID TRANSFORMATION ==========")

image = cv2.imread("chip.jpg")

if image is None:
    print("Error: chip.jpg not found")
else:
    height, width = image.shape[:2]
    cx, cy = width / 2, height / 2

    # STEP 1: Angle & translation
    angle = -30
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    tx = 60
    ty = -40

    # STEP 2: T1 — Move center to origin
    T1 = np.float32([[1, 0, -cx],
                     [0, 1, -cy],
                     [0, 0,  1]])

    # STEP 3: R — Rotation (3×3 homogeneous)
    R = np.float32([[cos_t, -sin_t, 0],
                    [sin_t,  cos_t, 0],
                    [0,      0,     1]])

    # STEP 4: T2 — Move center back
    T2 = np.float32([[1, 0, cx],
                     [0, 1, cy],
                     [0, 0, 1]])

    # STEP 5: T3 — Slide to target position
    T3 = np.float32([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

    # STEP 6: Compose → single 3×3 matrix
    M_final = T3 @ T2 @ R @ T1

    # STEP 7: Slice top 2 rows → 2×3 → warpAffine
    result = cv2.warpAffine(image, M_final[:2, :], (width, height))

    # STEP 8: Display
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); show_image(image,  "Before: Chip")
    plt.subplot(1, 2, 2); show_image(result, "After: Rigid Transform")
    plt.tight_layout(); plt.show()


# #####################################################################
#  TASK 6 — SIMILARITY TRANSFORMATION  (blueprint.jpg)
# ---------------------------------------------------------------------
#  GOAL   : Uniform scale + rotate + translate (angles preserved).
#  ORDER  : M = T3 @ T2 @ R @ S @ T1
#  NOTE   : sx = sy for uniform scale.
# #####################################################################
print("\n========== TASK 6: SIMILARITY TRANSFORMATION ==========")

image = cv2.imread("blueprint.jpg")

if image is None:
    print("Error: blueprint.jpg not found")
else:
    height, width = image.shape[:2]
    cx, cy = width / 2, height / 2

    # STEP 1: Parameters
    s = 1.2
    angle = -20
    tx, ty = 50, 30

    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # STEP 2: T1 — center → origin
    T1 = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

    # STEP 3: S — uniform scale
    S = np.float32([[s, 0, 0],
                    [0, s, 0],
                    [0, 0, 1]])

    # STEP 4: R — rotation
    R = np.float32([[cos_t, -sin_t, 0],
                    [sin_t,  cos_t, 0],
                    [0,      0,     1]])

    # STEP 5: T2 — origin → center
    T2 = np.float32([[1, 0, cx], [0, 1, cy], [0, 0, 1]])

    # STEP 6: T3 — slide to target
    T3 = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # STEP 7: Compose
    M_final = T3 @ T2 @ R @ S @ T1

    # STEP 8: Apply
    result = cv2.warpAffine(image, M_final[:2, :], (width, height))

    # STEP 9: Display
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); show_image(image,  "Before: Blueprint")
    plt.subplot(1, 2, 2); show_image(result, "After: Similarity")
    plt.tight_layout(); plt.show()


# #####################################################################
#  TASK 7 — AFFINE TRANSFORM (3 POINT PAIRS)  (blueprint.jpg)
# ---------------------------------------------------------------------
#  GOAL   : Glitch image then fix it using affine transform.
#  FUNCTION: cv2.getAffineTransform(src, dst) → 2×3 matrix
#  ARGS   : src → 3 source points np.float32 shape (3,2)
#           dst → 3 destination points same order
#  NOTE   : Needs exactly 3 pairs (6 equations, 6 unknowns).
# #####################################################################
print("\n========== TASK 7: AFFINE TRANSFORM (3 POINTS) ==========")

image = cv2.imread("blueprint.jpg")

if image is None:
    print("Error: blueprint.jpg not found")
else:
    height, width = image.shape[:2]

    # STEP 1: 3 source landmarks
    src = np.float32([
        [0, 0],
        [width - 1, 0],
        [0, height - 1]
    ])

    # STEP 2: 3 destination landmarks (glitched positions)
    dst = np.float32([
        [60, 30],
        [width - 80, 90],
        [40, height - 60]
    ])

    # STEP 3: Build glitch matrix → apply
    M_glitch = cv2.getAffineTransform(src, dst)
    glitched = cv2.warpAffine(image, M_glitch, (width, height))

    # STEP 4: Reverse (swap src/dst) → fix
    M_fix = cv2.getAffineTransform(dst, src)
    print("Fix matrix (6 unknowns):")
    print("a=", M_fix[0,0], "b=", M_fix[0,1], "tx=", M_fix[0,2])
    print("c=", M_fix[1,0], "d=", M_fix[1,1], "ty=", M_fix[1,2])

    # STEP 5: Apply fix on glitched image
    fixed = cv2.warpAffine(glitched, M_fix, (width, height))

    # STEP 6: Display
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); show_image(image,    "Original")
    plt.subplot(1, 3, 2); show_image(glitched, "Glitched")
    plt.subplot(1, 3, 3); show_image(fixed,    "Fixed (Affine)")
    plt.tight_layout(); plt.show()


# #####################################################################
#  TASK 8 — PERSPECTIVE TRANSFORM (4 POINT PAIRS)  (chalk.jpg)
# ---------------------------------------------------------------------
#  GOAL   : Convert angled photo into top-down (birds-eye) view.
#  FUNCTION: cv2.getPerspectiveTransform(src, dst) → 3×3 matrix
#  ARGS   : src → 4 points np.float32 shape (4,2) TL→TR→BR→BL
#           dst → 4 points same order
#  NOTE   : Needs exactly 4 pairs (8 equations, 8 unknowns).
# #####################################################################
print("\n========== TASK 8: PERSPECTIVE (BIRDS-EYE) ==========")

image = cv2.imread("chalk.jpg")

if image is None:
    print("Error: chalk.jpg not found")
else:
    height, width = image.shape[:2]

    # STEP 1: 4 source corners (angled view)
    src = np.float32([
        [width * 0.20, height * 0.35],   # top-left
        [width * 0.80, height * 0.35],   # top-right
        [width * 0.95, height * 0.95],   # bottom-right
        [width * 0.05, height * 0.95]    # bottom-left
    ])

    # STEP 2: 4 destination corners (perfect square)
    size = 400
    dst = np.float32([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ])

    # STEP 3: Compute homography
    M = cv2.getPerspectiveTransform(src, dst)
    print("3x3 perspective matrix:\n", M)

    # STEP 4: Apply warpPerspective
    birdseye = cv2.warpPerspective(image, M, (size, size))

    # STEP 5: Display
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); show_image(image,    "Before: Angled")
    plt.subplot(1, 2, 2); show_image(birdseye, "After: Birds-Eye")
    plt.tight_layout(); plt.show()


# #####################################################################
#  TASK 9 — IMAGE STITCHING (PANORAMA)  (left.jpg + right.jpg)
# ---------------------------------------------------------------------
#  GOAL   : Combine two overlapping photos into a panorama.
#  CONCEPT: Homography maps right image into left image's space.
#  CANVAS : (w1 + w2, h1)
# #####################################################################
print("\n========== TASK 9: IMAGE STITCHING ==========")

img1 = cv2.imread("left.jpg")
img2 = cv2.imread("right.jpg")

if img1 is None or img2 is None:
    print("Error: left.jpg or right.jpg not found")
else:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # STEP 1: 4 matching points in RIGHT photo (source)
    pts_right = np.float32([
        [w2 * 0.02, h2 * 0.35],
        [w2 * 0.30, h2 * 0.35],
        [w2 * 0.30, h2 * 0.65],
        [w2 * 0.02, h2 * 0.65]
    ])

    # STEP 2: Same 4 points in LEFT photo (destination)
    pts_left = np.float32([
        [w1 * 0.55, h1 * 0.35],
        [w1 * 0.90, h1 * 0.35],
        [w1 * 0.90, h1 * 0.65],
        [w1 * 0.55, h1 * 0.65]
    ])

    # STEP 3: Homography mapping right → left space
    H = cv2.getPerspectiveTransform(pts_right, pts_left)
    print("Homography matrix:\n", H)

    # STEP 4: Warp right image into bigger canvas
    stitched = cv2.warpPerspective(img2, H, (w1 + w2, h1))

    # STEP 5: Paste left image on left side
    stitched[0:h1, 0:w1] = img1

    # STEP 6: Display
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); show_image(img1,     "Camera 1 (Left)")
    plt.subplot(1, 3, 2); show_image(img2,     "Camera 2 (Right)")
    plt.subplot(1, 3, 3); show_image(stitched, "Stitched Panorama")
    plt.tight_layout(); plt.show()


# #####################################################################
#  TASK 10 — THE MASTER FORGER  (frame.jpg + painting.jpg)
# ---------------------------------------------------------------------
#  GOAL   : Insert flat painting into an angled empty frame.
#  STAGES :
#      Stage 1: LINEAR SCALE   → shrink painting
#      Stage 2: RIGID TRANSFORM → rotate + move near frame
#      Stage 3: PROJECTIVE      → warp into frame's inner area
#  COMPOSITE: Mask + paste via indexing.
# #####################################################################
print("\n========== TASK 10: MASTER FORGER ==========")

frame = cv2.imread("frame.jpg")
painting = cv2.imread("painting.jpg")

if frame is None or painting is None:
    print("Error: frame.jpg or painting.jpg not found")
else:
    fh, fw = frame.shape[:2]
    ph, pw = painting.shape[:2]

    # STEP 0: Crop painting out of its wall background
    painting = painting[int(ph * 0.14):int(ph * 0.88),
                        int(pw * 0.04):int(pw * 0.96)]
    ph, pw = painting.shape[:2]

    # ---------- STAGE 1: LINEAR SCALE ----------
    #   Pattern: [[s, 0, 0], [0, s, 0]]  (2×3)
    scale = 0.35
    S = np.float32([
        [scale, 0, 0],
        [0, scale, 0]
    ])
    new_pw = int(pw * scale)
    new_ph = int(ph * scale)
    scaled = cv2.warpAffine(painting, S, (new_pw, new_ph))

    # ---------- STAGE 2: RIGID TRANSFORM ----------
    #   M = T2 @ R @ T1  →  rotate around center then move
    angle = -5
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    cx, cy = new_pw / 2, new_ph / 2
    target_cx = fw * 0.52
    target_cy = fh * 0.51

    T1 = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    R  = np.float32([[cos_t, -sin_t, 0],
                     [sin_t,  cos_t, 0],
                     [0,      0,     1]])
    T2 = np.float32([[1, 0, target_cx], [0, 1, target_cy], [0, 0, 1]])

    M_rigid = T2 @ R @ T1
    rigid = cv2.warpAffine(scaled, M_rigid[:2, :], (fw, fh))

    # ---------- STAGE 3: PROJECTIVE ----------
    #   Homography from painting bbox → frame inner corners
    frame_inner = np.float32([
        [fw * 0.35, fh * 0.30],   # top-left
        [fw * 0.68, fh * 0.27],   # top-right
        [fw * 0.72, fh * 0.72],   # bottom-right
        [fw * 0.38, fh * 0.76]    # bottom-left
    ])

    # Find painting's bounding box in rigid image
    gray = cv2.cvtColor(rigid, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 5)
    if len(xs) > 0:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
    else:
        x1, y1, x2, y2 = 0, 0, fw, fh

    painting_corners = np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ])

    H = cv2.getPerspectiveTransform(painting_corners, frame_inner)
    warped = cv2.warpPerspective(rigid, H, (fw, fh))

    # ---------- COMPOSITE (mask + paste) ----------
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask = (gray_w > 5).astype(np.uint8) * 255

    result = frame.copy()
    result[mask > 0] = warped[mask > 0]

    # ---------- DISPLAY ----------
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1); show_image(painting, "Original Painting")
    plt.subplot(1, 3, 2); show_image(frame,    "Empty Frame")
    plt.subplot(1, 3, 3); show_image(result,   "Final: Painting in Frame")
    plt.tight_layout(); plt.show()


# =====================================================================
#  END OF LAB 03
# ---------------------------------------------------------------------
#  QUICK REFERENCE:
#
#  MATRICES:
#    Translation  → [[1,0,tx],[0,1,ty]]
#    Scaling      → [[sx,0,0],[0,sy,0]]
#    Rotation     → [[cos,-sin,0],[sin,cos,0]]
#    Shear (x)    → [[1,k,0],[0,1,0]]
#    Shear (y)    → [[1,0,0],[k,1,0]]
#    Reflection   → [[-1,0,0],[0,1,0]]
#
#  FUNCTIONS:
#    cv2.getRotationMatrix2D(center, angle, scale) → 2×3
#    cv2.warpAffine(src, M, (w, h))               → image
#    cv2.getAffineTransform(src3, dst3)           → 2×3
#    cv2.getPerspectiveTransform(src4, dst4)      → 3×3
#    cv2.warpPerspective(src, M, (w, h))          → image
#    cv2.resize(src, None, fx, fy, interpolation) → image
#
#  PRESERVES:
#    Rigid      → distances, angles
#    Affine     → parallelism, collinearity, ratios
#    Projective → collinearity only
#
#  POSITIVE ANGLE = COUNTER-CLOCKWISE
#  BGR → RGB before Matplotlib!
# =====================================================================
