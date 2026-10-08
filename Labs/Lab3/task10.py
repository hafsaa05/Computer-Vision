import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load both images ----------
frame = cv2.imread("frame.jpg")       # museum wall with empty angled frame
painting = cv2.imread("painting.jpg") # flat painting

if frame is None or painting is None:
    print("Error: frame.jpg or painting.jpg not found")
else:
    fh, fw = frame.shape[:2]
    ph, pw = painting.shape[:2]

    # ============================================================
    # STAGE 1: LINEAR SCALE
    # Shrink the painting so it fits nicely inside the frame.
    # Linear scale = 2x2 matrix (no translation), anchored at origin.
    # ============================================================
    scale = 0.45   # shrink to 45%
    S = np.float32([
        [scale, 0, 0],
        [0, scale, 0]
    ])
    scaled = cv2.warpAffine(painting, S, (int(pw * scale), int(ph * scale)))
    sh, sw = scaled.shape[:2]

    # ============================================================
    # STAGE 2: RIGID TRANSFORMATION
    # Rotate + translate. No scaling, no shear.
    # Use center-based rotation: T3 @ T2 @ R @ T1
    # ============================================================
    angle = -8   # small tilt to match the frame's angle
    theta = np.radians(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    cx, cy = sw / 2, sh / 2
    tx, ty = -100, 0   # rough movement (will be corrected by Stage 3)

    T1 = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    R  = np.float32([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    T2 = np.float32([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    T3 = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    M_rigid = T3 @ T2 @ R @ T1
    rigid = cv2.warpAffine(scaled, M_rigid[:2, :], (fw, fh))
    # place on frame-sized canvas

    # ============================================================
    # STAGE 3: PROJECTIVE TRANSFORMATION
    # Warp the painting's 4 corners precisely into the frame's inner area.
    # ============================================================
    # 4 corners of the frame's inner white area (estimated from image)
    frame_inner = np.float32([
        [fw * 0.30, fh * 0.28],   # top-left
        [fw * 0.72, fh * 0.25],   # top-right
        [fw * 0.78, fh * 0.72],   # bottom-right
        [fw * 0.34, fh * 0.78]    # bottom-left
    ])

    # 4 corners of the rigid-transformed painting (its bounding box)
    # Find non-black region of 'rigid' to get actual corners
    gray = cv2.cvtColor(rigid, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(xs) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
    else:
        x1, y1, x2, y2 = 0, 0, fw, fh

    painting_corners = np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ])

    # Get homography from painting corners -> frame inner corners
    H = cv2.getPerspectiveTransform(painting_corners, frame_inner)
    warped = cv2.warpPerspective(rigid, H, (fw, fh))

    # ============================================================
    # MASK + PASTE
    # ============================================================
    # Create mask from warped painting (non-black pixels)
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask = (gray_w > 0).astype(np.uint8) * 255

    # Composite: use mask to paste painting into frame
    frame_copy = frame.copy()
    frame_copy[mask > 0] = warped[mask > 0]

    # ---------- Display ----------
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(painting, cv2.COLOR_BGR2RGB))
    plt.title("Original Painting")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Empty Frame")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
    plt.title("Final: Painting in Frame")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
