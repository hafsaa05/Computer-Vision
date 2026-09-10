import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load the fingerprint ----------
image = cv2.imread("fingerprint.png")

if image is None:
    print("Error: fingerprint.png not found")
else:
    height, width = image.shape[:2]
    print("Original size:", width, "x", height)

    # ---------- Step 1: scale factors (300% = 3 times bigger) ----------
    sx = 3
    sy = 3

    # ---------- Step 2: build the 2x2 scaling matrix by hand ----------
    S = np.float32([
        [sx, 0],
        [0, sy]
    ])
    print("\n2x2 scaling matrix:\n", S)

    # ---------- Step 3: find the centre of the image ----------
    cx = width / 2
    cy = height / 2

    # ---------- Step 4: where the centre lands after scaling ----------
    new_cx = sx * cx
    new_cy = sy * cy
    print("\nCentre moves from", (cx, cy), "to", (new_cx, new_cy))

    # ---------- Step 5: how far to slide it back ----------
    tx = cx - new_cx      # same as (1 - sx) * cx
    ty = cy - new_cy      # same as (1 - sy) * cy
    print("Shift needed -> tx:", tx, " ty:", ty)

    # ---------- Step 6: attach the shift -> 2x3 matrix for warpAffine ----------
    M = np.float32([
        [S[0, 0], S[0, 1], tx],
        [S[1, 0], S[1, 1], ty]
    ])
    print("\nFinal 2x3 matrix:\n", M)

    # ---------- Scaling WITHOUT centering (shows the origin problem) ----------
    M_naive = np.float32([
        [sx, 0, 0],
        [0, sy, 0]
    ])
    wrong = cv2.warpAffine(image, M_naive, (width, height))

    # ---------- Scaling WITH centering ----------
    centered = cv2.warpAffine(image, M, (width, height))

    # ---------- Display all three ----------
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Fingerprint")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(wrong, cv2.COLOR_BGR2RGB))
    plt.title("3x Scale (anchored to origin)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(centered, cv2.COLOR_BGR2RGB))
    plt.title("3x Scale (centered)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()