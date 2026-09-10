import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("city.jpg")

if image is None:
    print("Error: city.jpg not found")
else:
    height, width = image.shape[:2]

    # Spin it back by 45 degrees (flip the sign if it turns the wrong way)
    angle = -45
    theta = np.radians(angle)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # ---------- Step 1: build the 2x2 rotation matrix by hand ----------
    R = np.float32([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    print("2x2 rotation matrix:\n", R)

    # ---------- Step 2: new canvas size so corners are not chopped ----------
    new_w = int(width * abs(cos_t) + height * abs(sin_t))
    new_h = int(width * abs(sin_t) + height * abs(cos_t))
    print("\nCanvas grows from", (width, height), "to", (new_w, new_h))

    # ---------- Step 3: old centre and new centre ----------
    cx, cy = width / 2, height / 2
    new_cx, new_cy = new_w / 2, new_h / 2

    # ---------- Step 4: shift so the rotated image lands centred ----------
    tx = new_cx - (cos_t * cx - sin_t * cy)
    ty = new_cy - (sin_t * cx + cos_t * cy)
    print("Shift needed -> tx:", round(tx, 2), " ty:", round(ty, 2))

    # ---------- Step 5: 2x2 + shift = 2x3 matrix ----------
    M = np.float32([
        [R[0, 0], R[0, 1], tx],
        [R[1, 0], R[1, 1], ty]
    ])
    print("\nFinal 2x3 matrix:\n", M)

    # Rotation on the SAME canvas (corners get cut)
    M_cut = np.float32([
        [R[0, 0], R[0, 1], cx - (cos_t * cx - sin_t * cy)],
        [R[1, 0], R[1, 1], cy - (sin_t * cx + cos_t * cy)]
    ])
    cut = cv2.warpAffine(image, M_cut, (width, height))

    # Rotation on the EXPANDED canvas (nothing lost)
    full = cv2.warpAffine(image, M, (new_w, new_h))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original (tilted 45 degrees)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))
    plt.title("Rotated (corners chopped)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
    plt.title("Rotated (canvas expanded)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()