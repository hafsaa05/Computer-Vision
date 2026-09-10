import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("barcode.jpg")

if image is None:
    print("Error: barcode.jpg not found")
else:
    height, width = image.shape[:2]

    k = 0.5   # how badly the train motion slanted it

    # ---------- Step 1: the slant that the scanner produced ----------
    S_slant = np.float32([
        [1, k],
        [0, 1]
    ])
    print("Shear matrix that caused the problem:\n", S_slant)

    # Wider canvas, because slanting pushes pixels sideways
    slant_w = int(width + abs(k) * height)

    M_slant = np.float32([
        [1, k, 0],
        [0, 1, 0]
    ])
    slanted = cv2.warpAffine(image, M_slant, (slant_w, height))

    # ---------- Step 2: build the correcting shear matrix ----------
    S_fix = np.float32([
        [1, -k],
        [0,  1]
    ])
    print("\nCorrecting shear matrix:\n", S_fix)

    # ---------- Step 3: turn it into a 2x3 and apply ----------
    M_fix = np.float32([
        [S_fix[0, 0], S_fix[0, 1], 0],
        [S_fix[1, 0], S_fix[1, 1], 0]
    ])
    print("\nFinal 2x3 matrix:\n", M_fix)

    fixed = cv2.warpAffine(slanted, M_fix, (width, height))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Barcode")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(slanted, cv2.COLOR_BGR2RGB))
    plt.title("Slanted by Train Motion (k = +0.5)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
    plt.title("Corrected (k = -0.5)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()