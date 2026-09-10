import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("map.jpg")

if image is None:
    print("Error: map.jpg not found")
else:
    height, width = image.shape[:2]

    # ---------- Step 1: prove a 2x2 cannot translate ----------
    A = np.float32([
        [1, 0],
        [0, 1]
    ])
    origin = np.float32([0, 0])
    print("2x2 matrix times the origin:", A.dot(origin))
    print("-> always (0,0), so translation is impossible with 2x2\n")

    # ---------- Step 2: how far to slide ----------
    tx = 150   # X is 150 px to the left, so push right
    ty = 80    # X is 80 px up, so push down

    # ---------- Step 3: upgrade to a 3x3 homogeneous matrix ----------
    H = np.float32([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    print("3x3 translation matrix:\n", H)

    # ---------- Step 4: check it on a sample point ----------
    point = np.float32([100, 100, 1])   # the dummy 1 makes this work
    print("\nPoint (100,100) moves to:", H.dot(point)[:2])

    # ---------- Step 5: warpAffine only needs the top two rows ----------
    M = H[:2, :]
    print("\nSliced 2x3 matrix for warpAffine:\n", M)

    translated = cv2.warpAffine(image, M, (width, height))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Before: Misaligned Map")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
    plt.title("After: Translation (150, 80)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()