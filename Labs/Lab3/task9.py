import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("left.jpg")
img2 = cv2.imread("right.jpg")

if img1 is None or img2 is None:
    print("Error: left.jpg or right.jpg not found")
else:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 4 matching points in the RIGHT photo (the cliff area, left side)
    pts_right = np.float32([
        [w2 * 0.02, h2 * 0.35],
        [w2 * 0.30, h2 * 0.35],
        [w2 * 0.30, h2 * 0.65],
        [w2 * 0.02, h2 * 0.65]
    ])

    # The SAME cliff in the LEFT photo (right side)
    pts_left = np.float32([
        [w1 * 0.55, h1 * 0.35],
        [w1 * 0.90, h1 * 0.35],
        [w1 * 0.90, h1 * 0.65],
        [w1 * 0.55, h1 * 0.65]
    ])

    # Homography maps the right photo into the left photo's space
    H = cv2.getPerspectiveTransform(pts_right, pts_left)
    print("Homography matrix:\n", H)

    stitched = cv2.warpPerspective(img2, H, (w1 + w2, h1))
    stitched[0:h1, 0:w1] = img1

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Camera 1 (Left)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("Camera 2 (Right)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    plt.title("Stitched Panorama")
    plt.axis("off")

    plt.tight_layout()
    plt.show()