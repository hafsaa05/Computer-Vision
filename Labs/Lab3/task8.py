import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("chalk.jpg")

if image is None:
    print("Error: chalk.jpg not found")
else:
    height, width = image.shape[:2]

    # 4 corners of the chalk art as they look in the photo
    # (narrow at the top because it is further away)
    src = np.float32([
        [width * 0.20, height * 0.35],   # top-left
        [width * 0.80, height * 0.35],   # top-right
        [width * 0.95, height * 0.95],   # bottom-right
        [width * 0.05, height * 0.95]    # bottom-left
    ])

    # Where those corners should be for a perfect square
    size = 400
    dst = np.float32([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ])

    # Build the 3x3 perspective matrix
    M = cv2.getPerspectiveTransform(src, dst)
    print("3x3 perspective matrix:\n", M)

    birdseye = cv2.warpPerspective(image, M, (size, size))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Before: Angled View")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(birdseye, cv2.COLOR_BGR2RGB))
    plt.title("After: Birds-Eye View")
    plt.axis("off")

    plt.tight_layout()
    plt.show()