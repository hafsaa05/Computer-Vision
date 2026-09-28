import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("blueprint.jpg")

if image is None:
    print("Error: blueprint.jpg not found")
else:
    height, width = image.shape[:2]
    cx, cy = width / 2, height / 2

    # Similarity = scale + rotate + move
    s = 1.2                    # blueprint was too small
    angle = -20                # it was rotated randomly
    tx, ty = 50, 30            # it was in the wrong corner

    theta = np.radians(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Move centre to origin
    T1 = np.float32([[1, 0, -cx],
                     [0, 1, -cy],
                     [0, 0,  1]])

    # Scale (same value for x and y, so angles stay safe)
    S = np.float32([[s, 0, 0],
                    [0, s, 0],
                    [0, 0, 1]])

    # Rotate
    R = np.float32([[cos_t, -sin_t, 0],
                    [sin_t,  cos_t, 0],
                    [0,      0,     1]])

    # Move centre back
    T2 = np.float32([[1, 0, cx],
                     [0, 1, cy],
                     [0, 0, 1]])

    # Slide to the right spot
    T3 = np.float32([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

    # One matrix does all of it
    M_final = T3 @ T2 @ R @ S @ T1
    print("Single 3x3 matrix:\n", M_final)

    result = cv2.warpAffine(image, M_final[:2, :], (width, height))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Before: Blueprint")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("After: Similarity Transformation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()