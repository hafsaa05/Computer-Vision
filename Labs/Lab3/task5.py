import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("chip.jpg")

if image is None:
    print("Error: chip.jpg not found")
else:
    height, width = image.shape[:2]
    cx, cy = width / 2, height / 2

    # Rigid = rotate + move only. No resizing allowed.
    angle = -30
    theta = np.radians(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    tx = 60
    ty = -40

    # Move centre to origin
    T1 = np.float32([[1, 0, -cx],
                     [0, 1, -cy],
                     [0, 0,  1]])

    # Rotate
    R = np.float32([[cos_t, -sin_t, 0],
                    [sin_t,  cos_t, 0],
                    [0,      0,     1]])

    # Move centre back
    T2 = np.float32([[1, 0, cx],
                     [0, 1, cy],
                     [0, 0, 1]])

    # Slide it to the gripper
    T3 = np.float32([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

    # Multiply all four into one matrix
    M_final = T3 @ T2 @ R @ T1
    print("Single 3x3 matrix:\n", M_final)

    # Apply it (warpAffine only needs the top 2 rows)
    result = cv2.warpAffine(image, M_final[:2, :], (width, height))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Before: Chip")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("After: Rigid Transformation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()