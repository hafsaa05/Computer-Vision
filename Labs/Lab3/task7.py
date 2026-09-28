import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("blueprint.jpg")

if image is None:
    print("Error: blueprint.jpg not found")
else:
    height, width = image.shape[:2]

    # 3 landmarks in the good image
    src = np.float32([
        [0, 0],
        [width - 1, 0],
        [0, height - 1]
    ])

    # Where those same 3 landmarks ended up after the glitch
    dst = np.float32([
        [60, 30],
        [width - 80, 90],
        [40, height - 60]
    ])

    # Make the glitched version so we have something to fix
    M_glitch = cv2.getAffineTransform(src, dst)
    glitched = cv2.warpAffine(image, M_glitch, (width, height))

    # Solve the other way round to undo it
    M_fix = cv2.getAffineTransform(dst, src)
    print("Affine matrix that fixes it:\n", M_fix)

    print("\nThe 6 unknowns we solved for:")
    print("a =", M_fix[0, 0], " b =", M_fix[0, 1], " tx =", M_fix[0, 2])
    print("c =", M_fix[1, 0], " d =", M_fix[1, 1], " ty =", M_fix[1, 2])

    fixed = cv2.warpAffine(glitched, M_fix, (width, height))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Blueprint")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(glitched, cv2.COLOR_BGR2RGB))
    plt.title("Glitched Transmission")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
    plt.title("Fixed with Affine Matrix")
    plt.axis("off")

    plt.tight_layout()
    plt.show()