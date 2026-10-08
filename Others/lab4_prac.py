# =============================================================================
# LAB 04 — FEATURE EXTRACTION
# Computer Vision AI-4002
# Complete Code Reference for Open Book Exam
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
import cv2                      # OpenCV: image reading, processing, edge detection
import numpy as np              # NumPy: array operations, math
import matplotlib.pyplot as plt # Matplotlib: displaying images and plots
from skimage import feature     # skimage: LBP
from skimage.feature import hog # skimage: HOG
from skimage import exposure    # skimage: rescale intensity for HOG display


# =============================================================================
# SECTION 0: COMMON UTILITY — LOAD IMAGE (GRAYSCALE)
# =============================================================================
# Most feature extraction and edge detection tasks need GRAYSCALE.
# Why: Simplifies computation (single channel), avoids color bias.
#
# cv2.imread(path, flag)
#   - path: file path to image
#   - flag: cv2.IMREAD_GRAYSCALE → loads directly as grayscale
#           cv2.IMREAD_COLOR     → loads as BGR color (default)
#
# Always check if image is None (file not found).

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("image.jpg not found")


# =============================================================================
# SECTION 1: HOG — HISTOGRAM OF ORIENTED GRADIENTS
# =============================================================================
# WHY: Captures gradient orientation distribution → shape/edge info.
# WHERE: Pedestrian detection, object recognition.
# HOW: Divide image into cells → histogram per cell → normalize blocks → concat.
#
# hog(image, orientations, pixels_per_cell, cells_per_block, visualize, ...)
#   - image: grayscale input (2D array)
#   - orientations: number of bins (default 9 → 0°–180°)
#   - pixels_per_cell: tuple (e.g. (8,8))
#   - cells_per_block: tuple (e.g. (2,2)) for normalization
#   - visualize: True → also returns HOG image for display
#
# Returns: features (1D vector), hog_image (if visualize=True)

print("=" * 60)
print("SECTION 1: HOG — Histogram of Oriented Gradients")
print("=" * 60)

# Step 1: Compute HOG features
features, hog_image = hog(
    image,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)

# Step 2: Rescale HOG image for better visualization
# exposure.rescale_intensity(image, in_range)
#   - in_range: clip range for rescaling (here (0,10))
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Step 3: Display original vs HOG
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(image, cmap='gray'); plt.title('Original Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(hog_image_rescaled, cmap='gray'); plt.title('HOG Features'); plt.axis('off')
plt.tight_layout(); plt.show()

print("HOG feature vector length:", len(features))


# =============================================================================
# SECTION 2: LBP — LOCAL BINARY PATTERN
# =============================================================================
# WHY: Captures local texture patterns by comparing center pixel with neighbors.
# WHERE: Texture classification, face recognition.
# HOW: Neighbor ≥ center → 1, else → 0 → binary pattern → histogram.
#
# feature.local_binary_pattern(image, P, R, method)
#   - image: grayscale input
#   - P: number of circularly symmetric neighbor points (e.g. 8)
#   - R: radius of circle (e.g. 1)
#   - method: 'uniform' (common), 'default', 'ror', 'var'
#
# Returns: LBP image (same shape as input)

print("\n" + "=" * 60)
print("SECTION 2: LBP — Local Binary Pattern")
print("=" * 60)

radius = 1          # R
n_points = 8        # P

lbp_image = feature.local_binary_pattern(image, n_points, radius, method='uniform')

# Build histogram of LBP patterns
# np.histogram(a, bins, range)
#   - a: flattened LBP image (ravel)
#   - bins: bin edges (np.arange(0, n_points+3))
#   - range: (0, n_points+2)
hist, _ = np.histogram(
    lbp_image.ravel(),
    bins=np.arange(0, n_points + 3),
    range=(0, n_points + 2)
)

# Normalize histogram (scale-invariance)
hist = hist.astype("float")
hist /= (hist.sum() + 1e-6)   # 1e-6 avoids divide-by-zero

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(lbp_image, cmap='gray'); plt.title('LBP Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.bar(range(0, n_points + 2), hist); plt.title('LBP Histogram')
plt.xlabel('LBP Patterns'); plt.ylabel('Frequency')
plt.tight_layout(); plt.show()

print("LBP feature vector length:", len(hist))


# =============================================================================
# SECTION 3: HOC — HISTOGRAM OF COLOR
# =============================================================================
# WHY: Captures distribution of color intensities.
# WHERE: Image retrieval, color-based segmentation.
# HOW: Divide color space (BGR/HSV) into bins → count pixels.
#
# cv2.calcHist(images, channels, mask, histSize, ranges)
#   - images: [image]  (note: inside a LIST)
#   - channels: [0] for grayscale; [0],[1],[2] for B,G,R
#   - mask: None (whole image)
#   - histSize: [256] (number of bins)
#   - ranges: [0, 256] (intensity range; upper exclusive)

print("\n" + "=" * 60)
print("SECTION 3: HOC — Histogram of Color")
print("=" * 60)

# Load color image (BGR) for color histogram
color_image = cv2.imread('image.jpg')
color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# Compute histogram for each channel
plt.figure(figsize=(8, 5))
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([color_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)

plt.title('Histogram of Color (HOC)')
plt.xlabel('Intensity (0–255)')
plt.ylabel('Pixel Count')
plt.xlim([0, 256])
plt.show()


# =============================================================================
# SECTION 4: HED — HISTOGRAM OF EDGE DIRECTIONS
# =============================================================================
# WHY: Captures dominant edge orientations.
# WHERE: Texture analysis, object recognition, segmentation.
# HOW: Edge detection (Canny/Sobel) → gradient orientation → bin → histogram.
#
# Steps:
#   1. Gaussian blur (noise reduction)
#   2. Canny edge detection
#   3. Sobel gradients (x and y)
#   4. arctan2 → orientation
#   5. Histogram of orientations

print("\n" + "=" * 60)
print("SECTION 4: HED — Histogram of Edge Directions")
print("=" * 60)

# Step 1: Gaussian blur
# cv2.GaussianBlur(src, ksize, sigmaX)
#   - ksize: (5,5) odd kernel
#   - sigmaX: 0 → auto-calculate
image_smoothed = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Canny edge detection
# cv2.Canny(image, threshold1, threshold2)
#   - threshold1: lower (T_low)
#   - threshold2: upper (T_high)
edges = cv2.Canny(image_smoothed, 30, 70)

# Step 3: Sobel gradients
# cv2.Sobel(src, ddepth, dx, dy, ksize)
#   - ddepth: cv2.CV_64F (preserve negatives)
#   - dx=1, dy=0 → horizontal gradient
#   - dx=0, dy=1 → vertical gradient
gradient_x = cv2.Sobel(image_smoothed, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image_smoothed, cv2.CV_64F, 0, 1, ksize=3)

# Step 4: Orientation
# np.arctan2(y, x) → angle in radians; convert to degrees
gradient_orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

# Step 5: Histogram of orientations
# np.histogram(a, bins, range)
hist_hed, bin_edges = np.histogram(gradient_orientation, bins=8, range=(0, 360))

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(edges, cmap='gray'); plt.title('Edge Map'); plt.axis('off')
plt.subplot(1, 2, 2); plt.bar(bin_edges[:-1], hist_hed, width=45, align='center')
plt.title('Histogram of Edge Directions'); plt.xlabel('Edge Direction (degrees)'); plt.ylabel('Frequency')
plt.xticks(range(0, 360, 45))
plt.tight_layout(); plt.show()


# =============================================================================
# SECTION 5: HIG — HISTOGRAM OF INTENSITY GRADIENTS
# =============================================================================
# WHY: Captures distribution of gradient magnitudes/orientations directly.
# WHERE: Object recognition, texture analysis, classification.
# HOW: Compute gradients → magnitude & orientation → histogram.
#
# Difference from HED: HIG uses gradients directly (no explicit edge detection).

print("\n" + "=" * 60)
print("SECTION 5: HIG — Histogram of Intensity Gradients")
print("=" * 60)

# Step 1: Compute gradients using Sobel
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Step 2: Magnitude and orientation
magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

# Step 3: Histogram of orientations
hist_hig, bin_edges = np.histogram(orientation, bins=8, range=(0, 360))

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(magnitude, cmap='gray'); plt.title('Gradient Magnitude'); plt.axis('off')
plt.subplot(1, 2, 2); plt.bar(bin_edges[:-1], hist_hig, width=45, align='center')
plt.title('Histogram of Intensity Gradients'); plt.xlabel('Orientation (degrees)'); plt.ylabel('Frequency')
plt.xticks(range(0, 360, 45))
plt.tight_layout(); plt.show()


# =============================================================================
# SECTION 6: TEXTURE ENERGY AND CONTRAST HISTOGRAMS
# =============================================================================
# WHY: Quantifies texture roughness/smoothness.
# WHERE: Material classification (wood, metal, fabric).
# HOW: Local window → energy (sum of squares) + contrast (std dev) → histograms.
#
# cv2.filter2D(src, ddepth, kernel)
#   - ddepth: -1 → same as input
#   - kernel: averaging kernel (np.ones((k,k)))

print("\n" + "=" * 60)
print("SECTION 6: Texture Energy & Contrast")
print("=" * 60)

neighborhood_size = 3

# Step 1: Energy = sum of squared pixel values in local window
# image**2 → square pixel values; filter2D averages them
energy = cv2.filter2D(image.astype(np.float32) ** 2, -1,
                      np.ones((neighborhood_size, neighborhood_size)))

# Step 2: Contrast = std dev of pixel values in local window
# Approximation: filter2D of raw image (for visualization)
contrast = cv2.filter2D(image.astype(np.float32), -1,
                        np.ones((neighborhood_size, neighborhood_size)))

# Step 3: Histograms
energy_hist, energy_bins = np.histogram(energy, bins=256, range=(0, energy.max()))
contrast_hist, contrast_bins = np.histogram(contrast, bins=256, range=(0, contrast.max()))

# Normalize
energy_hist = energy_hist / (energy_hist.sum() + 1e-6)
contrast_hist = contrast_hist / (contrast_hist.sum() + 1e-6)

# Display
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(energy_bins[:-1], energy_hist, color='b')
plt.title("Texture Energy Histogram"); plt.xlabel("Energy"); plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.plot(contrast_bins[:-1], contrast_hist, color='r')
plt.title("Texture Contrast Histogram"); plt.xlabel("Contrast"); plt.ylabel("Frequency")
plt.tight_layout(); plt.show()


# =============================================================================
# SECTION 7: FILTERING & CONVOLUTION
# =============================================================================
# WHY: Modify pixel values using kernels for blur, sharpening, edge detection.
# WHERE: Noise reduction, edge detection, feature enhancement.
# HOW: Slide kernel over image → multiply + sum → new pixel value.
#
# cv2.filter2D(src, ddepth, kernel, borderType)
#   - src: input image
#   - ddepth: -1 → same depth as input
#   - kernel: filter matrix (float32)
#   - borderType: how to handle borders
#       cv2.BORDER_CONSTANT → zero padding
#       cv2.BORDER_REFLECT  → mirror padding

print("\n" + "=" * 60)
print("SECTION 7: Filtering & Convolution")
print("=" * 60)

# ---- 7.1 Box Blur (averaging) ----
# kernel = ones(k,k)/k²  → average of neighbors
box_kernel = np.ones((3, 3), dtype=np.float32) / 9
box_blurred = cv2.filter2D(image, -1, box_kernel)

# ---- 7.2 Gaussian Blur ----
# cv2.getGaussianKernel(ksize, sigma)
#   - ksize: kernel size (odd)
#   - sigma: standard deviation
gauss_kernel = cv2.getGaussianKernel(5, 1.0)
gaussian_blurred = cv2.filter2D(image, -1, gauss_kernel)

# ---- 7.3 Emboss (3D effect) ----
# Kernel emphasizes differences between neighbors
emboss_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=np.float32)
embossed = cv2.filter2D(image, -1, emboss_kernel)

# ---- Display all ----
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1); plt.imshow(image, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(1, 4, 2); plt.imshow(box_blurred, cmap='gray'); plt.title('Box Blur'); plt.axis('off')
plt.subplot(1, 4, 3); plt.imshow(gaussian_blurred, cmap='gray'); plt.title('Gaussian Blur'); plt.axis('off')
plt.subplot(1, 4, 4); plt.imshow(embossed, cmap='gray'); plt.title('Emboss'); plt.axis('off')
plt.tight_layout(); plt.show()


# =============================================================================
# SECTION 8: CONVOLUTION TYPES (Output Size)
# =============================================================================
# WHY: Different padding affects output size and boundary handling.
# WHERE: CNNs, image filtering.
#
# Output size formula: (W - K + 2P) / S + 1
#   W = input size, K = kernel size, P = padding, S = stride
#
# Valid (no padding): P = 0 → output smaller
# Same (zero padding): P = (K-1)/2 → output same as input

print("\n" + "=" * 60)
print("SECTION 8: Convolution Types")
print("=" * 60)

kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)

# Valid convolution (no padding)
valid_result = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

# Same convolution (with zero padding, simulated by BORDER_REFLECT)
same_result = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

print("Input shape:", image.shape)
print("Valid output shape:", valid_result.shape)   # ~ (H-2, W-2)
print("Same output shape:", same_result.shape)     # ~ (H, W)


# =============================================================================
# SECTION 9: EDGE DETECTION — SOBEL
# =============================================================================
# WHY: Gradient-based edge detection; simple and fast.
# WHERE: Preprocessing for object detection, segmentation.
# HOW: Sobel kernels in x and y → magnitude and direction.
#
# cv2.Sobel(src, ddepth, dx, dy, ksize)
#   - ddepth: cv2.CV_64F → preserve negative values
#   - dx, dy: 1 to compute derivative in that direction
#   - ksize: kernel size (3, 5, 7)

print("\n" + "=" * 60)
print("SECTION 9: Sobel Edge Detection")
print("=" * 60)

gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # horizontal
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # vertical

magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
direction = np.arctan2(gradient_y, gradient_x)

# Threshold magnitude to get edges
threshold = 100
edges_sobel = (magnitude > threshold).astype(np.uint8) * 255

plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1); plt.imshow(image, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(1, 4, 2); plt.imshow(np.abs(gradient_x), cmap='gray'); plt.title('Sobel X'); plt.axis('off')
plt.subplot(1, 4, 3); plt.imshow(np.abs(gradient_y), cmap='gray'); plt.title('Sobel Y'); plt.axis('off')
plt.subplot(1, 4, 4); plt.imshow(edges_sobel, cmap='gray'); plt.title('Sobel Magnitude'); plt.axis('off')
plt.tight_layout(); plt.show()


# =============================================================================
# SECTION 10: EDGE DETECTION — CANNY (MOST IMPORTANT)
# =============================================================================
# WHY: Best edge detector — accurate, thin edges, noise suppression.
# WHERE: Object detection, segmentation, feature extraction.
# HOW: 4 steps:
#   1. Gaussian Smoothing (noise reduction)
#   2. Gradient Calculation (Sobel)
#   3. Non-Maximum Suppression (thin edges)
#   4. Hysteresis Thresholding (2 thresholds)
#
# cv2.Canny(image, threshold1, threshold2)
#   - threshold1: T_low → below → rejected
#   - threshold2: T_high → above → strong edge
#   - Ratio T_high:T_low ≈ 2:1 or 3:1

print("\n" + "=" * 60)
print("SECTION 10: Canny Edge Detection")
print("=" * 60)

# Step 1: Gaussian blur (reduces noise)
blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

# Step 2–4: Canny
edges_canny = cv2.Canny(blurred, threshold1=50, threshold2=150)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1); plt.imshow(image, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(blurred, cmap='gray'); plt.title('Gaussian Blurred'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(edges_canny, cmap='gray'); plt.title('Canny Edges'); plt.axis('off')
plt.tight_layout(); plt.show()


# =============================================================================
# SECTION 11: EDGE DETECTION — LAPLACIAN OF GAUSSIAN (LoG)
# =============================================================================
# WHY: Second-derivative method; detects edges via zero-crossings.
# WHERE: Edge detection across scales.
# HOW: Gaussian smoothing → Laplacian → zero-crossings = edges.
#
# cv2.Laplacian(src, ddepth)
#   - ddepth: cv2.CV_64F to preserve signed values
# cv2.convertScaleAbs(src) → absolute + scale to uint8 for display

print("\n" + "=" * 60)
print("SECTION 11: Laplacian of Gaussian (LoG)")
print("=" * 60)

# Step 1: Gaussian blur
blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

# Step 2: Laplacian
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

# Step 3: Absolute + convert to uint8 for display
laplacian_abs = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1); plt.imshow(image, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(blurred, cmap='gray'); plt.title('Gaussian Blurred'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(laplacian_abs, cmap='gray'); plt.title('LoG'); plt.axis('off')
plt.tight_layout(); plt.show()


# =============================================================================
# SECTION 12: SCHARR OPERATOR (Stronger than Sobel)
# =============================================================================
# WHY: More accurate gradient approximation than Sobel.
# WHERE: When Sobel isn't sensitive enough.
#
# cv2.Scharr(src, ddepth, dx, dy)
#   - Similar to Sobel but uses different kernel weights

print("\n" + "=" * 60)
print("SECTION 12: Scharr Operator")
print("=" * 60)

scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1); plt.imshow(image, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(np.abs(scharr_x), cmap='gray'); plt.title('Scharr X'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(scharr_magnitude, cmap='gray'); plt.title('Scharr Magnitude'); plt.axis('off')
plt.tight_layout(); plt.show()


# =============================================================================
# END OF LAB 04 CODE
# =============================================================================
# QUICK REFERENCE — KEY FUNCTIONS & ARGUMENT ORDER
# =============================================================================
#
# HOG:
#   hog(image, pixels_per_cell, cells_per_block, visualize)
#
# LBP:
#   feature.local_binary_pattern(image, P, R, method)
#
# Color Histogram:
#   cv2.calcHist([image], [channel], mask, [bins], [range])
#
# Gaussian Blur:
#   cv2.GaussianBlur(src, ksize, sigmaX)
#
# Sobel:
#   cv2.Sobel(src, ddepth, dx, dy, ksize)
#
# Scharr:
#   cv2.Scharr(src, ddepth, dx, dy)
#
# Canny:
#   cv2.Canny(image, threshold1, threshold2)
#
# Laplacian:
#   cv2.Laplacian(src, ddepth)
#
# Convolution:
#   cv2.filter2D(src, ddepth, kernel, borderType)
#
# =============================================================================
