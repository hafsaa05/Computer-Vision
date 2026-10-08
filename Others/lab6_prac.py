"""
================================================================================
 LAB 6 — COMPUTER VISION TASKS (PYTHON FILE)
================================================================================

 TASKS COVERED:
    Task 1 — Computer Screen Detection          (Hough Lines)
    Task 2 — Asset Tracking in Lab              (SIFT + Homography)
    Task 3 — Sensor Anomaly Detection           (Wavelet Transform)
    Task 4 — Object Recognition in Video        (SIFT + FLANN)
    Task 5 — Panorama Image Stitching           (SIFT + Homography + Blend)
    Task 6 — Lane Detection                     (HoughLinesP)
    Task 7 — Coins Detection and Counting       (Hough Circles)
    Task 8 — Smart Security System              (MOG2 + Contours)

 COMMON PIPELINE (every task follows this):
    1. Read image/video
    2. Convert BGR -> Gray (or BGR -> RGB for display)
    3. Preprocess (Gaussian/Median blur)
    4. Detect (Canny / SIFT / Hough / MOG2)
    5. Post-process (RANSAC / morphology / polyfit / zone check)
    6. Draw results (rectangle / circle / polylines / putText)
    7. Save output (imwrite / VideoWriter)

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import cv2                          # OpenCV — core computer vision functions
import numpy as np                  # NumPy — arrays, math, linear algebra
import matplotlib.pyplot as plt     # Matplotlib — display images/graphs
import os                           # OS — create folders
import pywt                         # PyWavelets — wavelet transform (Task 3)

os.makedirs("outputs", exist_ok=True)


# ==============================================================================
# ────────────────────────────────────────────────────────────────────────────
#  TASK 1 — COMPUTER SCREEN DETECTION (Hough Lines)
# ────────────────────────────────────────────────────────────────────────────
#  Goal: Detect screens (rectangles) in a computer lab image.
#  Method: Canny edges -> HoughLines -> combine lines into rectangles.
#  Then check ON/OFF using HSV brightness.
# ==============================================================================

def task1_screen_detection(image_path):
    """
    FULL PROCESS:
        STEP 1 - Read image (BGR)
        STEP 2 - Convert BGR -> RGB (for matplotlib display)
        STEP 3 - Convert BGR -> Gray (for processing)
        STEP 4 - Gaussian blur -> remove noise
        STEP 5 - Canny edges (50, 150)
        STEP 6 - HoughLines -> (rho, theta) for each line
        STEP 7 - Split into horizontal (theta~90) and vertical (theta~0)
        STEP 8 - Remove duplicate lines (within 3 px)
        STEP 9 - Remove desk lines (90%+ row coverage)
        STEP 10 - Combine 2H + 2V lines -> rectangles (aspect + 4-side check)
        STEP 11 - Remove nested boxes (glass inside bezel)
        STEP 12 - HSV brightness -> ON/OFF
        STEP 13 - Draw boxes + labels
        STEP 14 - Save output
    """
    # ---- STEP 1: Read image ----
    image = cv2.imread(image_path)

    # ---- STEP 2: BGR -> RGB (matplotlib expects RGB) ----
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---- STEP 3: BGR -> Gray (processing on 1 channel) ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- STEP 4: Gaussian blur (sigma=1.4) ----
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # ---- STEP 5: Canny edges ----
    # cv2.Canny(image, low, high) — ratio usually 1:3
    edges = cv2.Canny(blur, 50, 150)

    # ---- STEP 6: HoughLines (standard) ----
    # cv2.HoughLines(image, rho, theta, threshold)
    #   image     : binary edge image
    #   rho=1     : distance resolution (1 px)
    #   theta=1°  : angle resolution
    #   threshold : minimum votes
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
    lines = lines.reshape(-1, 2)                 # each row is (rho, theta)

    # ---- STEP 7: Separate horizontal vs vertical ----
    ys, xs = [], []                              # y-values = horizontal, x-values = vertical
    for rho, theta in lines:
        deg = np.degrees(theta)
        if abs(deg - 90) < 2:                    # theta~90 -> horizontal
            ys.append(int(rho))
        elif deg < 2 or deg > 178:               # theta~0 or 180 -> vertical
            xs.append(int(abs(rho)))

    # ---- STEP 8: Remove duplicate lines ----
    def remove_close(vals, tol=3):
        keep = []
        for v in vals:
            if all(abs(v - k) > tol for k in keep):
                keep.append(v)
        return sorted(keep)

    ys = remove_close(ys)
    xs = remove_close(xs)

    # ---- STEP 9: Remove desk lines (rows that are 90%+ edges) ----
    d = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    ys = [y for y in ys if (d[y] > 0).mean() < 0.9]

    # ---- STEP 10: Combine lines into rectangles ----
    band = cv2.dilate(edges, np.ones((7, 7), np.uint8)) > 0
    boxes = []
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            y1, y2 = ys[i], ys[j]
            h = y2 - y1
            if h < 50 or h > 300:                # height filter
                continue
            for a in range(len(xs)):
                for b in range(a + 1, len(xs)):
                    x1, x2 = xs[a], xs[b]
                    w = x2 - x1
                    if w / h < 1.2 or w / h > 2.2:   # aspect ratio filter
                        continue
                    # Check all 4 sides have edge pixels
                    top = band[y1, x1:x2].mean()
                    bottom = band[y2, x1:x2].mean()
                    left = band[y1:y2, x1].mean()
                    right = band[y1:y2, x2].mean()
                    if min(top, bottom, left, right) > 0.85:
                        boxes.append((x1, y1, w, h))

    # ---- STEP 11: Remove nested boxes ----
    def inside(a, b):
        return (a[0] >= b[0] - 3 and a[1] >= b[1] - 3 and
                a[0] + a[2] <= b[0] + b[2] + 3 and
                a[1] + a[3] <= b[1] + b[3] + 3)

    screens = [a for a in boxes
               if not any(a != b and inside(a, b) for b in boxes)]

    # ---- STEP 12: HSV brightness -> ON/OFF ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    result = rgb.copy()
    for (x, y, w, h) in screens:
        # Sample center 2/3 region (avoid bezel)
        v = hsv[y + h // 6:y + h - h // 6,
                x + w // 6:x + w - w // 6, 2].mean()
        if v > 80:
            status, color = "ON", (0, 255, 0)     # green
        else:
            status, color = "OFF", (255, 165, 0)  # orange
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
        cv2.putText(result, status, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ---- STEP 13-14: Save ----
    cv2.imwrite("outputs/task1_screens.png",
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return result


# ==============================================================================
# ────────────────────────────────────────────────────────────────────────────
#  TASK 2 — ASSET TRACKING (SIFT + Homography)
# ────────────────────────────────────────────────────────────────────────────
#  Goal: Detect PC tags (PC-01, PC-02), monitors, keyboards in lab scene.
#  Then assign owner to each accessory.
# ==============================================================================

def task2_asset_tracking(scene_path, db_paths):
    """
    FULL PROCESS:
        STEP 1 - Read scene -> Gray
        STEP 2 - SIFT on scene -> keypoints + 128-dim descriptors
        STEP 3 - For each reference image (PC-01, MONITOR, etc.):
                   a) Read -> Gray -> SIFT -> store kp, des
        STEP 4 - Create BFMatcher (NORM_L2 for SIFT)
        STEP 5 - For each asset type:
                   a) knnMatch(k=2) -> Lowe ratio 0.75
                   b) Min 12 good matches
                   c) findHomography + RANSAC
                   d) Project 4 corners -> bounding box
                   e) Check convex + area
                   f) Mark matched keypoints dead (for multiple instances)
        STEP 6 - Draw boxes for each detection
        STEP 7 - Assign owner (nearest PC on right side)
        STEP 8 - Build inventory
    """
    # ---- STEP 1: Read scene ----
    scene = cv2.imread(scene_path)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    # ---- STEP 2: SIFT on scene ----
    sift = cv2.SIFT_create()
    kp_s, des_s = sift.detectAndCompute(scene_gray, None)

    # ---- STEP 3: SIFT on each reference image ----
    feats = {}
    for name, path in db_paths.items():
        im = cv2.imread(path)
        kp, des = sift.detectAndCompute(
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
        feats[name] = (kp, des, im.shape[:2])    # (kp, des, (h, w))

    # ---- STEP 4: BFMatcher ----
    # cv2.BFMatcher(normType) — NORM_L2 for SIFT float descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # ---- STEP 5: Match each asset ----
    found = []
    for name, (kp_r, des_r, (h, w)) in feats.items():
        # "alive" tracks scene keypoints not yet matched
        alive = np.ones(len(kp_s), bool)
        max_count = 1 if name.startswith("PC") else 6   # unique vs generic

        for t in range(max_count):
            idx = np.where(alive)[0]
            matches = bf.knnMatch(des_r, des_s[idx], k=2)

            # 5a: Lowe ratio test
            good = [m for m in matches
                    if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]

            # 5b: Min matches
            if len(good) < 12:
                break

            # 5c: Source & destination points
            src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_s[idx[m.trainIdx]].pt for m in good]).reshape(-1, 1, 2)

            # 5d: Homography + RANSAC
            M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if M is None or mask.sum() < 12:
                break

            # 5e: Project reference corners -> scene bounding box
            box = cv2.perspectiveTransform(
                np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2),
                M
            ).reshape(-1, 2)

            # 5f: Sanity check
            if (not cv2.isContourConvex(box.astype(np.int32)) or
                    cv2.contourArea(box) < 1500):
                break

            found.append((name, box, int(mask.sum())))

            # 5g: Mark matched keypoints as "dead"
            big = (box - box.mean(0)) * 1.1 + box.mean(0)
            for k in idx:
                if cv2.pointPolygonTest(big.astype(np.float32),
                                         kp_s[k].pt, False) >= 0:
                    alive[k] = False

    return found


# ==============================================================================
# ────────────────────────────────────────────────────────────────────────────
#  TASK 3 — SENSOR ANOMALY DETECTION (Wavelet Transform)
# ────────────────────────────────────────────────────────────────────────────
#  Goal: Denoise sensor signal, then find anomalies in the residual.
# ==============================================================================

def task3_wavelet_anomaly(signal, wavelet="db4", level=4):
    """
    FULL PROCESS:
        STEP 1 - pywt.wavedec() -> decompose signal into cA + cD coefficients
        STEP 2 - Universal threshold (Donoho-Johnstone)
                   sigma = MAD(finest cD) / 0.6745
                   thr   = sigma * sqrt(2 * ln(N))
        STEP 3 - Soft threshold on all cD (keep cA as-is)
        STEP 4 - pywt.waverec() -> reconstruct denoised signal
        STEP 5 - residual = original - denoised
        STEP 6 - limit = 3.5 * MAD(residual)
                 anomalies = |residual| > limit
        STEP 7 - Plot + save
    """
    N = len(signal)

    # ---- STEP 1: Decompose ----
    # pywt.wavedec(data, wavelet, level) -> [cA_n, cD_n, ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # ---- STEP 2: Universal threshold ----
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # finest detail = noise
    thr = sigma * np.sqrt(2 * np.log(N))

    # ---- STEP 3: Soft threshold on cD only ----
    # pywt.threshold(coeff, thr, mode) — "soft" or "hard"
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, thr, mode="soft"))

    # ---- STEP 4: Reconstruct ----
    # pywt.waverec(new_coeffs, wavelet) — inverse DWT
    denoised = pywt.waverec(new_coeffs, wavelet)[:N]

    # ---- STEP 5: Residual ----
    residual = signal - denoised

    # ---- STEP 6: Detect anomalies with MAD (robust to outliers) ----
    mad = np.median(np.abs(residual - np.median(residual))) / 0.6745
    limit = 3.5 * mad
    anomalies = np.where(np.abs(residual) > limit)[0]

    # ---- STEP 7: Plot ----
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label="Sensor Data")
    plt.plot(denoised, "--", label="Denoised")
    plt.legend(); plt.title("Sensor Data and Denoised Signal")
    plt.subplot(2, 1, 2)
    plt.plot(residual, "r", label="Residuals")
    plt.scatter(anomalies, residual[anomalies],
                c="g", zorder=3, label="Anomalies")
    plt.legend(); plt.title("Residuals and Detected Anomalies")
    plt.tight_layout()
    plt.savefig("outputs/task3_anomalies.png")
    plt.show()

    return anomalies


# ==============================================================================
# ────────────────────────────────────────────────────────────────────────────
#  TASK 4 — OBJECT RECOGNITION IN VIDEO (SIFT + FLANN)
# ────────────────────────────────────────────────────────────────────────────
#  Goal: Find object (from reference image) in every video frame.
# ==============================================================================

def task4_video_object_recognition(ref_path, video_path):
    """
    FULL PROCESS:
        STEP 1 - Read reference -> Gray -> SIFT -> keypoints + descriptors
        STEP 2 - Store 4 corners of reference image
        STEP 3 - Create FLANN matcher (KD-Tree)
        STEP 4 - For each video frame:
                   a) Read frame -> Gray -> SIFT
                   b) FLANN knnMatch(k=2) -> Lowe ratio 0.7
                   c) Min 10 good matches
                   d) findHomography + RANSAC (5.0)
                   e) Project 4 corners -> bounding box
                   f) Check convex + inliers >= 10
                   g) Draw green box + status text
        STEP 5 - Write to output video
        STEP 6 - Release resources
    """
    # ---- STEP 1: Reference features ----
    ref = cv2.imread(ref_path)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_gray, None)

    # ---- STEP 2: Reference corners ----
    rh, rw = ref_gray.shape
    ref_corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)

    # ---- STEP 3: FLANN matcher ----
    # cv2.FlannBasedMatcher(index_params, search_params)
    #   algorithm=1  : KD-Tree
    #   trees=5      : 5 KD-trees
    #   checks=50    : 50 nodes per query
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),
                                    dict(checks=50))

    # ---- STEP 4: detect() function per frame ----
    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray, None)
        out = frame.copy()
        found, inliers = False, 0

        if des2 is not None and len(kp2) > 2:
            # 4a: knnMatch k=2
            matches = flann.knnMatch(des1, des2, k=2)
            # 4b: Lowe ratio 0.7
            good = [m for m in matches
                    if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

            # 4c: Min matches
            if len(good) >= 10:
                src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # 4d: Homography
                M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

                if M is not None:
                    inliers = int(mask.sum())
                    box = cv2.perspectiveTransform(ref_corners, M)

                    # 4e-f: Convex + inliers check
                    if inliers >= 10 and cv2.isContourConvex(np.int32(box)):
                        found = True
                        cv2.polylines(out, [np.int32(box)], True,
                                       (0, 255, 0), 3)

        # 4g: Status text
        text = "found " + str(inliers) if found else "not found"
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(out, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return out, inliers, found

    # ---- STEP 5: Video processing ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(3))
    H = int(cap.get(4))
    out_vid = cv2.VideoWriter("outputs/task4_object_recognition.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (W, H))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res, _, _ = detect(frame)
        out_vid.write(res)

    # ---- STEP 6: Release ----
    cap.release()
    out_vid.release()
    print("Task 4 done.")


# ==============================================================================
# ────────────────────────────────────────────────────────────────────────────
#  TASK 5 — PANORAMA STITCHING (SIFT + Homography + Blend)
# ────────────────────────────────────────────────────────────────────────────
#  Goal: Stitch 3 overlapping images into one panorama.
#  Anchor image = middle (index 1). Left + right warped onto it.
# ==============================================================================

def task5_panorama(image_paths):
    """
    FULL PROCESS:
        STEP 1 - Read each image -> resize to width=1000
        STEP 2 - SIFT features on each image
        STEP 3 - Compute homography H(a->b) for:
                   H1 = image0 -> image1 (left -> middle)
                   H3 = image2 -> image1 (right -> middle)
        STEP 4 - Hs = [H1, np.eye(3), H3]  (middle stays unchanged)
        STEP 5 - Compute canvas bounds (warp 4 corners of each image)
        STEP 6 - Translation matrix T (shift negatives to 0)
        STEP 7 - Warp each image + compute weight via distanceTransform
        STEP 8 - Weighted average: pano = sum(wp * wt) / sum(wt)
        STEP 9 - Crop black borders (98% valid rows/cols)
        STEP 10 - Save panorama
    """
    # ---- STEP 1: Load + resize ----
    imgs = []
    for p in image_paths:
        im = cv2.imread(p)
        s = 1000 / im.shape[1]
        imgs.append(cv2.resize(im, None, fx=s, fy=s))

    # ---- STEP 2: SIFT ----
    sift = cv2.SIFT_create()
    kps, dess = [], []
    for im in imgs:
        kp, des = sift.detectAndCompute(
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
        kps.append(kp)
        dess.append(des)

    # ---- STEP 3: get_H(a, b) helper ----
    bf = cv2.BFMatcher()

    def get_H(a, b):
        """Return H that maps image a -> image b."""
        matches = bf.knnMatch(dess[a], dess[b], k=2)
        good = [m for m, n in matches
                if m.distance < 0.75 * n.distance]
        src = np.float32([kps[a][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kps[b][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        return H

    # ---- STEP 4: Anchor = middle image ----
    H1 = get_H(0, 1)                 # left -> middle
    H3 = get_H(2, 1)                 # right -> middle
    Hs = [H1, np.eye(3), H3]         # middle = identity (no warp)

    # ---- STEP 5: Canvas bounds ----
    all_pts = []
    for im, H in zip(imgs, Hs):
        h, w = im.shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        all_pts.append(cv2.perspectiveTransform(pts, H))
    all_pts = np.concatenate(all_pts)
    xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    # ---- STEP 6: Translation matrix ----
    # T shifts all coordinates so none are negative
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=float)
    size = (xmax - xmin, ymax - ymin)

    # ---- STEP 7: Warp + weights ----
    warped, weights = [], []
    for im, H in zip(imgs, Hs):
        wp = cv2.warpPerspective(im, T @ H, size).astype(np.float32)
        m = cv2.warpPerspective(np.ones(im.shape[:2], np.uint8),
                                 T @ H, size)
        # distanceTransform: center of image gets higher weight
        wt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        warped.append(wp)
        weights.append(wt)

    # ---- STEP 8: Weighted average ----
    total = weights[0] + weights[1] + weights[2]
    pano = np.zeros_like(warped[0])
    for wp, wt in zip(warped, weights):
        pano += wp * wt[:, :, None]
    pano = (pano / np.maximum(total, 1e-6)[:, :, None]).astype(np.uint8)

    # ---- STEP 9: Crop black borders ----
    valid = total > 0
    rows = np.where(valid.mean(axis=1) > 0.98)[0]
    cols = np.where(valid[rows[0]:rows[-1]].mean(axis=0) > 0.98)[0]
    final = pano[rows[0]:rows[-1], cols[0]:cols[-1]]

    # ---- STEP 10: Save ----
    cv2.imwrite("outputs/task5_panorama.jpg", final)
    return final


# ==============================================================================
# ────────────────────────────────────────────────────────────────────────────
#  TASK 6 — LANE DETECTION (HoughLinesP + Polyfit)
# ────────────────────────────────────────────────────────────────────────────
#  Goal: Detect left + right lane lines on road.
# ==============================================================================

def lane_detect(img):
    """
    FULL PROCESS:
        STEP 1 - Gray + Gaussian blur + Canny
        STEP 2 - ROI (trapezoid) mask — only road area
        STEP 3 - HoughLinesP -> segments (x1, y1, x2, y2)
        STEP 4 - Classify by slope:
                   slope < -0.5  -> LEFT lane
                   slope >  0.5  -> RIGHT lane
        STEP 5 - np.polyfit(ys, xs, 1) -> x = a*y + b (average line)
        STEP 6 - Draw thick red lines + green filled polygon
        STEP 7 - addWeighted overlay
    """
    h, w = img.shape[:2]

    # ---- STEP 1: Preprocess ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ---- STEP 2: ROI (trapezoid) ----
    # Bottom wide, top narrow (perspective effect)
    roi = np.array([[(int(0.05 * w), h),
                     (int(0.45 * w), int(0.6 * h)),
                     (int(0.55 * w), int(0.6 * h)),
                     (int(0.97 * w), h)]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    # ---- STEP 3: HoughLinesP ----
    # cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
    lines = cv2.HoughLinesP(roi_edges, 2, np.pi / 180, 40,
                             minLineLength=30, maxLineGap=150)
    if lines is None:
        return img, [None, None]
    lines = lines.reshape(-1, 4)

    # ---- STEP 4: Classify by slope ----
    top = int(0.62 * h)
    left, right = [], []
    for x1, y1, x2, y2 in lines:
        if x1 == x2:                       # skip vertical
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.5:                   # left lane
            left.append((x1, y1, x2, y2))
        elif slope > 0.5:                  # right lane
            right.append((x1, y1, x2, y2))

    # ---- STEP 5: Average line with polyfit (x = a*y + b) ----
    lanes = []
    for side in [left, right]:
        if len(side) == 0:
            lanes.append(None)
            continue
        xs, ys = [], []
        for x1, y1, x2, y2 in side:
            xs += [x1, x2]
            ys += [y1, y2]
        a, b = np.polyfit(ys, xs, 1)        # x = a*y + b
        lanes.append((int(a * h + b), h, int(a * top + b), top))

    # ---- STEP 6: Draw ----
    layer = np.zeros_like(img)
    for l in lanes:
        if l is not None:
            cv2.line(layer, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 12)
    # Fill polygon between left and right lanes
    if lanes[0] is not None and lanes[1] is not None:
        L, R = lanes
        pts = np.array([[(L[0], L[1]), (L[2], L[3]),
                         (R[2], R[3]), (R[0], R[1])]], np.int32)
        cv2.fillPoly(layer, pts, (0, 70, 0))

    # ---- STEP 7: Blend ----
    result = cv2.addWeighted(img, 1, layer, 0.6, 0)
    return result, lanes


def task6_lane_video(video_path):
    """
    Process video with TEMPORAL SMOOTHING:
        70% previous lane + 30% new lane -> smooth transitions.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(3))
    H = int(cap.get(4))
    out = cv2.VideoWriter("outputs/task6_lanes_video.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (W, H))

    prev = [None, None]
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, lanes = lane_detect(frame)

        # Temporal smoothing
        new = []
        for p, l in zip(prev, lanes):
            if l is None:
                new.append(p)                  # keep old
            elif p is None:
                new.append(l)                  # first detection
            else:
                new.append(tuple(int(0.7 * a + 0.3 * b)
                                 for a, b in zip(p, l)))
        prev = new

        res, _ = lane_detect(frame)
        out.write(res)

    cap.release()
    out.release()


# ==============================================================================
# ────────────────────────────────────────────────────────────────────────────
#  TASK 7 — COINS DETECTION & COUNTING (Hough Circles)
# ────────────────────────────────────────────────────────────────────────────
#  Goal: Detect all coins + classify small vs big.
# ==============================================================================

def task7_coin_detection(image_path):
    """
    FULL PROCESS:
        STEP 1 - Read image -> Gray
        STEP 2 - medianBlur(5) — removes coin design, keeps outer circle
        STEP 3 - HoughCircles -> (x, y, r) per circle
        STEP 4 - Draw circles + center dots + numbers + count
        STEP 5 - Classify small vs big via 2-means on radius
        STEP 6 - Save output
    """
    # ---- STEP 1: Read + Gray ----
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- STEP 2: Median blur ----
    # cv2.medianBlur(gray, 5) — 5x5 kernel
    blur = cv2.medianBlur(gray, 5)

    # ---- STEP 3: HoughCircles ----
    # cv2.HoughCircles(image, method, dp, minDist,
    #                  param1, param2, minRadius, maxRadius)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
                                dp=1.2, minDist=25,
                                param1=100, param2=30,
                                minRadius=12, maxRadius=40)
    if circles is None:
        return img, []
    circles = np.round(circles.reshape(-1, 3)).astype(int)

    # ---- STEP 4: Draw ----
    out = img.copy()
    n = 1
    for x, y, r in circles:
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)       # green outer
        cv2.circle(out, (x, y), 2, (0, 0, 255), 3)       # red center
        cv2.putText(out, str(n), (x - 8, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        n += 1
    cv2.putText(out, "Count: " + str(len(circles)),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)
    cv2.imwrite("outputs/task7_coins.png", out)

    # ---- STEP 5: 2-means on radius (small vs big) ----
    radius = circles[:, 2].astype(float)
    c1, c2 = radius.min(), radius.max()
    for _ in range(20):
        big = np.abs(radius - c1) > np.abs(radius - c2)
        c1 = radius[~big].mean()
        c2 = radius[big].mean()

    return out, (c1, c2, big)


# ==============================================================================
# ────────────────────────────────────────────────────────────────────────────
#  TASK 8 — SMART SECURITY SYSTEM (MOG2 + Zone Check)
# ────────────────────────────────────────────────────────────────────────────
#  Goal: Detect any object entering the security zone -> trigger alarm.
# ==============================================================================

def task8_security_system(video_path, zone_points):
    """
    FULL PROCESS:
        STEP 1 - Define zone polygon (4 points)
        STEP 2 - Create zone_mask (binary image of zone)
        STEP 3 - Create MOG2 background subtractor
        STEP 4 - Create morphological kernels (small + big)
        STEP 5 - For each frame:
                   a) bg.apply() -> foreground mask
                   b) First 60 frames: auto learning
                   c) After 60: freeze (learningRate=0.0001)
                   d) Threshold 200 (remove shadows)
                   e) MORPH_OPEN (small kernel) — noise
                   f) MORPH_CLOSE (big kernel)  — holes
                   g) dilate x2
                   h) findContours -> filter area < 350
                   i) For each contour:
                        - foot point test (pointPolygonTest)
                        - overlap test (>30%)
                   j) If in zone -> RED alarm
        STEP 6 - Write output video
        STEP 7 - Release resources
    """
    # ---- STEP 1: Zone polygon ----
    zone = np.array(zone_points, np.int32)

    # ---- STEP 2: Video setup ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(3))
    H = int(cap.get(4))
    out = cv2.VideoWriter("outputs/task8_security.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (W, H))

    # ---- STEP 3: Zone mask ----
    zone_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(zone_mask, [zone], 255)

    # ---- STEP 4: MOG2 background subtractor ----
    # cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
    bg = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=True)

    # ---- STEP 5: Morphological kernels ----
    # cv2.getStructuringElement(shape, ksize)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   # small
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))   # big

    # ---- STEP 6: Main loop ----
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 6a-6c: Background subtraction with learning rate
        if n < 60:
            fg = bg.apply(frame)                         # learn bg
        else:
            fg = bg.apply(frame, learningRate=0.0001)    # freeze

        # 6d: Threshold — remove shadows (grey -> 0)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # 6e: MORPH_OPEN — remove small noise
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k1)

        # 6f: MORPH_CLOSE — fill holes
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k2)

        # 6g: Dilate (make objects bigger for contour)
        fg = cv2.dilate(fg, None, iterations=2)

        # 6h: Find contours
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        res = frame.copy()
        in_zone = 0

        if n >= 60:
            for c in contours:
                if cv2.contourArea(c) < 350:              # skip tiny
                    continue

                x, y, w, h = cv2.boundingRect(c)

                # 6i-1: Foot point test (bottom-center)
                foot = (x + w // 2, y + h)
                foot_in = cv2.pointPolygonTest(zone, foot, False) >= 0

                # 6i-2: Overlap test
                obj = fg[y:y + h, x:x + w]
                overlap = (np.count_nonzero(obj & zone_mask[y:y + h, x:x + w])
                           / max(1, np.count_nonzero(obj)))

                if foot_in or overlap > 0.3:
                    in_zone += 1
                    color = (0, 0, 255)                   # red
                else:
                    color = (0, 255, 0)                   # green
                cv2.rectangle(res, (x, y), (x + w, y + h), color, 1)

        # 6j: Alarm banner
        if in_zone > 0:
            cv2.polylines(res, [zone], True, (0, 0, 255), 3)
            cv2.rectangle(res, (0, 0), (W, 45), (0, 0, 255), -1)
            cv2.putText(res, "ALARM! object in security zone",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
        else:
            cv2.polylines(res, [zone], True, (0, 200, 255), 2)
            cv2.putText(res, "clear", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        out.write(res)
        n += 1

    # ---- STEP 7: Release ----
    cap.release()
    out.release()
    print("Task 8 done. Frames:", n)


# ==============================================================================
# MAIN — UNCOMMENT THE TASK YOU WANT TO RUN
# ==============================================================================

if __name__ == "__main__":
    # TASK 1
    # task1_screen_detection("images/computer_lab.png")

    # TASK 2
    # task2_asset_tracking("images/lab_scene.png", {
    #     "PC-01": "images/pc01.png",
    #     "PC-02": "images/pc02.png",
    #     "MONITOR": "images/monitor.png",
    #     "KEYBOARD": "images/keyboard.png",
    # })

    # TASK 3
    # data = np.random.normal(0, 1, 1000)
    # task3_wavelet_anomaly(data)

    # TASK 4
    # task4_video_object_recognition("images/box.png", "images/test_video.mp4")

    # TASK 5
    # task5_panorama(["images/boat1.jpg", "images/boat2.jpg", "images/boat3.jpg"])

    # TASK 6
    # task6_lane_video("images/solidWhiteRight.mp4")

    # TASK 7
    # task7_coin_detection("images/coins.png")

    # TASK 8
    # task8_security_system("images/vtest.avi",
    #     [[360, 385], [560, 405], [610, 525], [330, 525]])

    pass
