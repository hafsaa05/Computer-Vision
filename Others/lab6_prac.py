"""
========================================================================
LAB 4 - Computer Vision Tasks
========================================================================
Topics covered:
  - Hough Line Transform  (Tasks 1, 6)
  - Hough Circle Transform (Task 7)
  - SIFT Feature Matching  (Tasks 2, 4, 5)
  - Wavelet Transform      (Task 3)
  - MOG2 Background Sub    (Task 8)
========================================================================
"""

# ======================================================================
# IMPORTS
# ======================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import pywt  # wavelet transform

# Create output folder
os.makedirs("outputs", exist_ok=True)


# ======================================================================
# COMMON HELPER FUNCTIONS
# ======================================================================

def load_image(path, grayscale=True):
    """
    Load an image from path.
    
    Args:
        path (str): image file path
        grayscale (bool): convert to grayscale?
    
    Returns:
        If grayscale: 2D array (H, W)
        Else: 3D array (H, W, 3) in BGR
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found: " + path)
    if grayscale:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def preprocess_for_edges(gray, blur_ksize=(5, 5), sigma=1.4,
                         canny_low=50, canny_high=150):
    """
    Standard pipeline: Gaussian blur + Canny edges.
    
    Args:
        gray:        grayscale image
        blur_ksize:  Gaussian kernel size (w, h) - must be odd
        sigma:       Gaussian sigma (X std deviation)
        canny_low:   Canny lower threshold
        canny_high:  Canny upper threshold (usually 2x or 3x low)
    
    Returns:
        edges: binary edge image (0 or 255)
    """
    # STEP 1: Gaussian blur - removes noise
    blur = cv2.GaussianBlur(gray, blur_ksize, sigma)
    
    # STEP 2: Canny - produces thin, clean edges
    edges = cv2.Canny(blur, canny_low, canny_high)
    return edges


def show(images, titles, grid=(1, 1), figsize=(16, 6)):
    """
    Display multiple images side by side.
    
    Args:
        images: list of images (BGR or gray)
        titles: list of strings
        grid:   tuple (rows, cols)
        figsize: figure size
    """
    plt.figure(figsize=figsize)
    for i, (im, t) in enumerate(zip(images, titles)):
        plt.subplot(grid[0], grid[1], i + 1)
        if len(im.shape) == 2:
            plt.imshow(im, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ======================================================================
# TASK 1: COMPUTER SCREEN DETECTION (Hough Lines)
# ======================================================================

def task1_screen_detection(image_path):
    """
    Detect computer screens in a lab image using Hough Line Transform.
    
    Pipeline:
        1. Read image
        2. Convert to grayscale
        3. Gaussian blur (reduce noise)
        4. Canny edges
        5. HoughLines (standard, polar form)
        6. Filter horizontal (theta ~ 90) and vertical (theta ~ 0/180) lines
        7. Remove duplicate lines
        8. Form rectangles by combining 2 H + 2 V lines
        9. Check ON/OFF using HSV brightness
        10. Draw boxes + labels
    """
    # ---- STEP 1: Load ----
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ---- STEP 2: Preprocess ----
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blur, 50, 150)
    
    # ---- STEP 3: Hough Lines ----
    # Args:  image, rho=1, theta=pi/180, threshold=60
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
    lines = lines.reshape(-1, 2)
    
    # ---- STEP 4: Split horizontal and vertical lines ----
    ys, xs = [], []  # y-values for horizontal, x-values for vertical
    for rho, theta in lines:
        deg = np.degrees(theta)
        if abs(deg - 90) < 2:            # horizontal line
            ys.append(int(rho))
        elif deg < 2 or deg > 178:       # vertical line
            xs.append(int(abs(rho)))
    
    # ---- STEP 5: Remove duplicates (within 3px) ----
    def remove_close(vals):
        keep = []
        for v in vals:
            if all(abs(v - k) > 3 for k in keep):
                keep.append(v)
        return sorted(keep)
    
    ys = remove_close(ys)
    xs = remove_close(xs)
    
    # ---- STEP 6: Remove desk lines (full-width) ----
    d = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    ys = [y for y in ys if (d[y] > 0).mean() < 0.9]
    
    # ---- STEP 7: Find rectangles ----
    band = cv2.dilate(edges, np.ones((7, 7), np.uint8)) > 0
    boxes = []
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            y1, y2 = ys[i], ys[j]
            h = y2 - y1
            if h < 50 or h > 300:
                continue
            for a in range(len(xs)):
                for b in range(a + 1, len(xs)):
                    x1, x2 = xs[a], xs[b]
                    w = x2 - x1
                    if w / h < 1.2 or w / h > 2.2:
                        continue
                    top = band[y1, x1:x2].mean()
                    bottom = band[y2, x1:x2].mean()
                    left = band[y1:y2, x1].mean()
                    right = band[y1:y2, x2].mean()
                    if min(top, bottom, left, right) > 0.85:
                        boxes.append((x1, y1, w, h))
    
    # ---- STEP 8: Remove nested boxes ----
    def inside(a, b):
        return (a[0] >= b[0]-3 and a[1] >= b[1]-3 and
                a[0]+a[2] <= b[0]+b[2]+3 and a[1]+a[3] <= b[1]+b[3]+3)
    
    screens = [a for a in boxes
               if not any(a != b and inside(a, b) for b in boxes)]
    
    # ---- STEP 9: Check ON/OFF via HSV ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    result = rgb.copy()
    for (x, y, w, h) in screens:
        v = hsv[y+h//6:y+h-h//6, x+w//6:x+w-w//6, 2].mean()
        color = (0, 255, 0) if v > 80 else (255, 165, 0)
        status = "ON" if v > 80 else "OFF"
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 3)
        cv2.putText(result, status, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # ---- STEP 10: Save ----
    cv2.imwrite("outputs/task1_screens.png",
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return result


# ======================================================================
# TASK 2, 4, 5: SIFT-BASED TASKS
# ======================================================================

def task2_asset_tracking(scene_path, db_paths):
    """
    Track PC assets using SIFT.
    
    db_paths: dict {name: image_path} - reference images
    """
    # ---- STEP 1: Scene features ----
    scene = cv2.imread(scene_path)
    sift = cv2.SIFT_create()
    gray_s = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    kp_s, des_s = sift.detectAndCompute(gray_s, None)
    
    # ---- STEP 2: Reference features ----
    feats = {}
    for name, path in db_paths.items():
        im = cv2.imread(path)
        kp, des = sift.detectAndCompute(
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
        feats[name] = (kp, des)
    
    # ---- STEP 3: Matcher ----
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # ---- STEP 4: Match each asset ----
    found = []
    for name, (kp_r, des_r) in feats.items():
        # Lowe's ratio test with k=2
        matches = bf.knnMatch(des_r, des_s, k=2)
        good = [m for m, n in matches
                if m.distance < 0.75 * n.distance]
        
        if len(good) < 12:
            continue
        
        src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        
        if M is None:
            continue
        
        h, w = db_paths[name].shape[:2] if False else (100, 100)
        box = cv2.perspectiveTransform(
            np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1, 1, 2),
            M
        ).reshape(-1, 2)
        found.append((name, box, int(mask.sum())))
    
    return found


def task4_video_object_recognition(ref_path, video_path):
    """
    Recognize object in each video frame using SIFT + FLANN.
    """
    ref = cv2.imread(ref_path)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    
    rh, rw = ref.shape[:2]
    ref_corners = np.float32(
        [[0,0], [rw,0], [rw,rh], [0,rh]]).reshape(-1, 1, 2)
    
    # FLANN: KD-Tree (algorithm=1), 5 trees, 50 checks
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),
                                   dict(checks=50))
    
    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray, None)
        out = frame.copy()
        found, inliers = False, 0
        
        if des2 is not None and len(kp2) > 2:
            matches = flann.knnMatch(des1, des2, k=2)
            good = [m for m in matches
                    if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
            
            if len(good) >= 10:
                src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if M is not None:
                    inliers = int(mask.sum())
                    box = cv2.perspectiveTransform(ref_corners, M)
                    if inliers >= 10 and cv2.isContourConvex(np.int32(box)):
                        found = True
                        cv2.polylines(out, [np.int32(box)], True,
                                       (0, 255, 0), 3)
        
        text = "found " + str(inliers) if found else "not found"
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(out, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return out, inliers, found
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(3))
    H = int(cap.get(4))
    out_vid = cv2.VideoWriter("outputs/task4.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (W, H))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res, _, _ = detect(frame)
        out_vid.write(res)
    
    cap.release()
    out_vid.release()


def task5_panorama(image_paths):
    """
    Stitch multiple overlapping images into a panorama.
    """
    # ---- STEP 1: Load + resize ----
    imgs = []
    for p in image_paths:
        im = cv2.imread(p)
        s = 1000 / im.shape[1]
        imgs.append(cv2.resize(im, None, fx=s, fy=s))
    
    # ---- STEP 2: SIFT features ----
    sift = cv2.SIFT_create()
    kps, dess = [], []
    for im in imgs:
        kp, des = sift.detectAndCompute(
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
        kps.append(kp)
        dess.append(des)
    
    # ---- STEP 3: Homography helper ----
    bf = cv2.BFMatcher()
    
    def get_H(a, b):
        matches = bf.knnMatch(dess[a], dess[b], k=2)
        good = [m for m, n in matches
                if m.distance < 0.75 * n.distance]
        src = np.float32([kps[a][m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kps[b][m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        return H
    
    # Anchor = middle image
    H1 = get_H(0, 1)
    H3 = get_H(2, 1)
    Hs = [H1, np.eye(3), H3]
    
    # ---- STEP 4: Canvas bounds ----
    all_pts = []
    for im, H in zip(imgs, Hs):
        h, w = im.shape[:2]
        pts = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
        all_pts.append(cv2.perspectiveTransform(pts, H))
    all_pts = np.concatenate(all_pts)
    xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    T = np.array([[1,0,-xmin], [0,1,-ymin], [0,0,1]], dtype=float)
    size = (xmax - xmin, ymax - ymin)
    
    # ---- STEP 5: Warp + weighted blend ----
    warped, weights = [], []
    for im, H in zip(imgs, Hs):
        wp = cv2.warpPerspective(im, T @ H, size).astype(np.float32)
        m = cv2.warpPerspective(np.ones(im.shape[:2], np.uint8),
                                 T @ H, size)
        wt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        warped.append(wp)
        weights.append(wt)
    
    total = weights[0] + weights[1] + weights[2]
    pano = np.zeros_like(warped[0])
    for wp, wt in zip(warped, weights):
        pano += wp * wt[:, :, None]
    pano = (pano / np.maximum(total, 1e-6)[:, :, None]).astype(np.uint8)
    
    # ---- STEP 6: Crop black borders ----
    valid = total > 0
    rows = np.where(valid.mean(axis=1) > 0.98)[0]
    cols = np.where(valid[rows[0]:rows[-1]].mean(axis=0) > 0.98)[0]
    final = pano[rows[0]:rows[-1], cols[0]:cols[-1]]
    
    cv2.imwrite("outputs/task5_panorama.jpg", final)
    return final


# ======================================================================
# TASK 3: WAVELET ANOMALY DETECTION
# ======================================================================

def task3_wavelet_anomaly(signal, wavelet="db4", level=4):
    """
    Detect anomalies in sensor data using wavelet transform.
    
    Args:
        signal: 1D array of sensor readings
        wavelet: wavelet family (db4 = Daubechies-4)
        level: decomposition levels
    
    Returns:
        anomalies: indices of detected anomalies
    """
    N = len(signal)
    
    # ---- STEP 1: Wavelet decomposition ----
    # Output: [cA_n, cD_n, cD_n-1, ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # ---- STEP 2: Universal threshold (Donoho-Johnstone) ----
    # sigma from finest detail coefficients (mostly noise)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(N))
    
    # ---- STEP 3: Soft threshold on detail coeffs ----
    # Keep approximation (cA) untouched
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, thr, mode="soft"))
    
    # ---- STEP 4: Reconstruct denoised signal ----
    denoised = pywt.waverec(new_coeffs, wavelet)[:N]
    
    # ---- STEP 5: Residual = original - denoised ----
    residual = signal - denoised
    
    # ---- STEP 6: Anomaly detection using MAD (robust) ----
    mad = np.median(np.abs(residual - np.median(residual))) / 0.6745
    limit = 3.5 * mad
    anomalies = np.where(np.abs(residual) > limit)[0]
    
    # ---- STEP 7: Visualize ----
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label="Sensor Data")
    plt.plot(denoised, "--", label="Denoised")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(residual, "r", label="Residuals")
    plt.scatter(anomalies, residual[anomalies],
                c="g", zorder=3, label="Anomalies")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/task3_anomalies.png")
    plt.show()
    
    return anomalies


# ======================================================================
# TASK 6: LANE DETECTION (HoughLinesP)
# ======================================================================

def lane_detect(img):
    """
    Detect lane lines using HoughLinesP + polyfit averaging.
    
    Args:
        img: BGR image
    
    Returns:
        result: image with lanes drawn
        lanes:  [left_lane, right_lane] or [None, None]
    """
    h, w = img.shape[:2]
    
    # ---- STEP 1: Preprocess ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # ---- STEP 2: Region of Interest (trapezoid) ----
    roi = np.array([[(int(0.05*w), h),
                     (int(0.45*w), int(0.6*h)),
                     (int(0.55*w), int(0.6*h)),
                     (int(0.97*w), h)]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, 255)
    roi_edges = cv2.bitwise_and(edges, mask)
    
    # ---- STEP 3: HoughLinesP ----
    # Args: image, rho=2, theta=pi/180, threshold=40,
    #       minLineLength=30, maxLineGap=150
    lines = cv2.HoughLinesP(roi_edges, 2, np.pi/180, 40,
                             minLineLength=30, maxLineGap=150)
    if lines is None:
        return img, [None, None]
    lines = lines.reshape(-1, 4)
    
    # ---- STEP 4: Classify left/right by slope ----
    top = int(0.62 * h)
    left, right = [], []
    for x1, y1, x2, y2 in lines:
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.5:
            left.append((x1, y1, x2, y2))
        elif slope > 0.5:
            right.append((x1, y1, x2, y2))
    
    # ---- STEP 5: Average via polyfit (x = a*y + b) ----
    lanes = []
    for side in [left, right]:
        if len(side) == 0:
            lanes.append(None)
            continue
        xs, ys = [], []
        for x1, y1, x2, y2 in side:
            xs += [x1, x2]
            ys += [y1, y2]
        a, b = np.polyfit(ys, xs, 1)
        lanes.append((int(a*h + b), h, int(a*top + b), top))
    
    # ---- STEP 6: Draw ----
    layer = np.zeros_like(img)
    for l in lanes:
        if l is not None:
            cv2.line(layer, (l[0], l[1]), (l[2], l[3]),
                     (0, 0, 255), 12)
    if lanes[0] is not None and lanes[1] is not None:
        L, R = lanes
        pts = np.array([[(L[0], L[1]), (L[2], L[3]),
                         (R[2], R[3]), (R[0], R[1])]], np.int32)
        cv2.fillPoly(layer, pts, (0, 70, 0))
    result = cv2.addWeighted(img, 1, layer, 0.6, 0)
    return result, lanes


def task6_lane_video(video_path):
    """
    Process video frame-by-frame with temporal smoothing.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(3))
    H = int(cap.get(4))
    out = cv2.VideoWriter("outputs/task6_lanes.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (W, H))
    
    prev = [None, None]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        _, lanes = lane_detect(frame)
        
        # Temporal smoothing: 70% old + 30% new
        new = []
        for p, l in zip(prev, lanes):
            if l is None:
                new.append(p)
            elif p is None:
                new.append(l)
            else:
                new.append(tuple(int(0.7*a + 0.3*b)
                                 for a, b in zip(p, l)))
        prev = new
        
        result, _ = lane_detect(frame)  # recompute or reuse
        out.write(result)
    
    cap.release()
    out.release()


# ======================================================================
# TASK 7: COIN DETECTION (Hough Circles)
# ======================================================================

def task7_coin_detection(image_path):
    """
    Detect and count coins using Hough Circle Transform.
    """
    # ---- STEP 1: Load + grayscale ----
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ---- STEP 2: Median blur ----
    # Median removes coin design (internal details) 
    # while keeping the outer circle edge
    blur = cv2.medianBlur(gray, 5)
    
    # ---- STEP 3: Hough Circles ----
    # Args: image, method, dp, minDist, param1, param2,
    #       minRadius, maxRadius
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
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)      # outer
        cv2.circle(out, (x, y), 2, (0, 0, 255), 3)      # center
        cv2.putText(out, str(n), (x-8, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        n += 1
    
    cv2.putText(out, "Count: " + str(len(circles)),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)
    cv2.imwrite("outputs/task7_coins.png", out)
    
    # ---- STEP 5: Small vs Big via 2-means ----
    radius = circles[:, 2].astype(float)
    c1, c2 = radius.min(), radius.max()
    for _ in range(20):
        big = np.abs(radius - c1) > np.abs(radius - c2)
        c1 = radius[~big].mean()
        c2 = radius[big].mean()
    
    return out, (c1, c2, big)


# ======================================================================
# TASK 8: SMART SECURITY SYSTEM (MOG2 + Zone)
# ======================================================================

def task8_security_system(video_path, zone_points):
    """
    Detect objects entering a security zone using MOG2 background
    subtraction.
    
    Args:
        video_path: path to video
        zone_points: list of 4 points defining zone polygon
    """
    zone = np.array(zone_points, np.int32)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(3))
    H = int(cap.get(4))
    out = cv2.VideoWriter("outputs/task8_security.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (W, H))
    
    # ---- Zone mask (binary) ----
    zone_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(zone_mask, [zone], 255)
    
    # ---- Background subtractor ----
    # Args: history=300, varThreshold=40, detectShadows=True
    bg = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=True)
    
    # ---- Morphological kernels ----
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # small
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # big
    
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ---- Background subtraction ----
        # First 60 frames = learning phase
        if n < 60:
            fg = bg.apply(frame)
        else:
            fg = bg.apply(frame, learningRate=0.0001)  # freeze
        
        # ---- Cleanup ----
        # Threshold 200 removes shadows (grey ~127)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k1)   # noise
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k2)  # holes
        fg = cv2.dilate(fg, None, iterations=2)
        
        # ---- Contours ----
        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = frame.copy()
        in_zone = 0
        
        if n >= 60:
            for c in contours:
                if cv2.contourArea(c) < 350:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                # Foot point = bottom-center
                foot = (x + w//2, y + h)
                foot_in = cv2.pointPolygonTest(zone, foot, False) >= 0
                
                # Overlap test
                obj = fg[y:y+h, x:x+w]
                overlap = (np.count_nonzero(obj & zone_mask[y:y+h, x:x+w])
                           / max(1, np.count_nonzero(obj)))
                
                if foot_in or overlap > 0.3:
                    in_zone += 1
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(res, (x, y), (x+w, y+h), color, 1)
        
        # ---- Alarm banner ----
        if in_zone > 0:
            cv2.polylines(res, [zone], True, (0, 0, 255), 3)
            cv2.rectangle(res, (0, 0), (W, 45), (0, 0, 255), -1)
            cv2.putText(res, "ALARM! object in zone", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
        else:
            cv2.polylines(res, [zone], True, (0, 200, 255), 2)
            cv2.putText(res, "clear", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 200, 0), 2)
        
        out.write(res)
        n += 1
    
    cap.release()
    out.release()


# ======================================================================
# MAIN - RUN ALL TASKS
# ======================================================================

if __name__ == "__main__":
    # Example: run task 7 (simplest to test)
    # out, info = task7_coin_detection("images/coins.png")
    # print("Detected:", len(info), "coins")
    pass
  
