# ============================================================
# 📘 LAB 1 – COMPUTER VISION (AI-4002)
# COMPLETE CODE FILE – TASKS 1 to 9
# ============================================================
# This file contains all Lab 1 code with:
#   - Function names & patterns
#   - Arguments (order, meaning, mandatory?)
#   - Step-by-step process comments
#   - Headings for every topic
# ============================================================


# ============================================================
# 📦 SECTION 0: IMPORTS
# ============================================================
import cv2                       # OpenCV – image processing
import numpy as np               # numerical arrays
import matplotlib.pyplot as plt  # display images
import pandas as pd              # DataFrame for pixel stats


# ============================================================
# 🧮 SECTION 1: TASK 1 – GROCERY MANAGER (OOP)
# ============================================================
# Process:
#   1. Create class with empty dict
#   2. add_item  -> store {name: {quantity, price}}
#   3. remove_item -> check key first, then delete
#   4. view_list -> loop + compute subtotal
#   5. calculate_total -> sum(qty * price)

class GroceryManager:
    # __init__(self) : constructor, no args
    def __init__(self):
        self.items = {}                       # dict: name -> {quantity, price}

    # add_item(self, item, quantity, price)
    #   item      -> str  (mandatory)
    #   quantity  -> int  (mandatory)
    #   price     -> int  (mandatory)
    def add_item(self, item, quantity, price):
        self.items[item] = {'quantity': quantity, 'price': price}
        print(f"Added: {item} (x{quantity}) @ Rs.{price} each")

    # remove_item(self, item)
    #   item -> str (mandatory)
    def remove_item(self, item):
        if item not in self.items:
            print(f"Error: '{item}' is not in the list.")
            return
        del self.items[item]
        print(f"Removed: {item}")

    # view_list(self)
    #   prints a formatted table
    def view_list(self):
        if not self.items:
            print("Grocery list is empty.")
            return
        print(f"{'Item':<15} {'Qty':<6} {'Price':<8} {'Subtotal'}")
        print("-" * 42)
        for item, info in self.items.items():
            sub = info['quantity'] * info['price']
            print(f"{item:<15} {info['quantity']:<6} {info['price']:<8} {sub}")

    # calculate_total(self)
    #   returns sum of (quantity * price)
    def calculate_total(self):
        total = sum(i['quantity'] * i['price'] for i in self.items.values())
        print(f"\nTotal Cost: Rs.{total}")
        return total


# ---- TASK 1 EXECUTION ----
gm = GroceryManager()
gm.add_item("Eggs", 6, 200)
gm.add_item("Milk", 2, 250)
gm.add_item("Bread", 3, 100)
gm.add_item("Butter", 2, 350)

print()
gm.view_list()
gm.calculate_total()

print()
gm.remove_item("Bread")
gm.remove_item("Juice")       # should trigger error

print()
gm.view_list()
gm.calculate_total()


# ============================================================
# 🧮 SECTION 2: TASK 2 – STUDENTS (Nested Dict + Lambda)
# ============================================================
# Process:
#   1. Store records as dict of dicts
#   2. highest_average -> use max() with lambda key = avg
#   3. search_by_major -> list comprehension filter

students = {
    "STD001": {"Name": "Ahmed",  "Major": "AI", "Grades": [88, 92, 79, 95]},
    "STD002": {"Name": "Fatima", "Major": "SE", "Grades": [72, 68, 81, 74]},
    "STD003": {"Name": "Zain",   "Major": "CS", "Grades": [90, 94, 87, 91]},
    "STD004": {"Name": "Maryam", "Major": "CS", "Grades": [65, 70, 60, 72]},
    "STD005": {"Name": "Bilal",  "Major": "AI", "Grades": [85, 88, 92, 80]},
}


# highest_average(records)
#   records -> dict (mandatory)
#   returns -> name of topper
#   Lambda key = sum(Grades)/len(Grades)
def highest_average(records):
    best_id = max(
        records,
        key=lambda sid: sum(records[sid]["Grades"]) / len(records[sid]["Grades"])
    )
    best = records[best_id]
    avg = sum(best["Grades"]) / len(best["Grades"])
    print(f"Highest Average: {best['Name']} ({best_id}) — Avg = {avg:.2f}")
    return best["Name"]


# search_by_major(records, major)
#   records -> dict (mandatory)
#   major   -> str  (mandatory)
#   Uses list comprehension to filter
def search_by_major(records, major):
    found = [(sid, info["Name"]) for sid, info in records.items() if info["Major"] == major]
    if not found:
        print(f"No students found in '{major}'.")
        return
    print(f"Students in {major}:")
    for sid, name in found:
        print(f"  {sid} — {name}")


# ---- TASK 2 EXECUTION ----
highest_average(students)
print()
search_by_major(students, "AI")
print()
search_by_major(students, "CS")


# ============================================================
# 📷 SECTION 3: TASK 3 – READ & DISPLAY IMAGE
# ============================================================
# Process:
#   1. imread  -> load as BGR
#   2. Check if None
#   3. cvtColor -> BGR to RGB
#   4. imshow -> display

# Function: cv2.imread(path)
#   path -> str (mandatory)
#   returns -> NumPy array (BGR) OR None
image = cv2.imread('Sukuna.jpeg')

if image is None:
    print("Error: Image not found. Check the filename/path.")
else:
    # Function: cv2.cvtColor(src, code)
    #   src  -> image (BGR)    [mandatory]
    #   code -> cv2.COLOR_BGR2RGB  [mandatory]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Figure with custom size
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title('King of Curses - Ryomen Sukuna', fontsize=16, color='darkred', pad=15)
    plt.axis('off')       # hide axes
    plt.show()


# ============================================================
# 🎨 SECTION 4: TASK 4 – DRAWING (Circle + Rectangle)
# ============================================================
# Process:
#   1. Create black canvas via np.zeros((h, w, 3))
#   2. cv2.circle for concentric rings
#   3. cv2.rectangle for bounding box
#   4. Convert BGR -> RGB, display

# Function: np.zeros(shape, dtype)
#   shape -> (height, width, channels) [mandatory]
#   dtype -> np.uint8 [mandatory for images]
canvas = np.zeros((800, 800, 3), dtype=np.uint8)

center = (400, 400)     # (x, y) pivot
print(f"Center of canvas: {center}")

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
radii = [200, 160, 120, 80, 40]

# Function: cv2.circle(img, center, radius, color, thickness)
#   img       -> canvas [mandatory]
#   center    -> (x, y) tuple [mandatory]
#   radius    -> int [mandatory]
#   color     -> BGR tuple [mandatory]
#   thickness -> int; -1 = filled [mandatory]
for r, color in zip(radii, colors):
    cv2.circle(canvas, center, r, color, 3)

# Function: cv2.rectangle(img, pt1, pt2, color, thickness)
#   pt1 -> top-left corner (x, y) [mandatory]
#   pt2 -> bottom-right corner (x, y) [mandatory]
top_left = (center[0] - radii[0] - 3, center[1] - radii[0] - 3)
bottom_right = (center[0] + radii[0] + 3, center[1] + radii[0] + 3)
cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 2)

canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(7, 7))
plt.imshow(canvas_rgb)
plt.title('Target Board with Bounding Box', fontsize=14)
plt.axis('off')
plt.show()


# ============================================================
# 🌀 SECTION 5: TASK 5 – BLUR + ROI + PANDAS STATS
# ============================================================
# Process:
#   1. Read image, convert BGR->RGB
#   2. cv2.GaussianBlur for smoothing
#   3. Compute ROI coords (center 300x300)
#   4. Slice image[y1:y2, x1:x2] for original + blurred
#   5. Display side-by-side with subplots
#   6. Flatten and analyze with Pandas describe()

image = cv2.imread('mountain.jpg')

if image is None:
    print("Error: Image not found.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Function: cv2.GaussianBlur(src, ksize, sigmaX)
    #   src    -> input image [mandatory]
    #   ksize  -> (w, h) tuple; MUST BE ODD [mandatory]
    #   sigmaX -> std dev; 0 = auto [mandatory]
    blurred = cv2.GaussianBlur(image_rgb, (25, 25), 0)

    # Get height, width (note: shape[:2] = (h, w))
    h, w = image_rgb.shape[:2]

    # Define ROI coordinates (center 300x300 box)
    y1 = h // 2 - 150
    y2 = h // 2 + 150
    x1 = w // 2 - 150
    x2 = w // 2 + 150

    # Cropping via NumPy slicing: image[y_start:y_end, x_start:x_end]
    # Rule: Y (rows) first, X (columns) second
    roi_original = image_rgb[y1:y2, x1:x2]
    roi_blurred  = blurred[y1:y2, x1:x2]

    # Subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(roi_original)
    axes[0].set_title('Original ROI (300x300)')
    axes[0].axis('off')

    axes[1].imshow(roi_blurred)
    axes[1].set_title('Blurred ROI (300x300)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Flatten: (H, W, 3) -> (H*W, 3)
    # -1 = auto-calc; 3 = channels
    flat = image.reshape(-1, 3)

    # DataFrame: columns = B, G, R (because image is BGR)
    df = pd.DataFrame(flat, columns=['B', 'G', 'R'])
    print("Pixel Statistics per Channel:")
    print(df.describe())      # count, mean, std, min, 25%, 50%, 75%, max


# ============================================================
# 🎨 SECTION 6: TASK 6 – ALPHA BLENDING + CAPTION
# ============================================================
# Process:
#   1. Read image, convert BGR->RGB
#   2. Copy image as overlay
#   3. Draw filled rectangle (bottom 20%)
#   4. cv2.addWeighted -> blend (semi-transparent effect)
#   5. cv2.putText -> caption
#   6. Display

image = cv2.imread('mountain.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# .copy() -> independent copy (original safe)
overlay = image_rgb.copy()
h, w = image_rgb.shape[:2]

# Blue rectangle covering bottom 20%
# cv2.rectangle(img, pt1, pt2, color, thickness=-1)
cv2.rectangle(overlay, (0, int(h * 0.8)), (w, h), (30, 60, 150), -1)

# Function: cv2.addWeighted(src1, alpha, src2, beta, gamma)
#   src1  -> first image [mandatory]
#   alpha -> weight of src1 [mandatory]
#   src2  -> second image [mandatory]
#   beta  -> weight of src2 [mandatory]
#   gamma -> brightness offset [mandatory]
#   Formula: result = alpha*src1 + beta*src2 + gamma
#   For natural blend: alpha + beta = 1
blended = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

# Function: cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
#   img       -> image [mandatory]
#   text      -> string [mandatory]
#   org       -> BOTTOM-LEFT corner (x, y) [mandatory]
#   fontFace  -> e.g. FONT_HERSHEY_SIMPLEX [mandatory]
#   fontScale -> size [mandatory]
#   color     -> BGR [mandatory]
#   thickness -> line width [mandatory]
cv2.putText(blended, "Mountain Lake Reflection", (20, int(h * 0.93)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

plt.figure(figsize=(10, 6))
plt.imshow(blended)
plt.title('Alpha Blended Caption')
plt.axis('off')
plt.show()


# ============================================================
# 📊 SECTION 7: TASK 9 – STATISTICAL IMAGE PROFILING (Pandas)
# ============================================================
# Process:
#   1. Read image, convert BGR->RGB
#   2. Flatten to (total_pixels, 3)
#   3. Build DataFrame with columns Red, Green, Blue
#   4. describe() for stats

image = cv2.imread('mountain.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape(-1, 3):
#   -1 -> auto-calc total pixels (H*W)
#    3 -> channels
flat = image_rgb.reshape(-1, 3)

# Columns: ['Red', 'Green', 'Blue'] (because image is RGB)
# NOTE: If using BGR image, columns would be ['B', 'G', 'R']
df = pd.DataFrame(flat, columns=['Red', 'Green', 'Blue'])

print("Pixel Statistics per Channel:")
# describe() returns: count, mean, std, min, 25%, 50%, 75%, max
print(df.describe())


# ============================================================
# 📌 SECTION 8: FUNCTION REFERENCE (Quick Lookup)
# ============================================================
# cv2.imread(path)                              -> load image as BGR (or None)
# cv2.cvtColor(src, code)                       -> convert color space
# plt.imshow(img, cmap=None)                    -> display image
# plt.axis('off')                               -> hide axes
# np.zeros((h, w, 3), dtype=np.uint8)           -> black canvas
# cv2.circle(img, center, radius, color, thick) -> draw circle
# cv2.rectangle(img, pt1, pt2, color, thick)    -> draw rectangle
# cv2.GaussianBlur(src, ksize, sigmaX)          -> blur (ksize odd)
# image[y1:y2, x1:x2]                           -> crop (Y first, X second)
# plt.subplots(rows, cols, figsize)             -> multiple subplots
# cv2.addWeighted(src1, a, src2, b, g)          -> linear blend
# cv2.putText(img, text, org, font, scale, c, t)-> draw text (org=bottom-left)
# image.reshape(-1, 3)                          -> flatten (H*W, 3)
# pd.DataFrame(data, columns=[...])             -> create DataFrame
# df.describe()                                 -> 8-stat summary


# ============================================================
# 🎯 SECTION 9: MCQ GOLD POINTS (Comments for recall)
# ============================================================
# - OpenCV reads  : BGR
# - Matplotlib    : RGB
# - Grayscale     : 0–255
# - Crop syntax   : image[y1:y2, x1:x2]
# - Gaussian kernel: ODD only
# - cv2.threshold : returns (ret, image)
# - cv2.addWeighted: linear blend (α + β = 1)
# - putText origin: bottom-left
# - Rotation matrix: 2×3
# - equalizeHist  : grayscale uint8 only
# - DIP hardest   : Segmentation
# - Image origin  : top-left
# - shape[:2]     : (height, width)
# - Pandas cols (BGR) : ['B','G','R']
# - Pandas cols (RGB) : ['Red','Green','Blue']


# ============================================================
# ✅ END OF LAB 1 CODE FILE
# ============================================================
