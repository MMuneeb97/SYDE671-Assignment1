import numpy as np
import matplotlib.pyplot as plt

'''
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

filepath = filedialog.askopenfilename(
    title = "Select image to open",
    filetypes = [("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("All files","*.*")]
)

if not filepath:
    print("No image selected.")
    exit()
'''

# --- CUSTOM SLICE-BASED SHIFT FUNCTION (Replaces np.roll) ---
def shift_array_slice(arr, shift_y, shift_x):
    """
    Shifts a 2D numpy array using slice assignment instead of np.roll.
    Pads out-of-bounds regions with zeros.
    """
    h, w = arr.shape
    shifted = np.zeros_like(arr)
    
    # Calculate source (arr) and destination (shifted) slice coordinates
    src_y_start = max(0, -shift_y)
    src_y_end   = min(h, h - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end   = min(w, w - shift_x)
    
    dst_y_start = max(0, shift_y)
    dst_y_end   = min(h, h + shift_y)
    dst_x_start = max(0, shift_x)
    dst_x_end   = min(w, w + shift_x)
    
    if (src_y_end > src_y_start) and (src_x_end > src_x_start):
        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            arr[src_y_start:src_y_end, src_x_start:src_x_end]
            
    return shifted

# 1. Read Image
im = plt.imread(r'C:\Users\Muhammad Muneeb\Desktop\Documents\Masters\SYDE 671\data\31421v.jpg')

# Normalize image to float [0, 1] if loaded as integers
if im.dtype == np.uint8:
    im = im.astype(float) / 255.0

# 2. Split into B, G, R channels (top to bottom order in glass plates: B, G, R)
h, w = im.shape
channel_h = h // 3

blue = im[0:channel_h, :]
green = im[channel_h:2*channel_h, :]
red = im[2*channel_h:3*channel_h, :]

# Function to downsample image for Gaussian pyramid
def scale_img(image):
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel = kernel / np.sum(kernel)

    horizontal = np.zeros_like(image, dtype=float)
    horizontal[:, 2:-2] = ( 
        image[:, :-4] * kernel[0] + 
        image[:, 1:-3] * kernel[1] + 
        image[:, 2:-2] * kernel[2] + 
        image[:, 3:-1] * kernel[3] + 
        image[:, 4:] * kernel[4] 
    )

    blurred = np.zeros_like(image, dtype=float)
    blurred[2:-2, :] = ( 
        horizontal[:-4, :] * kernel[0] + 
        horizontal[1:-3, :] * kernel[1] + 
        horizontal[2:-2, :] * kernel[2] + 
        horizontal[3:-1, :] * kernel[3] + 
        horizontal[4:, :] * kernel[4] 
    )

    return blurred[::2, ::2]

# Build Pyramids
N = 5
blue_pyramid = [blue]
green_pyramid = [green]
red_pyramid = [red]

for k in range(1, N):
    blue_pyramid.append(scale_img(blue_pyramid[k-1]))
    green_pyramid.append(scale_img(green_pyramid[k-1]))
    red_pyramid.append(scale_img(red_pyramid[k-1]))

# NCC Function with Zero-Mean Normalization
def ncc(reference_region, moving_region):
    ref_norm = reference_region - np.mean(reference_region)
    mov_norm = moving_region - np.mean(moving_region)

    nume = np.sum(ref_norm * mov_norm)
    denom = np.sqrt(np.sum(ref_norm**2) * np.sum(mov_norm**2))

    if denom == 0:
        return -1
    return nume / denom

# Crop function to remove borders during alignment score calculations
def crop_inner_region(img, crop_percent=0.10):
    h, w = img.shape
    ch, cw = int(h * crop_percent), int(w * crop_percent)
    return img[ch:h-ch, cw:w-cw]

# Alignment around a center displacement
def find_displacement_around(reference, moving, center_dx, center_dy, margin):
    best_score = -1
    best_dx, best_dy = center_dx, center_dy

    # Crop outer edges to avoid matching border noise
    crop_h, crop_w = int(reference.shape[0] * 0.12), int(reference.shape[1] * 0.12)

    for dy in range(center_dy - margin, center_dy + margin + 1):
        for dx in range(center_dx - margin, center_dx + margin + 1):
            
            # REPLACED np.roll with custom slice function
            shifted_moving = shift_array_slice(moving, dy, dx)

            # Evaluate score ONLY on the inner region of the cropped content
            ref_crop = reference[crop_h:-crop_h, crop_w:-crop_w]
            mov_crop = shifted_moving[crop_h:-crop_h, crop_w:-crop_w]

            score = ncc(ref_crop, mov_crop)

            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy

# Recursive Pyramid Alignment
def align_pyramid(reference_pyramid, moving_pyramid, level, margin=15):
    if level == len(reference_pyramid) - 1:
        return find_displacement_around(
            reference_pyramid[level], 
            moving_pyramid[level], 
            0, 0, margin
        )

    # Recurse to coarser level
    dx, dy = align_pyramid(reference_pyramid, moving_pyramid, level + 1, margin)

    # Scale shift prediction up by 2x
    dx *= 2
    dy *= 2

    # Refine shift at current resolution level
    return find_displacement_around(
        reference_pyramid[level], 
        moving_pyramid[level], 
        dx, dy, margin=3
    )

# Compute optimal displacements using Blue channel as reference
best_green_dx, best_green_dy = align_pyramid(blue_pyramid, green_pyramid, 0)
best_red_dx, best_red_dy = align_pyramid(blue_pyramid, red_pyramid, 0)

print(f"Green displacement (dx, dy): {best_green_dx}, {best_green_dy}")
print(f"Red displacement (dx, dy): {best_red_dx}, {best_red_dy}")

# REPLACED np.roll with custom slice function for final channel application
green_aligned = shift_array_slice(green, best_green_dy, best_green_dx)
red_aligned = shift_array_slice(red, best_red_dy, best_red_dx)

# Combine channels into final RGB image
final_img = np.dstack((red_aligned, green_aligned, blue))

# Crop border artifact margins from display result
crop_margin = 40
final_cropped = final_img[crop_margin:-crop_margin, crop_margin:-crop_margin]

plt.figure(figsize=(10, 8))
plt.imshow(np.clip(final_cropped, 0, 1))
plt.axis('off')
plt.show()
