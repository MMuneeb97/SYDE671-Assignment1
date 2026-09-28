import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD IMAGE
# ============================================================

im = plt.imread(
    r'C:\Users\Muhammad Muneeb\Desktop\Documents\Masters\SYDE 671\data\00056v.jpg'
)

# Display original stacked image
plt.imshow(im)
# plt.show()


# ============================================================
# 2. SPLIT IMAGE INTO BLUE, GREEN, AND RED CHANNELS
# ============================================================

h, w = im.shape

channel_h = h // 3

blue = im[0:channel_h, :]
green = im[channel_h:2*channel_h, :]
red = im[2*channel_h:3*channel_h, :]


# ============================================================
# 3. NCC FUNCTION
# ============================================================

def ncc(reference_region, moving_region):

    # Convert to floating point
    reference_region = reference_region.astype(float)
    moving_region = moving_region.astype(float)

    # Subtract the mean from each image
    reference_region = reference_region - np.mean(reference_region)
    moving_region = moving_region - np.mean(moving_region)

    # Numerator
    nume = np.sum(reference_region * moving_region)

    # Denominator
    denom = np.sqrt(
        np.sum(reference_region**2) *
        np.sum(moving_region**2)
    )

    # Avoid division by zero
    if denom == 0:
        return -1

    # NCC score
    return nume / denom


# ============================================================
# 4. FIND DISPLACEMENT
# ============================================================

def find_displacement(reference, moving, margin):

    h, w = reference.shape

    # NCC is higher when the images match better
    best_score = -1

    best_dx = 0
    best_dy = 0

    # Search through possible y displacements
    for dy in range(-margin, margin + 1):

        # Search through possible x displacements
        for dx in range(-margin, margin + 1):

            # Determine size of overlapping region
            overlap_h = h - abs(dy)
            overlap_w = w - abs(dx)

            # Skip invalid overlap
            if overlap_h <= 0 or overlap_w <= 0:
                continue

            # Starting position in moving image
            moving_y_start = max(0, dy)
            moving_x_start = max(0, dx)

            # Starting position in reference image
            reference_y_start = max(0, -dy)
            reference_x_start = max(0, -dx)

            # Extract moving region
            moving_region = moving[
                moving_y_start:moving_y_start + overlap_h,
                moving_x_start:moving_x_start + overlap_w
            ]

            # Extract reference region
            reference_region = reference[
                reference_y_start:reference_y_start + overlap_h,
                reference_x_start:reference_x_start + overlap_w
            ]

            # Make sure both regions have the same size
            if moving_region.shape != reference_region.shape:
                continue

            # Calculate NCC
            score = ncc(
                reference_region,
                moving_region
            )

            # Keep the displacement with the highest NCC
            if score > best_score:

                best_score = score

                best_dx = dx
                best_dy = dy

    return best_dx, best_dy


# ============================================================
# 5. SET SEARCH WINDOW
# ============================================================

margin = 15


# ============================================================
# 6. ALIGN GREEN TO BLUE
# ============================================================

green_dx, green_dy = find_displacement(
    blue,
    green,
    margin
)


# ============================================================
# 7. ALIGN RED TO BLUE
# ============================================================

red_dx, red_dy = find_displacement(
    blue,
    red,
    margin
)


# ============================================================
# 8. PRINT DISPLACEMENTS
# ============================================================

print("Green displacement:", green_dx, green_dy)
print("Red displacement:", red_dx, red_dy)


# ============================================================
# 9. FIND COMMON OVERLAPPING REGION
# ============================================================

top = max(
    0,
    green_dy,
    red_dy
)

bottom = min(
    channel_h,
    channel_h + green_dy,
    channel_h + red_dy
)

left = max(
    0,
    green_dx,
    red_dx
)

right = min(
    w,
    w + green_dx,
    w + red_dx
)


# ============================================================
# 10. CALCULATE FINAL IMAGE SIZE
# ============================================================

height = bottom - top
width = right - left


# ============================================================
# 11. CROP BLUE CHANNEL
# ============================================================

blue_aligned = blue[
    top:bottom,
    left:right
]


# ============================================================
# 12. CROP ALIGNED GREEN CHANNEL
# ============================================================

green_aligned = green[
    top - green_dy:
    top - green_dy + height,

    left - green_dx:
    left - green_dx + width
]


# ============================================================
# 13. CROP ALIGNED RED CHANNEL
# ============================================================

red_aligned = red[
    top - red_dy:
    top - red_dy + height,

    left - red_dx:
    left - red_dx + width
]


# ============================================================
# 14. COMBINE CHANNELS INTO COLOR IMAGE
# ============================================================

final_img = np.dstack((
    red_aligned,
    green_aligned,
    blue_aligned
))


# ============================================================
# 15. DISPLAY FINAL IMAGE
# ============================================================

plt.figure()

plt.imshow(final_img)

plt.axis('off')

plt.show()