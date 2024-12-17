from SAMethods import SAM_Image, recommended_kwargs
import numpy as np
import cv2
path = 'Cage4841876-Mouse3RL\\+'
#D:\Katie\Hippocampus-Segmentation\Cage4841876-Mouse3RL\\s1-NeuN.tif
#D:\Katie\Hippocampus-Segmentation\Cage5195087-Mouse3RL\\NeuN-s1.tif
im = SAM_Image(path, **recommended_kwargs)
import matplotlib.pyplot as plt

def get_mask_center(mask):
    nz = np.nonzero(mask)
    return np.average(nz[1]), np.average(nz[0])

#(x1,y1,x2,y2)
def get_mask_bounds(mask):
    nz = np.nonzero(mask)
    return (np.min(nz[1]), np.min(nz[0]), np.max(nz[1]), np.max(nz[0]))

#Example of how to get GCL
#masks, scores, logits = im.get_best_mask([[6000, 3600], [6000, 3200], [6000, 2500], [6000, 4000]], [1, 1, 0, 0])
#im.display_masks(masks, scores)

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.show() 
_, binary_image = cv2.threshold(image, 25, 225, cv2.THRESH_BINARY_INV)

height, width = binary_image.shape
center_fraction = 3
start_x = width // center_fraction
end_x = width - width // center_fraction
start_y = height // center_fraction
end_y = height - height // center_fraction

center_binary = binary_image[start_y:end_y, start_x:end_x]
distance_transform = cv2.distanceTransform(center_binary, cv2.DIST_L2, 5)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(distance_transform)

center_ventricle = (max_loc[0] + start_x, max_loc[1] + start_y)

#TEST INV IMAGE 
# plt.figure(figsize=(10, 6))
# plt.imshow(binary_image, cmap='gray')
# plt.scatter(center_ventricle[0], center_ventricle[1], color='red', s=100, label='Center Ventricle')
# plt.title('Center Ventricle Identified')
# plt.axis('off')
# plt.legend()
# plt.show()
# center_x, center_y = center_ventricle

distance_transform_copy = distance_transform.copy()

#Circular mask around the first point
mask_radius = 550
for y in range(distance_transform_copy.shape[0]):
    for x in range(distance_transform_copy.shape[1]):
        if (x - max_loc[0])**2 + (y - max_loc[1])**2 <= mask_radius**2:
            distance_transform_copy[y, x] = 0

min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(distance_transform_copy)
second_center_ventricle = (max_loc2[0] + start_x, max_loc2[1] + start_y)

found_white_point = False
nearest_white_point = None

# Define a range for search (expand from center ventricle)
search_radius = 500  # You can adjust this based on your need

# Loop in expanding square pattern starting from the center ventricle
# for radius in range(1, search_radius):
#     # Check in all four directions (up, down, left, right) in a square pattern
#     for dx in range(-radius, radius+1):
#         for dy in range(-radius, radius+1):
#             x = center_ventricle[0] + dx
#             y = center_ventricle[1] + dy
            
#             # Check if the point is within the bounds of the image
#             if 0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]:
#                 if binary_image[y, x] == 255:  # Found white point
#                     nearest_white_point = (x, y)
#                     found_white_point = True
#                     break
#         if found_white_point:
#             break
#     if found_white_point:
#         break
# found_white_point = False
# nearest_white_point = None

# # Loop through the entire image to find a white point
# for y in range(binary_image.shape[0]):
#     for x in range(binary_image.shape[1]):
#         if binary_image[y, x] == 225:  # White point in the binary image
#             nearest_white_point = (x, y)
#             found_white_point = True
#             break
#     if found_white_point:
#         break

# ventricle_mask = np.zeros_like(binary_image)  # Assuming this is your ventricle mask
# ventricle_mask[start_y:end_y, start_x:end_x] = center_binary  # Fill ventricle region into mask

# # Invert the ventricle mask to get the outside region
# outside_mask = cv2.bitwise_not(ventricle_mask)

# # Find white points (255) outside the ventricle region
# outside_white_points = np.nonzero(outside_mask == 255)

# # Pick the first white point outside the ventricle
# if len(outside_white_points[0]) > 0:
#     white_x, white_y = outside_white_points[0][0], outside_white_points[1][0]
#     nearest_white_point = (white_x, white_y)
# else:
#     nearest_white_point = (None, None)


points = [[center_ventricle[0], center_ventricle[1]], [second_center_ventricle[0], second_center_ventricle[1]],[center_ventricle[0]+1000, center_ventricle[1]+1000]]
labels = [1, 1,0]

masks, scores, logits = im.get_best_mask(points, labels)

im.display(masks=masks, labels=labels, points=points)