import cv2
import numpy as np
from SAMethods import SAM_Image, recommended_kwargs

def get_mask_center(mask):
    nz = np.nonzero(mask)
    return np.average(nz[1]), np.average(nz[0])

#(x1,y1,x2,y2)
def get_mask_bounds(mask):
    nz = np.nonzero(mask)
    return (np.min(nz[1]), np.min(nz[0]), np.max(nz[1]), np.max(nz[0]))

path = 'Cage4841876-Mouse3RL\\s2-NeuN.tif'
#D:\Katie\Hippocampus-Segmentation\Cage4841876-Mouse3RL\\s1-NeuN.tif
#D:\Katie\Hippocampus-Segmentation\Cage5195087-Mouse3RL\\NeuN-s1.tif
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

#Example of how to get GCL
#masks, scores, logits = im.get_best_mask([[6000, 3600], [6000, 3200], [6000, 2500], [6000, 4000]], [1, 1, 0, 0])
#im.display_masks(masks, scores)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_binary = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

height, width = cleaned_binary.shape

center_fraction = 3
start_x = width // center_fraction
end_x = width - width // center_fraction

center_binary = cleaned_binary[:, start_x:end_x]
distance_transform = cv2.distanceTransform(center_binary, cv2.DIST_L2, 5)

_, _, _, max_loc = cv2.minMaxLoc(distance_transform)
center_ventricle = (max_loc[0] + start_x, max_loc[1])

def find_second_positive_point(distance_transform, first_point, min_distance=100):
    flat_indices = np.argsort(-distance_transform.flatten())  
    for idx in flat_indices:
        y, x = np.unravel_index(idx, distance_transform.shape)
        if np.linalg.norm(np.array([x, y]) - np.array(first_point)) > min_distance:
            return (x, y)  
    return None

second_local = find_second_positive_point(distance_transform, max_loc)
if second_local:
    second_point = (second_local[0] + start_x, second_local[1]) 
else:
    raise ValueError("Unable to find a second positive point.")

def generate_negative_points_outside_center_y(center_y, start_x, end_x, num_points=10):
    negative_points = []
    for _ in range(num_points):
        while True:
            x = np.random.randint(0, width)
            if x < start_x or x > end_x: 
                negative_points.append((x, center_y))
                break
    return negative_points

negative_points = generate_negative_points_outside_center_y(center_ventricle[1], start_x, end_x, num_points=10)


positive_points = [center_ventricle, second_point]
points = positive_points + negative_points
labels = [1] * len(positive_points) + [0] * len(negative_points)

im = SAM_Image(path, **recommended_kwargs)
masks, scores, logits = im.get_best_mask(points, labels)

im.display(masks=masks, labels=labels, points=points)
