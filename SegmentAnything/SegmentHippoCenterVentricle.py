from SAMethods import SAM_Image, recommended_kwargs
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_mask_center(mask):
    nz = np.nonzero(mask)
    return np.average(nz[1]), np.average(nz[0])

#(x1,y1,x2,y2)
def get_mask_bounds(mask):
    nz = np.nonzero(mask)
    return (np.min(nz[1]), np.min(nz[0]), np.max(nz[1]), np.max(nz[0]))

def generate_negative_points_outside_center_x(center_y, start_x, end_x, width, num_points=10):
    # Generate negative points systematically outside the center region along a fixed y-coordinate
    negative_points = []
    step_x = max(1, width // num_points)  # Step size for x-coordinates to distribute points

    for i in range(num_points):
        if i % 2 == 0:  # Alternate points on the left side of the center region
            x = (start_x - (i // 2 + 1) * step_x) % width
        else:  # Alternate points on the right side of the center region
            x = (end_x + (i // 2 + 1) * step_x) % width

        # Add the point with a fixed y-coordinate (center_y)
        negative_points.append((x, center_y))

    return negative_points

def get_central_ventricle(im, display=False):
    image = im.image
    _, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_binary = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)[:,:,0]

    height, width = cleaned_binary.shape
    center_fraction = 3
    start_x = width // center_fraction
    end_x = width - width // center_fraction
    center_binary = cleaned_binary[:, start_x:end_x]

    distance_transform = cv2.distanceTransform(center_binary, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(distance_transform)
    center_ventricle = (max_loc[0] + start_x, max_loc[1])

    negative_points = generate_negative_points_outside_center_x(
        center_y=center_ventricle[1], 
        start_x=start_x, 
        end_x=end_x, 
        width=width,
        num_points=10
    )

    positive_points = [center_ventricle]
    points = positive_points + negative_points
    labels = [1] * len(positive_points) + [0] * len(negative_points)

    _, _, logits = im.get_best_mask(points, labels)
    masks, scores, logits = im.get_best_mask(points, labels, logits)

    selected_mask = None
    for mask in masks:
        if mask[center_ventricle[1], center_ventricle[0]]:
            selected_mask = mask
            break

    if selected_mask is None:
        selected_mask = masks[0]

    if display:
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.imshow(selected_mask, alpha=0.4, cmap='Greens')

        for (x, y), label in zip(points, labels):
            color = 'green' if label == 1 else 'red'
            plt.scatter(x, y, c=color, s=50, edgecolors='black')

        plt.title("Center Ventricle Mask")
        plt.axis('off')
        plt.show()

    return selected_mask, scores, logits

#Same as in Hilus, we should keep test code in either SegmentHippo or other dedicated test files, because SegmentHippo needs to be able to import these files (without running test code)
r'''
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

image = Image.open(r'Cage5195087-Mouse3RL\NeuN-s1.tif')
image_array = np.array(image)

if image_array.dtype == np.uint16 or str(image_array.dtype).startswith('>u2'):
    image_array = (image_array / 256).astype(np.uint8)

if image_array.ndim == 2:
    image_array = np.stack([image_array] * 3, axis=-1)  # shape becomes (H, W, 3)

im = SAM_Image(image_array, **recommended_kwargs)
masks, scores, logits = get_central_ventricle(im)
'''