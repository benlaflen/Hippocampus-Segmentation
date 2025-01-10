from SAMethods import SAM_Image, recommended_kwargs
import numpy as np
import cv2

def get_mask_center(mask):
    nz = np.nonzero(mask)
    return np.average(nz[1]), np.average(nz[0])

#(x1,y1,x2,y2)
def get_mask_bounds(mask):
    nz = np.nonzero(mask)
    return (np.min(nz[1]), np.min(nz[0]), np.max(nz[1]), np.max(nz[0]))
# Load the image in grayscale
path = 'Cage4841876-Mouse3RL\\s3-NeuN.tif'
#D:\Katie\Hippocampus-Segmentation\Cage4841876-Mouse3RL\\s1-NeuN.tif
#D:\Katie\Hippocampus-Segmentation\Cage5195087-Mouse3RL\\NeuN-s1.tif
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# Threshold the image to create a binary image
_, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

# Create a small, oval-shaped kernel to process the image & clean up noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Morphological opening to remove the noise and smooth image boundaries
cleaned_binary = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Get the height and width of the cleaned binary image
height, width = cleaned_binary.shape

# Define the center region of the image to focus on for ventricle detection
center_fraction = 3
start_x = width // center_fraction  # Start of the center region
end_x = width - width // center_fraction  # End of the center region

# Crop the binary image & only include the center region
center_binary = cleaned_binary[:, start_x:end_x]

# Perform a distance transform to calculate the Euclidean distance to the nearest zero pixel for each pixel
distance_transform = cv2.distanceTransform(center_binary, cv2.DIST_L2, 5)

# Find the pixel with the maximum distance in the distance transform
_, _, _, max_loc = cv2.minMaxLoc(distance_transform)

# Adjust the coordinates to account for cropping and define the center ventricle point
center_ventricle = (max_loc[0] + start_x, max_loc[1])

def generate_negative_points_outside_center_x(center_y, start_x, end_x, num_points=10):
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

# Generate a set of negative points for training outside the center region
negative_points = generate_negative_points_outside_center_x(
    center_y=center_ventricle[1], 
    start_x=start_x, 
    end_x=end_x, 
    num_points=10
)
# Combine positive (center ventricle-related) and negative points with labels
positive_points = [center_ventricle]
points = positive_points + negative_points
labels = [1] * len(positive_points) + [0] * len(negative_points)

# Create a SAM_Image object using the image path and recommended settings
im = SAM_Image(path, **recommended_kwargs)

# Get the best masks using the points and labels for guidance
masks, scores, logits = im.get_best_mask(points, labels)

# Display the resulting masks, labels, and points on the image
im.display(masks=masks, labels=labels, points=points)
