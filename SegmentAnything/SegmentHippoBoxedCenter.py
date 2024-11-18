from SAMethods import SAM_Image, recommended_kwargs
import numpy as np
import cv2
im = SAM_Image(r'Cage5195087-Mouse3RL\\NeuN-s1.tif', **recommended_kwargs)
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

#Collects the shape and height of the generated image to find the center point
# height, width = im.image.shape[:2]
# center_x, center_y = width // 2, height // 2
# box_width, box_height = 5000, 6000
# x_min = center_x - box_width // 2
# y_min = center_y - box_height // 2
# x_max = center_x + box_width // 2 -800 #center ventrical tends to shift left on image
# y_max = center_y + box_height // 2
# boxes = [[x_min, y_min, x_max, y_max]]
# points = [[center_x, center_y]]
# labels = [1]
# im.display(labels=labels, boxes=boxes)
image = cv2.imread('Cage5195087-Mouse3RL\\NeuN-s2.tif', cv2.IMREAD_GRAYSCALE)
height, width = image.shape
start_x = width // 3
end_x = 2 * (width // 3)
start_y = height // 3
end_y = 2 * (height // 3)
center_image = image[start_y:end_y, start_x:end_x]
_, binary_image = cv2.threshold(center_image, 1, 255, cv2.THRESH_BINARY_INV)
transformed_dist = cv2.distanceTransform(center_image, cv2.DIST_L2, 5)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(transformed_dist)

widest_black_point = (max_loc[0] + start_x, max_loc[1] + start_y)
print(f"Widest black point located at: {widest_black_point}")
center_x, center_y = widest_black_point[0], widest_black_point[1]
#print(f"Maximum distance: {center_x}")
marked_image = image.copy()  # Create a copy to mark the point
cv2.circle(marked_image, widest_black_point, 5, (255, 0, 0), 2)  # Blue circle (255, 0, 0)
    
# points = [[center_x, center_y]]
# labels = [1]
# im.display(labels=labels, points=points)
cv2.imshow("Marked Image", marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()