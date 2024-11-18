from SAMethods import SAM_Image, recommended_kwargs
import numpy as np
im = SAM_Image(r'Cage5195087-Mouse3RL\NeuN-s3.tif', **recommended_kwargs)
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
height, width = im.image.shape[:2]
center_x, center_y = width // 2, height // 2
box_width, box_height = 5000, 6000
x_min = center_x - box_width // 2
y_min = center_y - box_height // 2
x_max = center_x + box_width // 2 -800 #center ventrical tends to shift left on image
y_max = center_y + box_height // 2
boxes = [[x_min, y_min, x_max, y_max]]
points = [[center_x, center_y]]
labels = [1]
im.display(labels=labels, boxes=boxes)
