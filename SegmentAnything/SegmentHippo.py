from SAMethods import SAM_Image, recommended_kwargs
import numpy as np
<<<<<<< HEAD
from scipy.signal import argrelextrema
im = SAM_Image('Cage5195087-Mouse3RL\\NeuN-s2.tif', **recommended_kwargs)
=======
im = SAM_Image('Cage5195087-Mouse3RL\\NeuN-s1.tif', **recommended_kwargs)
>>>>>>> parent of 0bec8b8 (Merge branch 'main' of https://github.com/benlaflen/Hippocampus-Segmentation)
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

#Get central ventricle
points = [[7500, 3500]]
labels = [1]
masks, scores, logits = im.get_best_mask(points, labels)
vent_x,vent_y = get_mask_center(masks[0])
vent_box = get_mask_bounds(masks[0])

<<<<<<< HEAD
points = [[int(vent_box[0]*0.5), int((vent_box[1]+vent_box[3])/2)]]
get_left_GCL(im, vent_box, False)
#get_verticle_maxima(4132, im, True)
=======
#Scan to the left of the ventricle
target_x = int(vent_box[0]*0.75)
print(target_x)
target_column = im.image[:, target_x]
brightness = np.mean(target_column, 1)
plt.plot(brightness)
plt.show()
>>>>>>> parent of 0bec8b8 (Merge branch 'main' of https://github.com/benlaflen/Hippocampus-Segmentation)
