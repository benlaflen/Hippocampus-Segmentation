from SAMethods import SAM_Image, recommended_kwargs

im = SAM_Image('Cage5195087-Mouse3RL\\NeuN-s1.tif', **recommended_kwargs)

#Example of how to get GCL
#masks, scores, logits = im.get_best_mask([[6000, 3600], [6000, 3200], [6000, 2500], [6000, 4000]], [1, 1, 0, 0])
#im.display_masks(masks, scores)

#Get central ventricle
points = [[2640, 8000]]
labels = [1]
masks, scores, logits = im.get_masks(points, labels)
im.display_masks(masks, scores, points, labels)