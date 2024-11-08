from SAMethods import SAM_Image, recommended_kwargs

im = SAM_Image('Cage5195087-Mouse3RL\\NeuN-s1.tif', **recommended_kwargs)

#Example of how to get GCL
#masks, scores, logits = im.get_best_mask([[6000, 3600], [6000, 3200], [6000, 2500], [6000, 4000]], [1, 1, 0, 0])
#im.display_masks(masks, scores)

#Get central ventricle
x, y = 2640, 8000 #the center point
width = 5000
height = 1000
top_left = (x - width//2, y - height//2)
top_right = (x + width//2, y + height//2)
bottom_left = (x - width//2, y - height//2)
bottom_right = (x + width//2, y + height//2)
box = [top_left, top_right, bottom_left, bottom_right]
#points = [[top_left], [top_right], [bottom_left],[bottom_right]]
labels = [1,1,1,1]
masks, scores, logits = im.get_masks(boxes = box)
im.display_masks(masks, scores, labels)