from SAMethods import SAM_Image, recommended_kwargs
import numpy as np
from scipy.signal import argrelextrema
im = SAM_Image('Cage5195087-Mouse3RL\\NeuN-s2.tif', **recommended_kwargs)
import matplotlib.pyplot as plt

def get_mask_center(mask):
    nz = np.nonzero(mask)
    return np.average(nz[1]), np.average(nz[0])

#(x1,y1,x2,y2)
def get_mask_bounds(mask):
    nz = np.nonzero(mask)
    return (np.min(nz[1]), np.min(nz[0]), np.max(nz[1]), np.max(nz[0]))

def get_verticle_maxima(target_x, image, disp=False):
    target_column = image.image[:, target_x]
    brightness = np.mean(target_column, 1)

    window_size = 100
    weights = np.ones(window_size) / window_size
    sma = np.concatenate(([0 for x in range(int(window_size/2))],np.convolve(brightness, weights, mode='valid'),[0 for x in range(int(window_size/2))]))

    maximums = argrelextrema(sma, np.greater)[0]

    cutoff = 2*np.average(sma)

    finalmaxes = []
    current = -1
    for x in range(len(sma)):
        if sma[x] < cutoff:
            if current != -1:
                finalmaxes.append(current)
                current = -1
        elif x in maximums:
            if current == -1 or sma[x] > sma[current]:
                current = x
    if disp:
        plt.axhline(cutoff)
        plt.plot(brightness)
        plt.plot(sma)
        for point in finalmaxes:
            plt.plot(point, sma[point], 'bo')
        plt.show()
    return finalmaxes





#Example of how to get GCL
#masks, scores, logits = im.get_best_mask([[6000, 3600], [6000, 3200], [6000, 2500], [6000, 4000]], [1, 1, 0, 0])
#im.display_masks(masks, scores)

#Scan to the left of the ventricle
def get_left_GCL(im, vent_box, display_maxes=False):
    scan = []
    store_points = []
    store_labels = []
    step = 160
    for x in range(step):
        target_x = int(vent_box[0]-((0.2+((0.6/step)*x))*vent_box[0]))
        maximums = get_verticle_maxima(target_x, im)
        scan.append(maximums)
        for y in maximums:
            store_points.append([target_x, y])
            store_labels.append(1)

    for x in range(len(scan)):
        if len(scan[x]) == 3 and len(scan[x-1]) == 3 and len(scan[x+1]) == 3 and (scan[x][2]-scan[x][1]) < (scan[x][1]-scan[x][0]) and [(abs(scan[x][y]-scan[x-1][y]) < 0.5*(scan[x][2]-scan[x][1])) for y in range(3)] == [True for y in range(3)] and [(abs(scan[x][y]-scan[x+1][y]) < 0.5*(scan[x][2]-scan[x][1])) for y in range(3)] == [True for y in range(3)]:
            off = int(vent_box[0]-((0.2+((0.6/step)*x))*vent_box[0]))
            points = [
                [off, scan[x][1]],
                [off, scan[x][2]],
                [off, (scan[x][1]+scan[x][2])/2],
                [off, scan[x][1]-(scan[x][2]-scan[x][1])],
                [off, scan[x][2]+(scan[x][2]-scan[x][1])]
            ]
            labels = [1, 1, 0, 0, 0]
            break
    masks, scores, logits = im.get_best_mask(points, labels)

    print(scores)
    if display_maxes:
        for x in range(len(store_points)):
            points.append(store_points[x])
            labels.append(1)
    im.display(masks=masks, points=points, labels=labels)

#Get central ventricle
points = [[10000, 2700]]
labels = [1]
masks, scores, logits = im.get_best_mask(points, labels)
vent_x,vent_y = get_mask_center(masks[0])
vent_box = get_mask_bounds(masks[0])

points = [[int(vent_box[0]*0.5), int((vent_box[1]+vent_box[3])/2)]]
get_left_GCL(im, vent_box, False)
#get_verticle_maxima(4132, im, True)
