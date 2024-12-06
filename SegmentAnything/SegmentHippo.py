import math
from SAMethods import SAM_Image, recommended_kwargs
import numpy as np
from scipy.signal import argrelextrema
im = SAM_Image(r'Cage5195087-Mouse3RL\NeuN-s3.tif', **recommended_kwargs)
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

def PointDist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
def PointAngle(p1, p2, p3):
    return math.atan2(p3[1] - p1[1], p3[0] - p1[0]) - math.atan2(p2[1] - p1[1], p2[0] - p1[0])
def get_bright_lines(im, vent_box, display_maxes=False):
    scan = []
    store_points = []
    step = 160
    for x in range(step):
        target_x = int(vent_box[0]-((0.2+((0.6/step)*x))*vent_box[0]))
        maximums = get_verticle_maxima(target_x, im)
        scan.append(maximums)
        for y in maximums:
            store_points.append([target_x, y])

    edges = []
    #Detect Lines
    for x,first in enumerate(store_points):
        NN = None
        dist = -1
        for y,second in enumerate(store_points):
            thist = PointDist(first, second)
            if x == y:
                continue
            if NN == None:
                NN = second
                dist = thist
            elif thist < dist:
                NN = second
                dist = thist
        edges.append((first, NN))
        SN = None
        dist = -1
        for y,second in enumerate(store_points):
            thist = PointDist(first, second)
            if x == y:
                continue
            thangle = PointAngle(first, second, NN)
            if thangle < math.pi/2:
                continue
            if SN == None:
                SN = second
                dist = thist
            elif thist < dist:
                SN = second
                dist = thist
        if SN != None:
            edges.append((first, SN))
    edge_distances = np.array([PointDist(x[0], x[1]) for x in edges])
    max_length = np.quantile(edge_distances, 0.75) + 1.5*(np.quantile(edge_distances, 0.75)-np.quantile(edge_distances, 0.25))
    index = 0
    while(index < len(edges)):
        if PointDist(edges[index][0], edges[index][1]) > max_length or (edges[index][1], edges[index][0]) not in edges:
            del edges[index]
        else:
            index += 1
    plt.figure(figsize=(10,10))
    for edge in edges:
        plt.plot([edge[0][0], edge[1][0]], [-edge[0][1], -edge[1][1]] )
    plt.show()

def get_left_GCL(im, vent_box, display_maxes=False):
    scan = []
    store_points = []
    labels = []
    step = 160
    for x in range(step):
        target_x = int(vent_box[0]-((0.2+((0.6/step)*x))*vent_box[0]))
        maximums = get_verticle_maxima(target_x, im)
        scan.append(maximums)
        for y in maximums:
            store_points.append([target_x, y])
            labels.append(1)
    if display_maxes:
        im.display(points=store_points, labels=labels)
    points = []
    labels = []
    scores = []
    x=1
    while x <len(scan)-1:
        if len(scan[x]) == 3 and len(scan[x-1]) == 3 and len(scan[x+1]) == 3 and (scan[x][2]-scan[x][1]) < (scan[x][1]-scan[x][0]) and [(abs(scan[x][y]-scan[x-1][y]) < 0.5*(scan[x][2]-scan[x][1])) for y in range(3)] == [True for y in range(3)] and [(abs(scan[x][y]-scan[x+1][y]) < 0.5*(scan[x][2]-scan[x][1])) for y in range(3)] == [True for y in range(3)]:
            off = int(vent_box[0]-((0.2+((0.6/step)*x))*vent_box[0]))
            points += [
                [off, scan[x][1]],
                [off, scan[x][2]],
                [off, (scan[x][1]+scan[x][2])/2],
                [off, scan[x][1]-(scan[x][2]-scan[x][1])],
                [off, scan[x][2]+(scan[x][2]-scan[x][1])]
            ]
            labels += [1, 1, 0, 0, 0]
            masks, scores, logits = im.get_best_mask(points, labels)
            if scores > 0.85:
                break
            else:
                print("Too small score: " + str(scores))
            x+=10
        x+=1

    if len(scores)==0 or scores[0] < 0.85:
        print("No Left GCL Detected!")
        return None
    else:
        print("Left GCL detected with accuracy: " + str(scores[0]))
        if display_maxes:
            for x in range(len(store_points)):
                points.append(store_points[x])
                labels.append(1)
        #im.display(masks=masks, points=points, labels=labels)
        return masks[0]

def get_right_GCL(im, vent_box, display_maxes=False):
    scan = []
    store_points = []
    labels = []
    step = 160
    for x in range(step):
        target_x = int(vent_box[2]+((0.2+((0.6/step)*x))*(im.image.shape[1] - vent_box[2])))
        maximums = get_verticle_maxima(target_x, im)
        scan.append(maximums)
        for y in maximums:
            store_points.append([target_x, y])
            labels.append(1)
    if display_maxes:
        im.display(points=store_points, labels=labels)
    points = []
    labels = []
    scores = []
    x=1
    while x< len(scan)-1:
        if len(scan[x]) == 3 and len(scan[x-1]) == 3 and len(scan[x+1]) == 3 and (scan[x][2]-scan[x][1]) < (scan[x][1]-scan[x][0]) and [(abs(scan[x][y]-scan[x-1][y]) < 0.5*(scan[x][2]-scan[x][1])) for y in range(3)] == [True for y in range(3)] and [(abs(scan[x][y]-scan[x+1][y]) < 0.5*(scan[x][2]-scan[x][1])) for y in range(3)] == [True for y in range(3)]:
            off = int(vent_box[2]+((0.2+((0.6/step)*x))*(im.image.shape[1] - vent_box[2])))
            points += [
                [off, scan[x][1]],
                [off, scan[x][2]],
                [off, (scan[x][1]+scan[x][2])/2],
                [off, scan[x][1]-(scan[x][2]-scan[x][1])],
                [off, scan[x][2]+(scan[x][2]-scan[x][1])]
            ]
            labels += [1, 1, 0, 0, 0]
            masks, scores, logits = im.get_best_mask(points, labels)
    #        im.display(masks=masks, points=points, labels=labels)
            if scores > 0.85:
                break
            else:
                print("Too small score: " + str(scores))
            x+=10
        x+=1
    if len(scores)==0 or scores[0] < 0.85:
        print("No Right GCL Detected!")
        return None
    else:
        print("Right GCL detected with accuracy: " + str(scores[0]))
        if display_maxes:
            for x in range(len(store_points)):
                points.append(store_points[x])
                labels.append(1)
    #    im.display(masks=masks, points=points, labels=labels)
        return masks[0]

#Get central ventricle
points = [[9600, 2600], [11300, 2600]]
labels = [1, 1]
masks, scores, logits = im.get_best_mask(points, labels)
vent_x,vent_y = get_mask_center(masks[0])
vent_box = get_mask_bounds(masks[0])
left_gcl = get_left_GCL(im, vent_box, False)
right_gcl = get_right_GCL(im, vent_box, False)
masks = []
if left_gcl is not None:
    masks.append(left_gcl)
if right_gcl is not None:
    masks.append(right_gcl)
im.display(masks=masks, boxes=[vent_box], points=points, labels=labels)