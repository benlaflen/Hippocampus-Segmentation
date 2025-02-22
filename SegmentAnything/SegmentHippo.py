import math
from SAMethods import SAM_Image, recommended_kwargs
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from SegmentHippoCenterVentricle import get_central_ventricle

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

def slice_image(im, coord, dir=0):
    if dir == 0:
        return im[:, coord]
    else:
        return im[coord, :]

def count_slice_regions(slice, disp=False):
    if len(slice.shape) > 1:
        slice = np.mean(slice, 1)

    window_size = 100
    weights = np.ones(window_size) / window_size
    sma = np.concatenate(([0 for x in range(int(window_size/2))],np.convolve(slice, weights, mode='valid'),[0 for x in range(int(window_size/2))]))

    maximums = argrelextrema(sma, np.greater_equal)[0]

    cutoff = 1.75*np.average(sma)

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
        plt.plot(slice)
        plt.plot(sma)
        for point in finalmaxes:
            plt.plot(point, sma[point], 'bo')
        plt.show()
    return finalmaxes

def Get_Left_CA3(im, left_gcl):
    #First, get the upper and lower farthest left points of the GCL
    x = 0
    while len(count_slice_regions(slice_image(left_gcl, x))) != 2 and x < len(left_gcl):
        x+=5
    if x > len(left_gcl):
        return None
    spots = count_slice_regions(slice_image(left_gcl, x))
    lower = (x, spots[1])
    upper = (x, spots[0])
    points = [upper, lower]
    labels = [0, 0]
    #our first point is midway between these two
    points.append((x,(lower[1]+upper[1])/2))
    labels.append(1)
    #now move straight left from the upper point until we hit a large bright band
    hband = count_slice_regions(slice_image(im.image, upper[1], 1))
    hband.reverse()
    collision = None
    for point in hband:
        if point < x-(200):
            collision = (point, upper[1])
            break
    if collision == None:
        return None
    #our second point is in the middle of this band
    points.append(collision)
    labels.append(1)
    #our third (negative) point is halfway to the band
    midneg = ((collision[0]+upper[0])/2, upper[1])
    points.append(midneg)
    labels.append(0)
    #now move straight down from that third point until we hit a large bright band
    vband = count_slice_regions(slice_image(im.image, int(midneg[0])), disp=False)
    collision2 = None
    for point in vband:
        if point > midneg[1]+(200):
            collision2 = (midneg[0], point)
            break
    if collision2 == None:
        return None
    #our fourth point is in the middle of this band
    points.append(collision2)
    labels.append(1)

    #Establish line of negative points to the right of the tip
    dist = 2*upper[0]-collision[0]
    points.extend([(dist, upper[1]), (dist, collision2[1]), (dist, lower[1]), (dist, (upper[1]+collision2[1])/2), (dist, (collision2[1]+lower[1])/2)])
    labels.extend([0, 0, 0, 0, 0])

    #Establish line of negative points above CA3
    dist = 2*upper[1]-lower[1]
    points.extend([(upper[0], dist), (midneg[0], dist), (collision[0], dist), (1.5*collision[0]-0.5*midneg[0], dist)])
    labels.extend([0, 0, 0, 0])


    #Get negative point halfway between second point and fourth point
    points.append(((collision[0]+collision2[0])/2,(collision[1]+collision2[1])/2))
    labels.append(0)

    mask,x,y=im.get_best_mask(points, labels)
    im.display(points=points, labels=labels,masks=[mask])
    return mask

#Get central ventricle
im = SAM_Image(r'Cage5195087-Mouse3RL\NeuN-s2.tif', **recommended_kwargs)
masks, scores, logits = get_central_ventricle(im)
vent_x,vent_y = get_mask_center(masks[0])
vent_box = get_mask_bounds(masks[0])
left_gcl = get_left_GCL(im, vent_box, False)
right_gcl = get_right_GCL(im, vent_box, False)
masks = [masks[0]]
points = []
labels = []
if left_gcl is not None:
    masks.append(left_gcl)
if right_gcl is not None:
    masks.append(right_gcl)
if left_gcl is not None:
    left_ca3 = Get_Left_CA3(im, left_gcl)
    if left_ca3 is not None:
        masks.append(left_ca3)
im.display(masks=masks, points=points, labels=labels)