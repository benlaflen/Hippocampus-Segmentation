import math
import cv2
from SAMethods import SAM_Image, recommended_kwargs
import numpy as np
from scipy.signal import argrelextrema, convolve2d
from skimage.morphology import skeletonize
from scipy.ndimage import label, center_of_mass, generic_filter
import matplotlib.pyplot as plt
from SegmentHippoCenterVentricle import get_central_ventricle

GCL_Threshold = 0.91
CA3_Threshold = 0.9

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

def ColumnBinarization(image, threshold): #0.2 for detecting blood vessels, #0.8 for detecting GCL and CA3
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    col_thresholds = np.mean(image, axis=0)
    thresholds = col_thresholds[None, :]
    binary_image = image >= thresholds*threshold*2
    return binary_image.astype(np.uint8)

def Threshold(image, threshold):
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    image = image*ColumnBinarization(image, threshold)
    return np.stack([image]*3,axis=2)

def GetComponents(image, threshold, minComponentSize, display=False):
    binary_image = ColumnBinarization(image, threshold)
    labeled_image, num_components = label(binary_image)
    component_sizes = np.bincount(labeled_image.ravel())

    large_labels = np.where(component_sizes >= minComponentSize)[0]
    large_labels = large_labels[large_labels != 0]

    keep_mask = np.isin(labeled_image, large_labels)
    if display:
        plt.figure(figsize=(10,10))
        plt.imshow(keep_mask, cmap="gray")
        for label_id in large_labels:
            y,x = center_of_mass(labeled_image == label_id)
            plt.text(x,y, f"{label_id}\n{component_sizes[label_id]}",
                     color="red", fontsize=8, ha="center", va="center", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        plt.axis("off")
        plt.show()
    return keep_mask.astype(np.uint8), labeled_image, large_labels

def SkeletonizeComponents(keep_mask, display=False):
    blobby_image = blobbyFilter(keep_mask)
    skeletonizedImage = skeletonize(blobby_image, method="lee")
    if display:
        plt.figure(figsize=(10,10))
        plt.imshow(skeletonizedImage, cmap="gray")
        plt.axis("off")
        plt.show()

def blobbyFilter(mask):
    kernel = np.ones((5,5), dtype=np.uint8)
    local_sum = convolve2d(mask, kernel, mode="same", boundary="symm")
    threshold = (kernel.size // 2)
    return (local_sum > threshold).astype(np.uint8)

def GetClosestComponent(contours, point, cutoff=-1000000):
    best_mask = 0
    best_dist = cutoff
    for x in contours.keys():
        dist = cv2.pointPolygonTest(contours[x][0][0], point, True)
        if dist > best_dist:
            best_dist = dist
            best_mask = x
    return best_mask

def get_leftmost_white_pixel(mask):
    white_pixel_indices = np.where(mask == 1)

    if white_pixel_indices[0].size == 0:
        return None  # No white pixel found

    leftmost_col_index = np.argmin(white_pixel_indices[1])
    row = white_pixel_indices[0][leftmost_col_index]
    col = white_pixel_indices[1][leftmost_col_index]

    return (row, col)

def get_rightmost_white_pixel(mask):
    white_pixel_indices = np.where(mask == 1)

    if white_pixel_indices[0].size == 0:
        return None  # No white pixel found

    rightmost_col_index = np.argmax(white_pixel_indices[1])
    row = white_pixel_indices[0][rightmost_col_index]
    col = white_pixel_indices[1][rightmost_col_index]

    return (row, col)





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
            _, _, logits = im.get_best_mask(points, labels)
            masks, scores, logits = im.get_best_mask(points, labels, logits)
            if scores > GCL_Threshold:
                break
            else:
                print("Too small score: " + str(scores))
            x+=10
        x+=1

    if len(scores)==0 or scores[0] < GCL_Threshold:
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
            _, _, logits = im.get_best_mask(points, labels)
            masks, scores, logits = im.get_best_mask(points, labels, logits)
    #        im.display(masks=masks, points=points, labels=labels)
            if scores > GCL_Threshold:
                break
            else:
                print("Too small score: " + str(scores))
            x+=10
        x+=1
    if len(scores)==0 or scores[0] < GCL_Threshold:
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

def count_slice_regions(slice, dist=False, disp=False):
    if len(slice.shape) > 1:
        slice = np.mean(slice, 1)

    window_size = 100
    weights = np.ones(window_size) / window_size
    sma = np.concatenate(([0 for x in range(int(window_size/2))],np.convolve(slice, weights, mode='valid'),[0 for x in range(int(window_size/2))]))

    maximums = argrelextrema(sma, np.greater_equal)[0]

    cutoff = 1.75*np.average(sma)

    finalmaxes = []
    finalsizes = []
    current = -1
    start = 0
    for x in range(len(sma)):
        if sma[x] < cutoff:
            if current != -1:
                finalmaxes.append(current)
                finalsizes.append(x-start)
                current = -1
            start = x
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
    if dist:
        return finalmaxes, finalsizes
    return finalmaxes

def Get_Left_CA3(im, left_gcl):
    #First, get the upper and lower farthest left points of the GCL
    x = 0
    while len(count_slice_regions(slice_image(left_gcl, x))) != 2 and x < len(left_gcl):
        x+=5
    if x > len(left_gcl):
        return None
    spots = count_slice_regions(slice_image(left_gcl, x))
    lower = (x, spots[1]+50)
    upper = (x, spots[0]+50)
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

def Get_Left_CA3_Method_2(im, left_gcl, labeled_image, labels):
    #Setup - get contours for all the components
    contours = {}
    for label in labels:
        mask = np.where(labeled_image == label, 1,0).astype(np.uint8)
        contours[label] = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        '''plt.figure(figsize=(10,10))
        canvas = np.zeros_like(im.image)
        print(contours[label][0][0:5])
        cv2.drawContours(canvas, contours[label][0], -1, (255,255,255), 1)

        plt.imshow(canvas)
        plt.show()'''
    
    #First, get the upper and lower farthest left points of the GCL
    x = 0
    while len(count_slice_regions(slice_image(left_gcl, x))) != 2 and x < len(left_gcl):
        x+=5
    if x > len(left_gcl):
        print("Aborted detecting CA3: Malformed Left GCL")
        return None
    spots = count_slice_regions(slice_image(left_gcl, x))
    lower = (x, spots[1]+50)
    upper = (x, spots[0]+50)
    midpoint = ((upper[0]+lower[0])/2, (upper[1]+lower[1])/2)
    
    #They should each be in or right next to one of our large components - remove those from the list
    best_mask_1_label = GetClosestComponent(contours, upper)
    best_mask_2_label = GetClosestComponent(contours, lower)
    sorted_contours = {k: v for k, v in contours.items() if k not in [best_mask_1_label, best_mask_2_label]}
    #Get the midpoint and the component it's closest to
    CA3Label = GetClosestComponent(sorted_contours, midpoint, cutoff=-abs(upper[1]-lower[1]))
    if CA3Label == 0:
        print("No Left CA3 detected")
        return None
    CA3 = np.where(labeled_image == CA3Label, 1,0)
    
    #Focusing just on that component, get the farthest over we'll go
    regions = count_slice_regions(slice_image(CA3, upper[1], 1))
    if len(regions) > 0:
        termX = regions[0]
        points = [upper, lower, midpoint, (termX, upper[1])]
        labels = [0, 0, 1, 0]
    else:
        termX = get_leftmost_white_pixel(CA3)[1]
        points = [upper, lower, midpoint]
        labels = [0, 0, 1]

    #Get a starting avg estimate
    x = upper[0]-200
    n=0
    avg = 0
    while x > termX+400:
        regions, sizes = count_slice_regions(slice_image(CA3, x, 0), True)
        avg += sizes[-1]
        n+=1
        x -= 1000
    #Now start tracing left from the top point and getting the midpoint in that component
    x = upper[0]-200
    avg /= n
    while x > termX+400:
        regions, sizes = count_slice_regions(slice_image(CA3, x, 0), True)
        avg = ((avg*n) + sizes[-1]) / (n+1)
        n+=1
        if len(regions) == 0 or regions[-1] < upper[1]:
            x -= 50
            continue
        place = regions[-1]
        points.extend([(x, place+50), (x, place - (1.5*avg)+50), (x, place + (1.5*avg)+50)])
        labels.extend([1,0,0])
        x -= 100

    mask1, score1, logit = im.get_best_mask(points=points, labels=labels)
    score = [0]

    x = 0
    while score[0] < CA3_Threshold and x < 10:
        mask, score, logit = im.get_best_mask(points=points, labels=labels, masks=logit)
        if score[0] < CA3_Threshold:
            print("Accuracy on left CA3 too small: " + str(score[0]))
            x += 1
        if score1 > score[0]:
            mask = mask1
            score = [score1]
            break
        mask1 = mask
        score1 = score[0]
    print("Left CA3 detected with accuracy: " + str(score[0]))
    #im.display(masks=[mask1, mask], points=points, labels=labels)
    return mask

def Get_Right_CA3_Method_2(im, right_gcl, labeled_image, labels):
    #Setup - get contours for all the components
    contours = {}
    for label in labels:
        mask = np.where(labeled_image == label, 1,0).astype(np.uint8)
        contours[label] = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        '''plt.figure(figsize=(10,10))
        canvas = np.zeros_like(im.image)
        print(contours[label][0][0:5])
        cv2.drawContours(canvas, contours[label][0], -1, (255,255,255), 1)

        plt.imshow(canvas)
        plt.show()'''
    
    #First, get the upper and lower farthest left points of the GCL
    x = len(right_gcl[0])-1
    while len(count_slice_regions(slice_image(right_gcl, x))) != 2 and x > 0:
        x-=5
    if x <= 0:
        print("Aborted detecting Right CA3: Malformed Right GCL")
        return None
    spots = count_slice_regions(slice_image(right_gcl, x))
    lower = (x, spots[1]+50)
    upper = (x, spots[0]+50)
    midpoint = ((upper[0]+lower[0])/2, (upper[1]+lower[1])/2)
    
    #They should each be in or right next to one of our large components - remove those from the list
    best_mask_1_label = GetClosestComponent(contours, upper)
    best_mask_2_label = GetClosestComponent(contours, lower)
    sorted_contours = {k: v for k, v in contours.items() if k not in [best_mask_1_label, best_mask_2_label]}
    #Get the midpoint and the component it's closest to
    CA3Label = GetClosestComponent(sorted_contours, midpoint, cutoff=-abs(upper[1]-lower[1]))
    if CA3Label == 0:
        print("No Right CA3 detected")
        return None
    CA3 = np.where(labeled_image == CA3Label, 1,0)
    
    #Focusing just on that component, get the farthest over we'll go
    regions = count_slice_regions(slice_image(CA3, upper[1], 1))
    if len(regions) > 0:
        termX = regions[-1]
        points = [upper, lower, midpoint, (termX, upper[1])]
        labels = [0, 0, 1, 0]
    else:
        termX = get_rightmost_white_pixel(CA3)[1]
        points = [upper, lower, midpoint]
        labels = [0, 0, 1]

    #Get a starting avg estimate
    x = upper[0]+200
    n=0
    avg = 0
    while x < termX-400:
        regions, sizes = count_slice_regions(slice_image(CA3, x, 0), True)
        avg += sizes[-1]
        n+=1
        x += 1000
    #Now start tracing left from the top point and getting the midpoint in that component
    x = upper[0]+200
    avg /= n
    while x < termX-400:
        regions, sizes = count_slice_regions(slice_image(CA3, x, 0), True)
        avg = ((avg*n) + sizes[-1]) / (n+1)
        n+=1
        if len(regions) == 0 or regions[-1] < upper[1]:
            x += 50
            continue
        place = regions[-1]
        points.extend([(x, place+50), (x, place - (1.5*avg)+50), (x, place + (1.5*avg)+50)])
        labels.extend([1,0,0])
        x += 100

    mask1, score1, logit = im.get_best_mask(points=points, labels=labels)
    score = [0]

    x = 0
    while score[0] < CA3_Threshold and x < 10:
        mask, score, logit = im.get_best_mask(points=points, labels=labels, masks=logit)
        if score[0] < CA3_Threshold:
            print("Accuracy on Right CA3 too small: " + str(score[0]))
            x += 1
        if score1 > score[0]:
            mask = mask1
            score = [score1]
            break
        mask1 = mask
        score1 = score[0]
    print("Right CA3 detected with accuracy: " + str(score[0]))
    #im.display(masks=[mask], points=points, labels=labels)
    return mask


#Get central ventricle
im = SAM_Image.from_path(r'Cage5195087-Mouse3RL\NeuN-s3.tif', **recommended_kwargs)
keep_mask, labeled_image, component_labels = GetComponents(im.image, 0.8, 100000, False)

masks, scores, logits = get_central_ventricle(im)
#im.display(masks=masks, points=[(7700,3400), (7700,4200)], labels=[1,1])
vent_x,vent_y = get_mask_center(masks)
vent_box = get_mask_bounds(masks)
left_gcl = get_left_GCL(im, vent_box, False)
right_gcl = get_right_GCL(im, vent_box, False)
masks = [masks]
points = []
labels = []
if left_gcl is not None:
    masks.append(left_gcl)
    left_ca3 = Get_Left_CA3_Method_2(im, left_gcl, labeled_image, component_labels)
    if left_ca3 is not None:
        masks.append(left_ca3)
if right_gcl is not None:
    masks.append(right_gcl)
    right_ca3 = Get_Right_CA3_Method_2(im, right_gcl, labeled_image, component_labels)
    if right_ca3 is not None:
        masks.append(right_ca3)
    
im.display(masks=masks, points=points, labels=labels)
