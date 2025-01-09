import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


image_path = r'Cage5195087-Mouse3RL\NeuN-s1.tif'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, newim = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(newim,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#newim = cv2.bitwise_not(image)
"""window = 100
newim = [[0 for _ in range(len(image[0])-(2*window))] for _ in range(len(image)-(2*window))]
total = (len(image)-(2*window))*(len(image[0])-(2*window))
count = 0
start = time.time()
for x in range(window,len(image)-window):
    for y in range(window,len(image[x])-window):
        newim[x-window][y-window] = 127*(image[x][y][0] / np.average(image[x-window:x+window][y-window:y+window]))
        count+=1
        if(count%100000 == 0):
            print(str(count) + "/" + str(total) + "  Estimated total time: " + str((time.time()-start) * (total/count)) + "s\n")
plt.figure(figsize=(10,10))"""
#plt.imshow(image)
index = 0
conts = []
while index < len(contours):
    if len(contours[index]) > 1:
        conts.append(contours[index])
        print(contours[index][0])
    index+=1
for arr in conts:
    tap = arr[0].copy()
    tap = np.vstack([tap, tap[0]])
    x,y = tap[:,0],tap[:,1]
    plt.fill(x,y, color='lightblue')
plt.show()