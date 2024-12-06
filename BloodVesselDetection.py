import cv2
import matplotlib.pyplot as plt
import numpy as np


image_path = r'Cage5195087-Mouse3RL\NeuN-s1.tif'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#_, newim = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY_INV)
newim = [[]]
window = 100
total = (len(image)-(2*window))*(len(image[0])-(2*window))
count = 0
for x in range(window,len(image)-window):
    for y in range(window,len(image[x])-window):
        newim[len(newim)-1].append(127*(image[x][y][0] / np.average(image[x-window:x+window][y-window:y+window])))
        count+=1
        if(count%100000 == 0):
            print(str(count) + "/" + str(total) + "\n")
    newim.append([])
plt.figure(figsize=(10,10))
#plt.imshow(image)
plt.imshow(newim)#, alpha=0.5)
plt.show()