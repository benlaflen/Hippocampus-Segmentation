import os
import numpy as np
from os.path import isfile, join
from SegmentAnything.SAMethods import SAM_Image, recommended_kwargs
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

input_folder = "Cage4841876-Mouse3RL" #Folder full of images to label
output_folder = "Training-Database" #Folder to store the outputted images, masks, and points
click_coords = None

colors = [
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0.75,0.75,0],
    [0.75,0,0.75],
    [0,0.75,0.75],
    [0.5,0.5,0.5]
]

def show_mask(mask, ax, color=None, opacity=0.4):
    if color==None:
        color = np.concatenate([np.random.random(3), np.array([opacity])], axis=0)
    else:
        color = np.array(color + [opacity])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25) 

def on_click(event):
    global click_coords
    if event.button is MouseButton.LEFT:
        click_coords = (event.xdata, event.ydata)
        plt.disconnect(binding_id)

for filename in os.listdir(input_folder):
    if(filename[0] == '.'):
        continue
    im = SAM_Image(join(input_folder,filename), **recommended_kwargs)
    for comp in ["Central Ventricle", "GCL", "Hilus", "CA3", "DG"]:
        plt.figure(figsize=(10,10))
        plt.imshow(im.image)
        plt.title(f"Select a point in the " + comp, fontsize=18)
        plt.axis('off')
        binding_id = plt.connect('motion_notify_event', on_click)
        plt.show()