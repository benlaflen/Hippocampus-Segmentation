import os
import numpy as np
from os.path import isfile, join, splitext
from SegmentAnything.SAMethods import SAM_Image, recommended_kwargs
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import pickle
import time
import shutil
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler

input_folder = "Cage4841876-Mouse3RL" #Folder full of images to label
output_folder = "Training-Database" #Folder to store the outputted images, masks, and points
click_coords = None
key_pressed = False

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

def on_key(event):
    global key_pressed
    if event.key == 'd':  # Check if the "D" key was pressed
        key_pressed = True
        plt.close()  # Close the figure after the key press
    elif event.key == 'r':
        plt.close()

for filename in os.listdir(input_folder):
    if filename[0] == '.' or splitext(filename)[0] in os.listdir(output_folder):
        continue
    im = SAM_Image(join(input_folder,filename), **recommended_kwargs)
    masks = []
    out_points = []
    for comp in ["Central Ventricle", "Left GCL", "Left Hilus", "Left CA3", "Left DG", "Right GCL", "Right Hilus", "Right CA3", "Right DG"]:
        mask = None
        points = []
        labels = []
        key_pressed = False
        while not key_pressed:            
            click_coords = None
            plt.figure(figsize=(10,10))
            plt.imshow(im.image)
            plt.title(f"Select points until the " + comp + " is fully highlighted, then press D.", fontsize=15)
            plt.axis('off')
            if mask is not None:
                show_mask(mask, plt.gca(), color=colors[0])
            for point in points:
                plt.plot(point[0], point[1], 'bo')
            
            plt.connect('key_press_event', on_key)
            zoom_factory(plt.gca())
            ph = panhandler(plt.gcf(), button=2)
            clicks = clicker(
                plt.gca(),
                ["positive", "negative"],
                markers=["o", "x"]
            )
            plt.show()
        if mask is not None:
            masks.append(mask)
            out_points.append(points)
    if masks != []:
        os.mkdir(join(output_folder,splitext(filename)[0]))
        with open(join(output_folder,splitext(filename)[0],"masks"), "wb") as file:
            pickle.dump(masks, file)
        with open(join(output_folder,splitext(filename)[0],"points"), "wb") as file:
            pickle.dump(points, file)
        shutil.copy(join(input_folder,filename), join(output_folder,splitext(filename)[0],"image"))
        