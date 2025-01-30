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

mask = None
image = None
fig = None
im = None
klicker = None

def update_mask(x,y):
    global mask
    global image
    global fig
    global klicker
    global im
    pos = klicker.get_positions()["positive"]
    neg = klicker.get_positions()["negative"]
    points = []
    labels = []
    for x,y in pos:
        points.append([x,y])
        labels.append(0)
    for x,y in neg:
        points.append([x,y])
        labels.append(1)
    mask, score, logit = im.get_best_mask(points, labels)
    with open("out.txt", "w") as file:
        for arr in mask[0]:
            for truth in arr:
                file.write(str(truth) + ", ")
            file.write("\n")
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print("Mask shape:", mask.shape)
    print("Image shape:", image.get_array().shape)
    image.set_data(mask_image)
    fig.canvas.draw()
    

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
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(im.image, origin="lower")
        plt.tight_layout()
        plt.title(f"Select points until the " + comp + " is fully highlighted, then press D.", fontsize=15)
        plt.axis('off')
        klicker = clicker(ax, ["positive", "negative"], markers=["o", "x"])
        shape_array = np.zeros((100, 100))
        image = ax.imshow(shape_array, extent=(0, im.image.shape[1], 0, im.image.shape[0]), origin="lower", cmap="Blues", alpha=0.6, vmin=0, vmax=1)
        klicker.on_point_added(update_mask)
        plt.show()
        

