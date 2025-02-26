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

input_file = r"Cage5195087-Mouse3RL\NeuN-s3.tif" #Folder full of images to label

mask = None
image = None
fig = None
im = None
klicker = None
package = None

def update_mask(x,y):
    global mask
    global image
    global fig
    global klicker
    global im
    global points
    global labels
    pos = klicker.get_positions()["positive"]
    neg = klicker.get_positions()["negative"]
    points = []
    labels = []
    for x,y in pos:
        points.append([x,y])
        labels.append(1)
    for x,y in neg:
        points.append([x,y])
        labels.append(0)
    mask, score, logit = im.get_best_mask(points, labels)
    color = np.array([1,0.5,0.5,0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print("Mask shape:", mask.shape)
    print("Image shape:", image.get_array().shape)
    image.set_data(mask_image)
    fig.canvas.draw()

def on_key(event):
    global points
    global labels
    global mask
    global package
    if event.key == 'd':  # Check if the "D" key was pressed
        pack = (mask, points, labels)
        package.append(pack)
        plt.close()  # Close the figure after the key press

im = SAM_Image(input_file, **recommended_kwargs)
masks = []
package = []
mask = None
points = []
labels = []
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im.image)#, origin="lower")
#plt.tight_layout()
plt.title(f"Select points", fontsize=15)
plt.axis('off')
klicker = clicker(ax, ["positive", "negative"], markers=["o", "x"])
shape_array = np.zeros((100, 100))
image = ax.imshow(shape_array, extent=(0, im.image.shape[1], 0, im.image.shape[0]), origin="lower", cmap="Blues", alpha=0.6, vmin=0, vmax=1)
klicker.on_point_added(update_mask)
plt.connect('key_press_event', on_key)
plt.show()