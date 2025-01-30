import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

clicker_instance = clicker(ax, ["points"], markers=["o"])

# Create a 100x100 grid and initialize the imshow object
shape_array = np.zeros((100, 100))
im = ax.imshow(shape_array, extent=(0, 10, 0, 10), origin="lower", cmap="Blues", alpha=0.6, vmin=0, vmax=1)

def update_shape(x,y):
    points = clicker_instance.get_positions()["points"]

    # Clear the shape array
    shape_array.fill(0)

    # Add circular regions around points
    for x, y in points:
        grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
        mask = np.sqrt((grid_x - x)**2 + (grid_y - y)**2) < 0.5
        shape_array[mask] = 1

    # Update imshow
    im.set_data(shape_array)
    fig.canvas.draw()

clicker_instance.on_point_added(update_shape)

plt.show()