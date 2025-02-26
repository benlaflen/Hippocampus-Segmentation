import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# Invert the horse image
image = cv2.imread(r'FiguresData\Figure_9.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if image.ndim == 3:
    image = np.mean(image, axis=2).astype(np.uint8)

# perform skeletonization
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()