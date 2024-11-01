from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

recommended_kwargs = {
    "sam_checkpoint": "SegmentAnything/sam_vit_h_4b8939.pth",
    "model_type": "vit_h",
    "device": "cuda"
}

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

class SAM_Image:
    def __init__(self, image_path, sam_checkpoint, model_type, device):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)
        self.predictor.set_image(self.image)
    
    #We'll take the raw arrays in and call np.array for the user. Exception is existing masks, since we assume we'll generate that
    #Returns masks, scores, logits
    def get_masks(self, points=None, labels=None, masks=None, boxes=None):
        points = np.array(points) if points != None else None
        labels = np.array(labels) if labels != None else None
        boxes = np.array(boxes) if boxes != None else None
        return self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=masks,
            box=boxes,
            multimask_output=True,
        )

    def get_best_mask(self, points=None, labels=None, masks=None, boxes=None):
        points = np.array(points) if points != None else None
        labels = np.array(labels) if labels != None else None
        boxes = np.array(boxes) if boxes != None else None
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=masks,
            box=boxes,
            multimask_output=True,
        )
        best_logit = logits[np.argmax(scores), :, :]
        return self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=best_logit[None, :, :],
            box=boxes,
            multimask_output=False,
        )

    def display_masks(self, masks, scores):
        plt.figure(figsize=(10,10))
        plt.imshow(self.image)
        for i,mask in enumerate(reversed(masks)):
            show_mask(mask, plt.gca(), color=colors[i % len(colors)])
        plt.title(f"Masks", fontsize=18)
        plt.axis('off')
        plt.show() 