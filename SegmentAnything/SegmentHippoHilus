import numpy as np
from SAMethods import SAM_Image, recommended_kwargs
from SegmentHippo import get_left_GCL, get_right_GCL
import matplotlib.pyplot as plt


def get_mask_bounds(mask):
    nz = np.nonzero(mask)
    return np.min(nz[1]), np.min(nz[0]), np.max(nz[1]), np.max(nz[0])


def find_bottom_and_center_points(mask, ventricle_box):
    y_coords, x_coords = np.where(mask)
    mask_points = np.array(list(zip(x_coords, y_coords)))

    bottommost_idx = np.argmax(mask_points[:, 1])  # Largest y-coordinate
    bottommost_point = tuple(mask_points[bottommost_idx])

    center_x = (ventricle_box[0] + ventricle_box[2]) // 2
    closest_to_center_idx = np.argmin(np.abs(mask_points[:, 0] - center_x))
    closest_to_center_point = tuple(mask_points[closest_to_center_idx])

    return bottommost_point, closest_to_center_point


def find_middle_point(mask, bottommost_point, center_point, find_highest=False):
    midpoint_x = (bottommost_point[0] + center_point[0]) // 2
    y_coords_at_x = np.where(mask[:, midpoint_x])[0]

    if len(y_coords_at_x) < 2:
        return None

    top_y = np.min(y_coords_at_x)
    bottom_y = np.max(y_coords_at_x)

    return midpoint_x, bottom_y if find_highest else top_y


def find_tail_point(mask):
    y_coords, x_coords = np.where(mask)
    mask_points = np.array(list(zip(x_coords, y_coords)))

    bottommost_idx = np.argmax(mask_points[:, 1])  # Largest y-coordinate
    return tuple(mask_points[bottommost_idx])


def find_intermediate_point(mask, tail_point, middle_point):
    midpoint_x = (tail_point[0] + middle_point[0]) // 2
    y_coords_at_x = np.where(mask[:, midpoint_x])[0]

    if len(y_coords_at_x) < 2:
        return None

    top_y = np.min(y_coords_at_x)
    bottom_y = np.max(y_coords_at_x)
    return midpoint_x, (top_y + bottom_y) // 2


def find_ca3_border_point(mask, tail_point):
    x_coord = tail_point[0]
    y_coords_at_x = np.where(mask[:, x_coord])[0]

    if len(y_coords_at_x) == 0:
        return None

    nearest_mask_y = np.min(y_coords_at_x)
    return x_coord, (tail_point[1] + nearest_mask_y) // 2


def sample_negative_points(mask, num_points=20):
    y_coords, x_coords = np.where(mask)
    mask_points = np.array(list(zip(x_coords, y_coords)))

    if len(mask_points) <= num_points:
        return [tuple(point) for point in mask_points]

    indices = np.linspace(0, len(mask_points) - 1, num_points).astype(int)
    return [tuple(mask_points[idx]) for idx in indices]


def create_sam_mask_for_gcl(im, ventricle_box, mask, num_negative_points=20):
    """
    Creates a SAM mask using positive and negative points for a GCL region.
    """
    bottommost_point, center_point = find_bottom_and_center_points(mask, ventricle_box)
    middle_point_low = find_middle_point(mask, bottommost_point, center_point, find_highest=False)
    middle_point_high = find_middle_point(mask, bottommost_point, center_point, find_highest=True)
    midpoint_between_middle_points = None
    if middle_point_low and middle_point_high:
        midpoint_between_middle_points = (
            middle_point_low[0],
            (middle_point_low[1] + middle_point_high[1]) // 2,
        )

    tail_point = find_tail_point(mask)
    intermediate_point = None
    if tail_point and midpoint_between_middle_points:
        intermediate_point = find_intermediate_point(mask, tail_point, midpoint_between_middle_points)

    ca3_border_point = find_ca3_border_point(mask, tail_point)

    # Compile positive and negative points
    positive_points = [
        midpoint_between_middle_points,
        intermediate_point,
        # ca3_border_point,
    ]
    positive_points = [p for p in positive_points if p is not None]

    negative_points = sample_negative_points(mask, num_points=num_negative_points)

    points = positive_points + negative_points
    labels = [1] * len(positive_points) + [0] * len(negative_points)

    masks, scores, logits = im.get_best_mask(points, labels)
    return masks, scores, logits, points, labels


def display_sam_mask(im, masks, points, labels, title="SAM Mask with Points"):
    """
    Visualize the SAM mask with positive and negative points.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(im.image, cmap='gray')
    plt.imshow(masks[0], alpha=0.5, cmap='Reds')

    for point, label in zip(points, labels):
        color = 'green' if label == 1 else 'red'
        plt.scatter(*point, color=color, s=100)

    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend(["Positive Points", "Negative Points"])
    plt.show()


# Main Script
im = SAM_Image(r'Cage5195087-Mouse3RL\NeuN-s3.tif', **recommended_kwargs)
masks, scores, logits = im.get_best_mask([[9600, 2600], [11300, 2600]], [1, 1])
ventricle_box = get_mask_bounds(masks[0])

# Process Left GCL
left_gcl_mask = get_left_GCL(im, ventricle_box, False)
if left_gcl_mask is not None:
    left_masks, left_scores, left_logits, left_points, left_labels = create_sam_mask_for_gcl(
        im, ventricle_box, left_gcl_mask, num_negative_points=20
    )
    display_sam_mask(im, left_masks, left_points, left_labels, title="Left GCL SAM Mask")

# Process Right GCL
right_gcl_mask = get_right_GCL(im, ventricle_box, False)
if right_gcl_mask is not None:
    right_masks, right_scores, right_logits, right_points, right_labels = create_sam_mask_for_gcl(
        im, ventricle_box, right_gcl_mask, num_negative_points=20
    )
    display_sam_mask(im, right_masks, right_points, right_labels, title="Right GCL SAM Mask")
else:
    print("No GCL mask detected.")
