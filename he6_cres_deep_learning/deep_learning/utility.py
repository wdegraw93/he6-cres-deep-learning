# Deep learning imports.
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torchmetrics

# Standard imports.
from typing import List, Union
import gc
import matplotlib.pyplot as plt
import numpy as np
import zipfile
from pathlib import Path

# Necessary for creating our images.
from skimage.draw import line_aa

# Interactive widgets for data viz.
# import ipywidgets as widgets
# from ipywidgets import interact, interact_manual


def show(imgs, figsize=(10.0, 10.0)):
    """Displays a single image or list of images. Taken more or less from
    the pytorch docs:
    https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html#visualizing-a-grid-of-images

    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        figsize (Tuple[float, float]): size of figure to display.

    Returns:
        None
    """

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = TF.to_pil_image(img)
        # TODO: Do I want the cmap to be gray?? May be making the labels weird colors.
        axs[0, i].imshow(np.asarray(img), origin="lower", aspect="auto", cmap="gray")
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None


def spec_to_numpy(
    spec_path, freq_bins=4096, slices=-1, packets_per_slice=1, start_packet=None
):
    """
    TODO: Document.
    Making this just work for one packet per spectrum because that works for simulation in Katydid.
    * Make another function that works with 4 packets per spectrum (for reading the Kr data).
    """

    BYTES_IN_PAYLOAD = freq_bins
    BYTES_IN_HEADER = 32
    BYTES_IN_PACKET = BYTES_IN_PAYLOAD + BYTES_IN_HEADER

    if slices == -1:
        spec_array = np.fromfile(spec_path, dtype="uint8", count=-1).reshape(
            (-1, BYTES_IN_PACKET)
        )[:, BYTES_IN_HEADER:]
    else:
        spec_array = np.fromfile(
            spec_path, dtype="uint8", count=BYTES_IN_PAYLOAD * slices
        ).reshape((-1, BYTES_IN_PACKET))[:, BYTES_IN_HEADER:]

    if packets_per_slice > 1:

        spec_flat_list = [
            spec_array[(start_packet + i) % packets_per_slice :: packets_per_slice]
            for i in range(packets_per_slice)
        ]
        spec_flat = np.concatenate(spec_flat_list, axis=1)
        spec_array = spec_flat

    return spec_array


def load_spec_dir(dir_path, freq_bins=4096):
    """
    Loads all of the images in a directory into torch images.

    Args:
        dir_path (str): path should point to a directory that only contains
            .JPG images. Or any image type compatible with cv2.imread().

        resize_factor (float): how to resize the image. Often one would
            like to reduce the size of the images to be easier/faster to
            use with our maskrcnn model.

    Returns:
        imgs (List[torch.ByteTensor[3, H, W]]): list of images (each a
            torch.ByteTensor of shape(3, H, W)).
    """
    path_glob = Path(dir_path).glob("**/*")
    files = [x for x in path_glob if x.is_file()]
    file_names = [str(x.name) for x in files]
    files = [str(x) for x in files]

    # Extract the file index from the file name.
    file_idxs = [int(re.findall(r"\d+", name)[0]) for name in file_names]

    # Sort the files list based on the file_idx.
    files = [
        file
        for (file, file_idx) in sorted(zip(files, file_idxs), key=lambda pair: pair[1])
    ]

    # Print statement. TODO: Delete this print statement.
    print("\n Loading files : \n")
    for file in files:
        print(file)

    if len(files) == 0:
        raise UserWarning("No files found at the input path.")

    imgs = []
    for file in files:

        img = spec_to_numpy(file, freq_bins=freq_bins)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.permute(0, 2, 1)
        imgs.append(img)

    return imgs


def labels_to_masks(labels):
    """Converts  a batch of segmentation labels into binary masks. Used
    with UNET or in other image segmentation tasks. This function works
    for both batches of labels or single (2d) image labels. The Args and
    return descriptions assume a full batch is input.

    Args:
        labels (torch.int64[batch_size, H, W]): A batch of segmentation
            labels. Each pixel is assigned a class (an integer value).

    Returns:
    binary_masks (torch.bool[batch_size, num_obj_ids, H, W]): A batch of
        corresponding binary masks. Layer i (of dim = 1) corresponds to
        a binary mask for class i. The total number of binary masks will
        be the number of unique object ids (num_obj_ids).
    """

    ids = labels.unique()
    # The below ensures each mask is associated with a given class even if not all
    # classes are present in the given label.
    class_ids = torch.arange(start=ids.min(), end=ids.max() + 1)

    if labels.dim() == 2:
        masks = labels == class_ids[:, None, None]

    if labels.dim() == 3:
        masks = (labels == class_ids[:, None, None, None]).permute(1, 0, 2, 3)

    return masks


def display_masks_unet(imgs, masks, class_map, alpha=0.4):
    """Takes a batch of images and a batch of masks of the same length and
    overlays the images with the masks using the "target_color" specified
    in the class_map.

    Args:
        imgs (List[torch.ByteTensor[batch_size, 3, H, W]]): a batch of
            images of shape (batch_size, 3, H, W).
        masks (torch.bool[batch_size, num_masks, H, W]]): a batch of
            corresponding boolean masks.
        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            key "target_color". class 0 is reserved for background.
            A valid example ("name" not necessary):
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.
        alpha (float): transparnecy of masks. In range (0-1).

    Returns:
        result_imgs (List[torch.ByteTensor[3, H, W]]]): list of images
            with overlaid segmentation masks.
    """
    num_imgs = len(imgs)

    result_imgs = [
        draw_segmentation_masks(
            imgs[i],
            masks[i],
            alpha=alpha,
            colors=[class_map[j]["target_color"] for j in list(class_map.keys())],
        )
        for i in range(num_imgs)
    ]

    return result_imgs
