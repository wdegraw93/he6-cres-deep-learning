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
import re

# Necessary for creating our images.
from skimage.draw import line_aa


class CRES_Dataset(torch.utils.data.Dataset):
    """DOCUMENT."""

    def __init__(self, root_dir, freq_bins=4096, max_pool=3, file_max = 10, transform=None):
        """
        Args:
            root_dir (string): Directory with all the spec files and targets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.freq_bins = freq_bins
        self.max_pool = max_pool
        self.file_max = file_max
        self.transform = transform

        self.imgs, self.targets = self.collect_imgs_and_targets()

        # Guarentee the correct type.
        self.imgs, self.targets = self.imgs.type(torch.ByteTensor), self.targets.long()

        # Targets don't need the color dimension.
        self.targets = self.targets.squeeze(1)

        return None

    def __getitem__(self, idx):

        img = self.imgs[idx]
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):

        return len(self.imgs)

    def apply_max_pooling(self, imgs, targets):

        maxpool = nn.MaxPool2d(self.max_pool, self.max_pool, return_indices=True)
        imgs_mp, indices = maxpool(imgs.float())
        targets_mp, indices = maxpool(
            targets.float()
        )  # Not sure if to use the indices or not here...
        # Retrieve corresponding elements from target according to indices.
        # targets_mp = self.retrieve_elements_from_indices(targets.float(), indices)

        return imgs_mp.type(torch.ByteTensor), targets_mp.long()

    def retrieve_elements_from_indices(self, tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=2)
        output = flattened_tensor.gather(
            dim=2, index=indices.flatten(start_dim=2)
        ).view_as(indices)
        return output

    def collect_imgs_and_targets(self):

        img_dir = self.root_dir + "/spec_files"
        target_dir = self.root_dir + "/label_files"

        imgs = self.load_spec_dir(img_dir)
        targets = self.load_spec_dir(target_dir)

        targets = targets.long()

        return imgs, targets

    def load_spec_dir(self, dir_path):
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
            for (file, file_idx) in sorted(
                zip(files, file_idxs), key=lambda pair: pair[1]
            )
        ]

        # Maxpool to use on images and labels.
        maxpool = nn.MaxPool2d(self.max_pool, self.max_pool, return_indices=False)

        if len(files) == 0:
            raise UserWarning("No files found at the input path.")

        imgs = []
        for file in files[:self.file_max]:

            img = self.spec_to_numpy(file)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.permute(0, 2, 1)

            # Apply max pooling now so we never have to hold the large images.

            imgs.append(maxpool(img.float()))

        imgs = torch.stack(imgs)
        return imgs

    def spec_to_numpy(
        self, spec_path, slices=-1, packets_per_slice=1, start_packet=None
    ):
        """
        TODO: Document.
        Making this just work for one packet per spectrum because that works for simulation in Katydid.
        * Make another function that works with 4 packets per spectrum (for reading the Kr data).
        """

        BYTES_IN_PAYLOAD = self.freq_bins
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


class CRES_DM(pl.LightningDataModule):
    """
    Self contained PyTorch Lightning DataModule for testing image
    segmentation models with PyTorch Lightning. Uses the torch dataset
    ImageSegmentation_DS.

    Args:
        train_val_size (int): total size of the training and validation
            sets combined.
        train_val_split (Tuple[float, float]): should sum to 1.0. For example
            if train_val_size = 100 and train_val_split = (0.80, 0.20)
            then the training set will contain 80 imgs and the validation
            set will contain 20 imgs.
        test_size (int): the size of the test data set.
        batch_size (int): batch size to be input to dataloaders. Applies
            for training, val, and test datasets.

    Notes: For now you can decide to shuffle the entire dataset or not but
    the train is always shuffled and the val/test isn't so you can look at
    the same images easily.
    """

    def __init__(
        self,
        root_dir,
        freq_bins=4096,
        max_pool=8,
        file_max = 10,
        transform=None,
        train_val_test_splits=(0.6, 0.3, 0.1),
        batch_size=1,
        shuffle_dataset=True,
        seed=42,
        class_map={
            0: {
                "name": "background",
                "target_color": (0, 0, 0),
            },
            1: {"name": "band 0", "target_color": (255, 0, 0)},
            2: {"name": "band 1", "target_color": (0, 255, 0)},
            3: {"name": "band 2", "target_color": (0, 0, 255)},
        },
    ):

        super().__init__()

        # Attributes.
        self.root_dir = root_dir
        self.freq_bins = freq_bins
        self.max_pool = max_pool
        self.file_max = file_max
        self.transform = transform
        self.class_map = class_map

        self.train_val_test_splits = train_val_test_splits
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.seed = seed

        self.setup()

    def setup(self, stage=None):

        self.cres_dataset = CRES_Dataset(
            self.root_dir,
            freq_bins=self.freq_bins,
            max_pool=self.max_pool,
            file_max = self.file_max,
            transform=self.transform,
        )

        # Creating data indices for training and validation splits:
        dataset_size = len(self.cres_dataset)
        indices = list(range(dataset_size))
        splits = self.train_val_test_splits
        split_idxs = [
            int(np.floor(splits[0] * dataset_size)),
            int(np.floor((splits[0] + splits[1]) * dataset_size)),
        ]

        if self.shuffle_dataset:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[: split_idxs[0]],
            indices[split_idxs[0] : split_idxs[1]],
            indices[split_idxs[1] :],
        )

        # Creating PT data samplers and loaders. For now only train is shuffled.
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.val_sampler = torch.utils.data.SequentialSampler(val_indices)
        self.test_sampler = torch.utils.data.SequentialSampler(test_indices)

        return None

    def train_dataloader(self):
        return DataLoader(
            self.cres_dataset, batch_size=self.batch_size, sampler=self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.cres_dataset, batch_size=self.batch_size, sampler=self.val_sampler
        )

    def test_dataloader(self):
        return DataLoader(
            self.cres_dataset, batch_size=self.batch_size, sampler=self.test_sampler
        )
