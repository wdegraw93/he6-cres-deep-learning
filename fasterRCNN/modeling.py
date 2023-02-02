# Standard imports
import numpy as np
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, fixed
import seaborn as sns
import sys
import yaml
from pathlib import Path
import shutil
import json
from typing import List, Union
import gc
import zipfile
from pathlib import Path
import re

# Deep learning imports.
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

import torchvision
from torchvision.utils import draw_bounding_boxes, make_grid
from torchvision.ops import masks_to_boxes, box_area
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision, compute_area

# Additional settings.
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

sys.path.append(sys.path[0]+'/..')
from he6_cres_deep_learning.daq import DAQ, Config
root_dir = sys.path[0]+'/config/fasterRCNN'

class CRES_Dataset(torch.utils.data.Dataset):
    """DOCUMENT."""

    def __init__(
        self, root_dir, freq_bins=4096, max_pool=3, file_max=10, transform=None
    ):
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
        self.imgs = self.imgs.type(torch.ByteTensor)

        return None

    def __getitem__(self, idx):

        img = self.imgs[idx]
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):

        return len(self.imgs)

    def collect_imgs_and_targets(self):

        img_dir = self.root_dir + "/spec_files"
        target_dir = self.root_dir + "/label_files"

        # TODO: make it so directories get spec_prefix instead of files
        imgs, exp_name = self.load_spec_dir(img_dir)
#---------------------------------------------------------------------------------------------------        
        # Is this really the best way to scale the bboxes?
        targets = self.load_target_dir(target_dir, exp_name, imgs[0][0].shape)
        # targets = targets.long()

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
        
        # extract experiment name to match to target file 
        exp_name = list(set(re.findall(r'[a-zA-Z0-9]+', name)[0] for name in file_names))
        
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
        for file in files[: self.file_max]:
            img = self.spec_to_numpy(file)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.permute(0, 2, 1)

            # Apply max pooling now so we never have to hold the large images.

            imgs.append(maxpool(img.float()))

        imgs = torch.stack(imgs)

        return imgs, exp_name
    
    def load_target_dir(self, dir_path, exp_name, spec_shape): # spec_shape[0] is frequency, spec_shape[1] is time
        """
        TODO: Document
        Load bbox json files
        """
        path_glob = Path(dir_path).glob(f"{exp_name}*")
        files = [x for x in path_glob if x.is_file()]
        files = [str(x) for x in files]

        if len(files) == 0:
            raise UserWarning("No files found at the input path.")
        
        # targets will be a list of dicts that contain the bboxes, labels, and scores
        targets = []
        targets_dict = {'boxes': [],
                        'labels': []}
        for file in files[: self.file_max]:
            # read all bboxes for experiment
            with open(file, 'r') as f:
                bboxes = json.load(f)
                
                # each value corresponds to a file number
                for file_num, bbox_dict in bboxes.items():
                    # make sure we've populated the dict at least once before appending to list
                    if file_num != '0':
                        targets.append(targets_dict)
                        targets_dict = {'boxes': [],
                                        'labels': []}
                        
                    # each bbox corresponds to an event in the file
                    for bbox in bbox_dict.values():
                        # apply maxpooling reduction before appending
                        bbox = torch.tensor(bbox)/self.max_pool
                        bbox = bbox.round().int()
                        # max pooling can lead to tracks with no pixel width, avoid this
                        if bbox[3] == bbox[1]:
                            bbox[3] += 1
                        if bbox[2] == bbox[0]:
                            bbox[2] += 1
                        targets_dict['boxes'].append(bbox)
                        targets_dict['labels'].append(torch.tensor([1]))
                        
        targets.append(targets_dict)
        targets_dict = {'boxes': [],
                        'labels': []}
        for target in targets:
            target['boxes'] = torch.stack(target['boxes'])
            target['labels'] = torch.tensor(target['labels'])
        return targets
        

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
        file_max=10,
        transform=None,
        train_val_test_splits=(0.6, 0.3, 0.1),
        batch_size=1,
        shuffle_dataset=True,
        seed=42,
        num_workers=0,
        class_map={
            0: {
                "name": "background",
                "target_color": (255, 255, 255),
            },
            1: {"name": "event", "target_color": (255, 0, 0)}
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
        self.num_workers = num_workers
        self.setup()

    def setup(self, stage=None):

        self.cres_dataset = CRES_Dataset(
            self.root_dir,
            freq_bins=self.freq_bins,
            max_pool=self.max_pool,
            file_max=self.file_max,
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

    def collate_fn(self, batch):
        """
        When dealing with lists of target dictionaries one needs to be 
        careful how the batches are collated. The default pytorch dataloader 
        behaviour is to return a single dictionary for the whole batch of 
        images which won't work as input to the mask rcnn model. Instead 
        we want a list of dictionaries; one for each image. See here for 
        more details on the dataloader collate_fn:
        https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3

        Returns:
            imgs (torch.UInt8Tensor[batch_size, 3, img_size, img_size]): 
                batch of images.
            targets (List[Dict[torch.Tensor]]): list of dictionaries of 
                length batch_size.

        """
        imgs = []
        targets = []

        for img, target in batch:
            imgs.append(img)
            targets.append(target)

        # Converts list of tensor images (of shape (3,H,W) and len batch_size)
        # into a tensor of shape (batch_size, 3, H, W).
        imgs = torch.stack(imgs)

        return imgs, targets
    
    def train_dataloader(self):
        return DataLoader(
            self.cres_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.cres_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.cres_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    
class CRES_LM(pl.LightningModule):

    def __init__(self, num_classes = 2, lr = 3e-4, pretrained = False):
        super().__init__()

        # LM Attributes.
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.lr = lr

        # Log hyperparameters. 
        self.save_hyperparameters()

        # Metrics.
        # self.iou = JaccardIndex(task='binary')
        # self.map_bbox = MeanAveragePrecision(iou_type = "bbox", class_metrics = False)

        # Faster RCNN model. 
        self.model = self.get_fasterrcnn_model(self.num_classes, self.pretrained)

    def forward(self, imgs):
        self.model.eval()
        imgs_normed = self.norm_imgs(imgs)
        return self.model(imgs_normed)

    def training_step(self, train_batch, batch_idx):

        imgs, targets = train_batch
        imgs_normed = self.norm_imgs(imgs)

        loss_dict = self.model(imgs_normed, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log('Loss/train_loss', losses)

        return losses

    def validation_step(self, val_batch, batch_idx):

        imgs, targets = val_batch
        preds = self.forward(imgs)
        
        
        iou_list = torch.tensor([box_iou(target["boxes"], pred["boxes"]).diag().mean() for target, pred in zip(targets, preds)])
        # print(iou_list)
        self.log('IoU_bbox/val',iou_list)

        return None

    def configure_optimizers(self): 

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def get_fasterrcnn_model(self, num_classes, pretrained):
        
        # load Faster RCNN pre-trained model
        if pretrained: 
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        else: 
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        
        # get the number of input features 
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

        return model

    def norm_imgs(self, imgs): 

        imgs_normed = imgs.float() / 255.0

        return imgs_normed
    
    
    
    
def main():
    # Define training object
    cres_dm = CRES_DM(root_dir,
                      max_pool=16,
                      file_max=1000,
                      batch_size=1,
                      num_workers=4
                      )
    
    # Create Instance of LightningModule
    cres_lm = CRES_LM(num_classes = 2, lr = 1e-4, pretrained = True)

    # Create callback for ModelCheckpoints. 
    checkpoint_callback = ModelCheckpoint(filename='{epoch:02d}', 
                                          save_top_k = 15, 
                                          monitor = "Loss/train_loss", 
                                          every_n_epochs = 1)

    # Define Logger. 
    logger = TensorBoardLogger("tb_logs", name="cres", log_graph = False)

    # Set device.
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Create an instance of a Trainer.
    trainer = pl.Trainer(logger = logger, 
                         callbacks = [checkpoint_callback], 
                         accelerator = device, 
                         max_epochs = 100, 
                         log_every_n_steps = 1, 
                         check_val_every_n_epoch= 1)

    # Fit. 
    trainer.fit(cres_lm, cres_dm.train_dataloader(), cres_dm.val_dataloader())
    
    return None


if __name__ == "__main__":
    main()