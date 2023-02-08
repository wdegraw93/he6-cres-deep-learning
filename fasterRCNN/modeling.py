# Standard imports
import numpy as np
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import yaml
import shutil
import json
from typing import List, Union
import gc
import zipfile
from pathlib import Path
import re
import argparse

# Deep learning imports.
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, nms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torchmetrics

def main():
    par = argparse.ArgumentParser()
    arg = par.add_argument
    arg(
        "-rd",
        "--root_dir",
        type=str,
        default=sys.path[0]+'/config/fasterRCNN',
        help="Path to the directory that contains the label_files and spec_files directories",
    )
    arg(
        '-fb',
        '--freq_bins',
        type=int,
        default=4096,
        help='Number of frequency bins in spec files'
    )
    arg(
        '-mp',
        '--max_pool',
        type=int,
        default=16,
        help='Max pooling factor to apply to data'   
    )
    arg(
        '-f',
        '--file_max',
        type=int,
        default=10,
        help='Max number of files to pull from spec_dir'
    )
    arg(
        '-t',
        '--transform',
        type=str,
        default=None,
        help='Transformation to apply to image'
    )
    arg(
        '-ts',
        '--train_val_test_splits',
        type=tuple,
        default=(0.6,0.3,0.1),
        help='Fraction of data to use for train/val/test splits (tf, vf, test_f)'
    )
    arg(
        '-bs',
        '--batch_size',
        type=int,
        default=1,
        help='batch size for dataloaders'
    )
    arg(
        '-sd',
        '--shuffle_dataset',
        type=bool,
        default=True,
        help='Whether or not to shuffle train/val datasets during training'
    )
    arg(
        '-seed',
        '--seed',
        type=int,
        default=42,
        help='Random seed to use for shuffling of data'
    )
    arg(
        '-nw',
        '--num_workers',
        type=int,
        default=0,
        help='Number of processors to use for loading data'
    )
    arg(
        '-cm',
        '--class_map',
        type=dict,
        default={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "event", "target_color": (255, 0, 0)}
        },
        help='Dict describing the class structure for the model'
    )
    arg(
        '-nc',
        '--num_classes',
        type=int,
        default=2,
        help='Number of classes for the model'
    )
    arg(
        '-lr',
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for Adam algorithm'
    )
    arg(
        '-p',
        '--pretrained',
        type=bool,
        default=True,
        help='Option to use pretrained weights for the resnet model'
    )
    arg(
        '-e',
        '--max_epochs',
        type=int,
        default=5,
        help='Epochs to use for training'
    )
    
    args = par.parse_args()

    # Define training object
    cres_dm = CRES_DM(root_dir = args.root_dir,
                      freq_bins = args.freq_bins,
                      max_pool = args.max_pool,
                      file_max = args.file_max,
                      batch_size = args.batch_size,
                      num_workers = args.num_workers,
                      transform = args.transform,
                      shuffle_dataset = args.shuffle_dataset,
                      seed = args.seed,
                      class_map = args.class_map,
                      train_val_test_splits = args.train_val_test_splits
                      )

    
    # Create Instance of LightningModule
    cres_lm = CRES_LM(num_classes = args.num_classes,
                      lr = args.learning_rate,
                      pretrained = args.pretrained)

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
                         max_epochs = args.max_epochs, 
                         log_every_n_steps = 1, 
                         check_val_every_n_epoch= 1)

    # Fit. 
    trainer.fit(cres_lm, cres_dm.train_dataloader(), cres_dm.val_dataloader())
    
    return None



class CRES_Dataset(torch.utils.data.Dataset):
    """
    Following the structure necessitated by PyTorch, defining the Dataset class for this
    data. This involves the required __init__ and __getitem__ functions, but also
    will be responsible for formatting the data into the torch.tensor structures required
    for the RCNN model. 
    """

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

#--------------------------------------------------------------------------------------------------- 
        # TODO: make it so directories get spec_prefix instead of files
        imgs, exp_name = self.load_spec_dir(img_dir)
#---------------------------------------------------------------------------------------------------

        targets = self.load_target_dir(target_dir, exp_name)
    
        return imgs, targets

    def load_spec_dir(self, dir_path):
        """
        Loads all of the images in a directory into torch images.

        Args:
            dir_path (str): path should point to a directory that only contains
                .JPG images. Or any image type compatible with cv2.imread().

        Returns:
            imgs (List[torch.ByteTensor[1, H, W]]): list of images (each a
                torch.ByteTensor of shape(1, H, W)). Also returns the experiment
                name in this case, as each simulation is not yet being aggregated
                to ensure accurate matching of file numbers/events. 
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
        
        # Maxpool to use on images
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
    
    def load_target_dir(self, dir_path, exp_name): 
        """
        Load bbox json files. Apply max pooling reduction and ensure
        boxes still make physical sense. 
        Returns list of dictionaries with keys 'boxes' and 'labels', 
        boxes are in [x1, y1, x2, y2] format, and labels will all be 1
        since this simulation is only looking for events. 
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
    The LightningDataModule handles the train/test splitting, and defines
    the DataLoaders for each of these cases. For use with the LightningModule,
    helps reduce the amount of boilerplate code that needs to be written. The
    class structure is defined here as a binary classification problem - either
    the bounding box surrounds an event or not. The custom collate function 
    ended up being the most finnicky part of the process. 
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
            imgs (torch.UInt8Tensor[batch_size, 1, img_size, img_size]): 
                batch of images.
            targets (List[Dict[torch.Tensor]]): list of dictionaries of 
                length batch_size.

        """
        imgs = []
        targets = []

        for img, target in batch:
            imgs.append(img)
            targets.append(target)

        # Converts list of tensor images (of shape (1,H,W) and len batch_size)
        # into a tensor of shape (batch_size, 1, H, W).
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
        
        # Applying Non-maximum suppression to prediction boxes to accuractely keep track of IoU
        # Store indices to keep in a list
        keep_list = [nms(pred['boxes'], pred['scores'], iou_threshold=.3) for pred in preds]
        # All IoU's greater than 0 kept in following list
        iou_list = []
        for target in targets:
            for i in range(len(keep_list)):
                iou = box_iou(target['boxes'], preds[i]['boxes'][keep_list[i]])
                iou = iou[iou>0]
                iou_list += list(iou)
                
        # Log values
        self.log('IoU_bbox/val',torch.tensor(iou_list).mean())
        self.log('Prediction_shape/val', float(len(preds[0]['boxes'])))
        self.log('targets_shape/val', float(len(targets[0]['boxes'])))
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
    


if __name__ == "__main__":
    main()