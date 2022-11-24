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


class DoubleConv(nn.Module):
    """
    A double convolution module used to extract features.

    Args:
        in_channels (int): number of input channels. For example for an
            input of shape (batch_size, 3, img_size, img_size) in_channels
            is 3.
        out_channels (int): number of output_channels desired. For example
            if the desired output shape is (batch_size, 3, img_size, img_size)
            in_channels is 3.
        kernel_size (int): A kernel of shape (kernel_size, kernel_size)
            will be applied to the imgs during both Conv2d layers.
        bias (bool): whether or not to add a bias to the Conv2d layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding="same",
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding="same",
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    """A PyTorch implimentation of a UNET image segmentation model based
    on this work: https://arxiv.org/abs/1505.04597. Specifics based on
    Aladdin Persson's implimentation:
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py

    Args:
        in_channels (int): number of input channels. For example for an
            put of shape (batch_size, 3, img_size, img_size) in_channels
            is 3.
        num_classes (int): number of output classes desired. The output
            shape of the model will be (batch_size, num_classes, img_size,
            img_size). For example output[0][i] is a binary segmentation
            mask for class i. Note that class 0 is reserved for background.
        first_feature_num (int): An int specifying the number of features to
            be used in the first DoubleConv layer.
        num_layers (int): Number of layers to use in the UNET architecture.
            The ith layer contains first_feature_num * 2**i features. Note
            that if img_size // 2**num_layers < 1 then the model will break.
        kernel_size (int): A kernel of shape (kernel_size, kernel_size)
            will be applied to the imgs during both Conv2d layers of
            DoubleConv.
        bias (bool): whether or not to add a bias to the DoubleConv Conv2d
            layers.
        track_x_shape (bool): whether or not to track the shape of x.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        first_feature_num=8,
        num_layers=3,
        skip_connect=True,
        kernel_size=3,
        bias=True,
        track_x_shape=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = [first_feature_num * 2**i for i in range(num_layers)]
        self.skip_connect = skip_connect
        self.kernel_size = kernel_size
        self.bias = bias
        self.track_x_shape = track_x_shape

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(self.features[-1], self.features[-1] * 2)
        self.final_conv = nn.Conv2d(self.features[0], self.num_classes, kernel_size=1)

        if self.track_x_shape:
            self.x_shape_tracker = []

        # Down part of UNET.
        for feature in self.features:
            self.downs.append(
                DoubleConv(
                    in_channels, feature, kernel_size=self.kernel_size, bias=self.bias
                )
            )
            in_channels = feature

        # Up part of UNET.
        for feature in reversed(self.features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            if self.skip_connect:
                self.ups.append(
                    DoubleConv(
                        feature * 2,
                        feature,
                        kernel_size=self.kernel_size,
                        bias=self.bias,
                    )
                )
            else:
                self.ups.append(
                    DoubleConv(
                        feature, feature, kernel_size=self.kernel_size, bias=self.bias
                    )
                )

    def track_shape(self, x, description):
        if self.track_x_shape:
            self.x_shape_tracker.append((f"{description}:\n\t {x.shape}"))

        return None

    def forward(self, x):
        skip_connections = []

        self.track_shape(x, "input shape")

        for idx, down in enumerate(self.downs):
            x = down(x)

            self.track_shape(x, f"double_conv (down) {idx}")

            skip_connections.append(x)
            x = self.pool(x)

            self.track_shape(x, f"max_pool {idx}")

        x = self.bottleneck(x)
        self.track_shape(x, "bottleneck")

        # Reverse the list of skip connections.
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)

            self.track_shape(x, f"conv_trans {idx//2}")

            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            if self.skip_connect:
                x = torch.cat((skip_connection, x), dim=1)
                self.track_shape(x, f"skip connection {idx//2}")

            x = self.ups[idx + 1](x)

            self.track_shape(x, f"double_conv (up) {idx//2}")

        x = self.final_conv(x)

        self.track_shape(x, "output shape")

        return x


class LightningImageSegmentation(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        first_feature_num=4,
        num_layers=2,
        skip_connect=True,
        kernel_size=3,
        bias=False,
        weight_loss=None,
        learning_rate = 1e-3
    ):
        super().__init__()

        # LM attributes.
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.first_feature_num = first_feature_num
        self.num_layers = num_layers
        self.skip_connect = skip_connect
        self.kernel_size = kernel_size
        self.bias = bias
        self.learning_rate = learning_rate

        if weight_loss == None:
            self.weight_loss = torch.ones(self.num_classes)
        else:
            self.weight_loss = weight_loss

        # Log hyperparameters.
        self.save_hyperparameters()

        # Metrics.
        self.train_acc = torchmetrics.Accuracy(
            full_state_update="false", mdmc_average="samplewise"
        )
        self.train_f1 = torchmetrics.F1Score(
            full_state_update="false", mdmc_average="samplewise"
        )
        self.train_iou = torchmetrics.JaccardIndex(
            full_state_update="false", num_classes=self.num_classes, average="macro"
        )
        self.val_acc = torchmetrics.Accuracy(
            full_state_update="false", mdmc_average="samplewise"
        )
        self.val_f1 = torchmetrics.F1Score(
            full_state_update="false", mdmc_average="samplewise"
        )
        self.val_iou = torchmetrics.JaccardIndex(
            full_state_update="false", num_classes=self.num_classes, average="macro"
        )

        # Loss function.
        self.loss = nn.CrossEntropyLoss(weight=self.weight_loss)

        # Define Model.
        self.model = UNET(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            first_feature_num=self.first_feature_num,
            skip_connect=self.skip_connect,
            kernel_size=self.kernel_size,
            bias=self.bias,
        )

        # Sample input. Used for logging the model graph.
        self.example_input_array = torch.rand(4, 1, 256, 256)

    def forward(self, imgs):

        imgs_normed = imgs.float() / 255.0
        return self.model(imgs_normed)

    def cross_entropy_loss(self, logits, labels):
        return self.loss(logits, labels)

    def training_step(self, train_batch, batch_idx):

        # Grab images and labels from batch.
        images, labels = train_batch
        logits = self.forward(images)

        # Calculate loss.
        loss = self.cross_entropy_loss(logits, labels)

        # Log step metrics.
        self.train_acc(logits, labels)
        self.train_f1(logits, labels)
        self.train_iou(logits, labels)

        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc, on_step=True)
        self.log("train/f1", self.train_f1, on_step=True)
        self.log("train/iou", self.train_iou, on_step=True)

        return loss

    def validation_step(self, val_batch, batch_idx):

        # Grab images and labels from batch.
        images, labels = val_batch
        logits = self.forward(images)

        # Calculate loss.
        loss = self.cross_entropy_loss(logits, labels)

        # log step metrics.
        self.val_acc(logits, labels)
        self.val_f1(logits, labels)
        self.val_iou(logits, labels)

        self.log("val/loss", loss)
        self.log("val/acc", self.val_acc, on_step=True)
        self.log("val/f1", self.val_f1, on_step=True)
        self.log("val/iou", self.val_iou, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
