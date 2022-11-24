# DL Imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys

sys.path.append("/home/drew/He6CRES/he6-cres-deep-learning/")
from he6_cres_deep_learning.deep_learning import ds
from he6_cres_deep_learning.deep_learning import util
from he6_cres_deep_learning.deep_learning import model


# ---- Train a UNET model. ----

experiment_name = "simple_ds"
dataset_path = (
    f"/media/drew/T7 Shield/cres_deep_learning/training_data/config/{experiment_name}"
)
logger_path = "/home/drew/He6CRES/he6-cres-deep-learning_results/tb_logs"

cres_dm = ds.CRES_DM(root_dir=dataset_path, max_pool=16, file_max=49, batch_size=8)

# Define weights for loss function.
weights = torch.tensor([1, 10]).float()

# Create callback for ModelCheckpoints.
checkpoint_callback = ModelCheckpoint(
    filename="{epoch:02d}", save_top_k=50, monitor="val/loss", every_n_epochs=1
)

# Define Logger.
logger = TensorBoardLogger(logger_path, name=experiment_name, log_graph=True)

# Create Instance of Lightning Module.
img_seg_lm = model.LightningImageSegmentation(
    in_channels=1,
    num_classes=2,
    first_feature_num=4,
    num_layers=2,
    skip_connect=True,
    kernel_size=3,
    bias=False,
    weight_loss=weights,
)

# -----------Set device.------------------
device = "gpu" if torch.cuda.is_available() else "cpu"

# Create an instance of a Trainer.
trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    accelerator=device,
    max_epochs=50,
    log_every_n_steps=2,
)

# Fit.
trainer.fit(img_seg_lm, cres_dm)
