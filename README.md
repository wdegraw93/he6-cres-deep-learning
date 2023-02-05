
<p align="center"><img width="75%" src="readme_imgs/he6-cres_logo.png"/></p>

--------------------------------------------------------------------------------
# Introduction
For my capstone project at General Assembly I chose to build off of the existing work by my former colleague, Drew Byron. I forked his he6_cres_deep_learning repo which has the simulation software necessary for generating spec files as seen in the He6-CRES experiment at the University of Washington, and building off examples in PyTorch in his tutorial series at https://github.com/drewbyron/pytorch-tutorials I built the framework necessary for training a Faster R-CNN object detection model on simulated CRES data. In order to accelerate the work a GPU instance was used on Amazon's AWS for the training of the model. This repo details the work done in order to get the simulation and modeling pipeline to a functional state, and this README serves as the instruction manual for how to generate your own spec files to train a Faster R-CNN model for event reconstruction in CRES events. 

--------------------------------------------------------------------------------
## he6_cres_deep_learning

This folders contains modules and scripts for transforming track information data into simulated spectrogram images as seen by the He6-CRES experiment.

--------------------------------------------------------------------------------
## fasterRCNN

This folder contains the work for this project. Three notebooks are present, the two `towards_faster_rcnn` and `towards_modeling` are where the spec simulation and modeling pipelines were figured out, respectively. 

--------------------------------------------------------------------------------
### Train a model to detect cres events.

#### Look at track and event classification overlaid on top of raw data: 


--------------------------------------------------------------------------------
### Instructions for how to build a training dataset: 

#### Overview: 

Below we will review how to use this package to create three different training datasets. Each dataset will consist of 50 35ms spec files along with label files. The first dataset contains simple high snr tracks with no snr fluctuations (two class; track, background). The second dataset contains tracks with severe snr fluctuations (two class; track, background). The third dataset contains tracks with sidebands (three class; main_band, sideband, background). The point of this study is to illustrate that this approach could work for us and potentially help us with the most difficult problems to solve in our analysis; snr fluctuations and sidebands. We would need both more realistic training data and much more training data to train and deploy a data-ready model. But the fact that we are seeing reasonable performance with 50 spec files is indicative that this is a promising avenue. Also, the fact that the max pooling doesn't significantly disrupt detection efficiencies we may be able to apply this to our raw data and save a factor of 64 or more in data written to disk.

#### Instructions: 


* **Build a Dataset**
	* `cd he6-cres-deep-learning`
	* `python3 ./build_simple_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 50 -n_events 4 -len .035 -seed 24436 -sanity_check False`
	* For the other datasets replace the `.py` file with the appropriate module. 