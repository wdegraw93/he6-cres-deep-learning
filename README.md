
<p style="text-align: center"><img width="75%" src="readme_imgs/he6-cres_logo.png"/></p>

--------------------------------------------------------------------------------
# Introduction
For my capstone project at General Assembly I chose to build off of the existing work by my former colleague, Drew Byron, to test the feasibility of using object detection CNN's for event reconstruction in the He6-CRES experiment. I forked his `he6_cres_deep_learning` repo which has the simulation software necessary for generating the kind of spectrogram images seen in the experiment at the University of Washington, edited it to include boundary box ground-truth outputs, and building off examples in PyTorch in his tutorial series at https://github.com/drewbyron/pytorch-tutorials I built the framework necessary for training a Faster R-CNN object detection model on the simulated data. In order to accelerate the work, a GPU instance was used on Amazon's AWS for the training of the model. This repo details what was done in order to get the simulation and modeling pipeline to a functional state, and this README serves as the instruction manual for how to generate your own spec files to train a Faster R-CNN model for event reconstruction in CRES events, as well as how to run it on an AWS instance if you so choose.

### he6_cres_deep_learning

This folder contains modules and scripts for transforming track information data into simulated spectrogram images as seen by the He6-CRES experiment. All done by Drew Byron in his original repo.

### fasterRCNN

This folder contains the work for this project. Three notebooks are present, the two `towards_faster_rcnn` and `towards_modeling` are where the spec simulation and modeling pipelines were figured out, respectively. `modeling_interpretation` is where the first Faster R-CNN model trained on CRES data is evaluated. The script `fasterRCNN_ds.py` is what allows one to generate spectrogram images with ground-truth boundary boxes for use in training a Faster R-CNN model. `modeling.py` is the script that will allow for the training of the model of choice. 

--------------------------------------------------------------------------------
# Background

The [He6-CRES](http://faculty.washington.edu/agarcia3/Chirality-flipping/He6-CRES/) experiment at UW is the second experiment ever to look at beta decay (proton/neutron spits out electron/positron and a neutrino and becomes neutron/proton) with the Cyclotron Radiation Emission Spectroscopy technique. The goal is to measure the shape of the beta spectrum (probability distribution for the energy of the electron/positron) to high enough precision to put new limits on the existence of physics not predicted by the Standard Model. The data is comprised of many spectrogram images which are (hopefully) full of events that look like this:

<p style="text-align: center"><img width="75%" src="readme_imgs/spectrogram_example.png"/></p>
<p style="text-align: center">Taken from Heather Harrington's General Exam</p>

What is being shown is the power captured by radiation emitted by an electron that is trapped inside of a superconducting magnet. The energy of the electron is inversely proportional to the frequency at which this radiation is emitted. Thus, by determining the start frequency of the event we can determine the energy of the electron *at the moment it was created*. This will allow for unprecedented accuracy in determining the energy at which the electron was created.
<br> 
The goal of this project is to see if an object detection model like Faster R-CNN is capable of identifying events of this type - broken track segments and all - and is worth pursuing to use in the analysis of CRES events. It will also lay the groundwork for applying more advanced instance segmentation models such as Mask R-CNN.

--------------------------------------------------------------------------------

# Methodology

I began by writing a script to generate track-level information of events. This is not a physical simulation, but the produced tracks replicate the look of real events well enough for the purpose of this project. The output also needed to include boundary boxes that will act as the ground-truth targets for training the model. 

<p style="text-align: center"><img width="75%" src="readme_imgs/sim_tracks.png"/></p>
<p style="text-align: center">Tracks with boundary boxes overlaid</p>

Then the track DataFrame was passed to Drew's simulation to be transformed into spectrogram files. In short this simulation transforms the tracks into images where the pixel width is a time-slice (5126 slices for the .035s long files used in this project), pixel height is a frequency bin (4096 frequency bins), and color is the power present in that pixel represented as an 8 bit number. 

<p style="text-align: center"><img width="75%" src="readme_imgs/sim_spec.png"/></p>
<p style="text-align: center">Example of simulated spectrogram, max-pool factor of 8 applied for visibility</p>

These spectrograms can be simulated in large number with the `fasterRCNN_ds.py` script, along with a 'labels' file with the ground-truth boundary boxes. Instructions for how to do this will be included later. These will be the input to the `modeling.py` script which houses the PyTorch-Lightning pipeline to train a Faster R-CNN model. For this work I used the built-in [ResNet50-FPN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn) backbone model with pre-trained weights from the [COCO](https://cocodataset.org/#home) challenge for fine-tuning and faster convergence. The workflow of this process looks like this:
* Dataset class loads the data and transforms it to form expected by the built-in Faster R-CNN model
* LightningDataModule class calls the Dataset and handles the train/val/test splits, batch collation, and shuffling of data
* LightningModule class configures the model and handles the forward/training/validation steps
* Training object takes in the latter two for training of the model and logging of parameters of choice as it goes
Once this pipeline was confirmed to be working the model was trained on an AWS GPU instance. The Adam optimizer was used with a learning rate of $10^{-4}$, 100 epochs to train. 1000 total images were used, 60% for training, 30% for validation, 10% for testing. A max-pool factor of 16 was applied before training.

--------------------------------------------------------------------------------

# Results

The model converged and produced some very successful results:

<p style="text-align: center"><img width="90%" src="readme_imgs/good_preds.png"/></p>
<p style="text-align: center">(Left) Spectrogram from the testing set. (Right) Spectrogram with target (red) and prediction (green) boundary boxes overlaid.</p>

It is clear that for distinct events the model is capable of performing very well, though this is not the case when events are close in frequency:

<p style="text-align: center"><img width="90%" src="readme_imgs/bad_preds.png"/></p>

However, even if this is an inescapable feature of these kinds of models on this data, this type of overlap can be avoided because **event rate is a tunable parameter in the experiment**. So, at a glance the model performing quite well. In order to quantify this I attempted to measure the Mean Average Precision (mAP) of the predictions on the test set, as all of the literature on object detection models agree that this is best metric for evaluating performance. Unfortuantely, I could not get the full range of the precision vs. recall curve necessary to make this calculation. Instead I looked at the Intersection Over Union metric (IoU). This measures the ratio of the area of the intersection predicted and target boxes to the total area comprised by the boxes. I also measured the number of total predictions to the number of targets to get an idea (but not an exact measure) of the accuracy of the model.

<p style="text-align: center"><img width="90%" src="readme_imgs/IoU.png"/></p>

Overall the results are very promising. Increasing the score cut on the output boundary boxes led to better average IoU and closer number of predictions to targets.

--------------------------------------------------------------------------------

# Conclusion

This project was the first example of applying an object detection deep learning model to (simulated) CRES data. With no hyperparameter tuning and a limited size dataset the model converged and provided some very good results. A mean IoU of ~.8 was achieved, with well over 90% of targets being matched to a prediciton. This work demonstrates that these types of deep learning frameworks are at the very least worth pursuing for reconstructing events in CRES data. Further, with few changes to the labels output in the simulation script and the PyTorch classes, a Mask R-CNN pipeline could be created and tested for possible further improvement. 

--------------------------------------------------------------------------------

### Instructions for building a training dataset: 

#### Overview: 

Below we will review how to use this package to create three different training datasets. Each dataset will consist of 50 35ms spec files along with label files. The first dataset contains simple high snr tracks with no snr fluctuations (two class; track, background). The second dataset contains tracks with severe snr fluctuations (two class; track, background). The third dataset contains tracks with sidebands (three class; main_band, sideband, background). The point of this study is to illustrate that this approach could work for us and potentially help us with the most difficult problems to solve in our analysis; snr fluctuations and sidebands. We would need both more realistic training data and much more training data to train and deploy a data-ready model. But the fact that we are seeing reasonable performance with 50 spec files is indicative that this is a promising avenue. Also, the fact that the max pooling doesn't significantly disrupt detection efficiencies we may be able to apply this to our raw data and save a factor of 64 or more in data written to disk.

#### Instructions: 


* **Build a Dataset**
	* `cd he6-cres-deep-learning`
	* `python3 ./build_simple_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 50 -n_events 4 -len .035 -seed 24436 -sanity_check False`
	* For the other datasets replace the `.py` file with the appropriate module. 