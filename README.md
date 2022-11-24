
<p align="center"><img width="30%" src="/demo/readme_imgs/he6-cres_logo.png" /></p>

--------------------------------------------------------------------------------
# he6_cres_deep_learning

This repo contains modules and scripts for development of a deep learning event detection pipeline for the He6-CRES experiment at the University of Washington. 

--------------------------------------------------------------------------------
### Train a model to detect cres events.

#### Look at track and event classification overlaid on top of raw data: 


<p align="center"><img width="42%" src="/demo/readme_imgs/sparse_spec.png" />              <img width="42%" src="/demo/readme_imgs/track_overlay.png" /><img width="80%" src="/demo/readme_imgs/event_overlay.png" /></p>


--------------------------------------------------------------------------------
### Instructions for how to build a training dataset: 

#### Overview: 

Below we will review how to use this package to create three different training datasets. Each dataset will consist of 50 35ms spec files along with label files. The first dataset contains simple high snr tracks with no snr fluctuations (two class; track, background). The second dataset contains tracks with severe snr fluctuations (two class; track, background). The third dataset contains tracks with sidebands (three class; main_band, sideband, background). The point of this study is to illustrate that this approach could work for us and potentially help us with the most difficult problems to solve in our analysis; snr fluctuations and sidebands. We would need both more realistic training data and much more training data to train and deploy a data-ready model. But the fact that we are seeing reasonable performance with 50 spec files is indicative that this is a promising avenue. Also, the fact that the max pooling doesn't significantly disrupt detection efficiencies we may be able to apply this to our raw data and save a factor of 64 or more in data written to disk.

#### Instructions: 


* **Build a Dataset**
	* `cd he6-cres-deep-learning`
	* `python3 ./build_simple_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 50 -n_events 4 -len .035 -seed 24436 -sanity_check False`
	* For the other datasets replace the `.py` file with the appropriate module. 

--------------------------------------------------------------------------------
### Instructions for how to train a UNET model on one of the above datasets

### Locally

* **Train UNET on Dataset**
	* When running locally it is easiest to train with a script. The below is an example of how to do this. 
	* `cd he6-cres-deep-learning`
	* `python3 demo/train_simple_ds.py`
	* For the other datasets replace the `.py` file with the appropriate module.
	* One done training use the `demo/model_predictions.ipynb` to investigate the restuls and save figures. (NEED TO PUT THIS IN REPO)
	* To open the tensorboard metric tracking run `tensorboard --logdir tb_logs/` wherever you've sent your checkpoints to be written to (specified in `train_simple_ds.py`).

### Google colab

* For now I am training the model either locally or using the pro version of google colab which gives you access to GPUs in a jupyter-lab type enviornment for training models such as this. The link to the colab notebook I used to train UNET on the above datasets is [here](https://colab.research.google.com/drive/112u4WldrYLWI_7iY7o0MWpBdhCMuhelc?usp=sharing). All you should need to do is copy the nb to your own drive and then follow the instructions in the notebook. Loading the data to the remote server is time consuming so this is not an ideal situation going forward. It's also worth noting that  I've also put a copy of the colab ipynb in the repo here: `/he6-cres-deep-learning/demo/cres_dl_train_unet_colab.ipynb`.


### Hyak GPU cluster

* Ideally we should get this training on HYAK soon, the GPU cluster here at UW. This should enable much faster iteration and training. 

--------------------------------------------------------------------------------
### Deep Learning: 

* Building out the repo. Then going to add everything in so it's easy to use and the ipynb is only a handful of cells long. 





--------------------------------------------------------------------------------
### TODO LIST: 

* Want to illustrate the model's performance on all 3 cases. Very simple dataset, snr fluctuations, sidebands. First work on making scripts for building these three datasets. Then work on the modelling and iterate. 
* Make a plan for what we want to have for each dataset, start making moves towards this part of thesis being done ASAP. make the plots then start writing. 

--------------------------------------------------------------------------------

### Done List: 

* Put a link to the colab nb in the readme. 
* If the maxpooling was done in the loading process this would make the loading much less intense on the ram. I should make this work before pushing on. DONE 
* Get jaccard index (IOU) tracked by the model.
* Ok the sideband dataset isn't working well because of the very weak tracks that I still have training data for (I think). I'm going to limit the range of h. 
* Maybe just use the local version to make cool plots of the performance? Need to move it out of this directory. 
* Getting somewhere but the imports aren't working. Get it working then clean up everything from the bottom.
--------------------------------------------------------------------------------

# For tomorrow: 

* Tonight run both of the datasets I need. 
* For tomorrow train on these (maybe one local one on colab?)
* Try increasing num_workers in dataloader?
* Create cool THESIS-worthy plots and add them to the readme. 

* **After breakfast**: 
	* Look at results from the simple ds and make plots about the results. 
	

* **By EOD**: 
	* Have a instructions and preliminary plots for each of the 4 datasets with 10 files. The idea being that all but the number of files will need to be increased. 




--------------------------------------------------------------------------------

### Notes as I build this out: 


* Note that the requant gain is what's causing my noise mean to be 8 here instead of 1. And it isn't exacly 8 due to the rounding. 

--------------------------------------------------------------------------------

## Notes on the different tests: 


**Simple DS**: 

	* This dataset is curropt. Need to rebuild it. Start back with 10 files. Then make it work in the simplest of cases. Make thesis worthy plots of it, then move on to SNR fluctuations. 

**Sideband ds**: 
	* Sidebands isn't working with the base model and 50 epochs. It guesses everything as class 2. Hmm. Maybe the dataset is too confusing? Try with a slightly deeper UNET and a simpler dataset. Start with the simplest case first. 
	* Also maybe the learning rate is too high on this run, try lowering it. 

--------------------------------------------------------------------------------

# Start here 11/24/22:

## Specific steps I'm following in training:

* **First Pass** Make things work to first order (overtraining at least) with 10 files. 
	* **Datasets**: 
		* simple: `python3 ./build_simple_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 10 -n_events 4 -len .035 -seed 24436 -sanity_check False`
		* **DONE** snr: `python3 ./build_snr_oscillation_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 10 -n_events 4 -len .035 -seed 24436 -sanity_check False`
		* sidebands: `python3 ./build_sideband_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 10 -n_events 1 -len .035 -seed 24436 -sanity_check False`
	* **Training**: 
		* simple: `python3 demo/train_simple_ds.py`
		* snr: `python3 demo/train_snr_oscillation_ds.py`
		* sidebands: `python3 demo/train_sidebands_ds.py`

	* **Investigation**
		* Make plots of all of the above, thesis worthy!