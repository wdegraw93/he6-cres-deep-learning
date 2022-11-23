
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

* Below we will review how to use this package to create three different training datasets. Each dataset will consist of 30 30ms spec files along with label files. This is to illustrate that this approach could work for us. We would need much more training data to train and deploy a data-ready model. But the fact that we are seeing reasonable performance with 30 
* **Step 0**: Building a simple example training data set.
	* *Description of dataset:*
		* All bands are labeled band 0. Gain is flat, noise is not flat. (IS GAIN ACTUALLY FLAT? FIX THIS) 
	* *Instructions*
		* Start by copying both the `config` and `gain_noise` directories in `he6-cres-deep-learning` into a directory suitable for writing training data to disk (ideally a harddrive with lots of space). 
		* Then run the following (will need to change paths): 
			* `cd he6-cres-spec-sims`
			* `python3 ./build_simple_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 1 -n_events 4 -seed 34234 -sanity_check True
`
			* Omit the sanity_check flag to no longer make plots to verify things are working. 
	* Here we've built a simple set of spec files with corresponding labels. 
	* NOTE SHOULD THIS BUILD A NEW CONFIG???

* **Step 1**: Building a training data set for signals with severe SNR fluctuations.
	* *Description of dataset:*
		* All bands are labeled band 0. Noise floor is flat. Gain is fluctuating (20 oscillations in our bandwidth). 
	* *Instructions*
		* Start by copying both the `config` and `gain_noise` directories in `he6-cres-deep-learning` into a directory suitable for writing training data to disk (ideally a harddrive with lots of space). 
		* Then run the following (will need to change paths): 
			* `cd he6-cres-spec-sims`
			* `python3 ./build_snr_oscillation_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 1 -n_events 4 -len .035 -seed 344 -sanity_check True`
			* Omit the sanity_check flag to no longer make plots to verify things are working. 
	* Here we've built a simple set of spec files with corresponding labels. 

* **Step 3**: 
	* `python3 ./build_sideband_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 1 -n_events 2 -len .035 -seed 34234 -sanity_check True`

--------------------------------------------------------------------------------
### Deep Learning: 

* Building out the repo. Then going to add everything in so it's easy to use and the ipynb is only a handful of cells long. 

* Getting somewhere but the imports aren't working. Get it working then clean up everything from the bottom.

* If the maxpooling was done in the loading process this would make the loading much less intense on the ram. I should make this work before pushing on. 

--------------------------------------------------------------------------------
### TODOS: 

* Want to illustrate the model's performance on all 3 cases. Very simple dataset, snr fluctuations, sidebands. First work on making scripts for building these three datasets. Then work on the modelling and iterate. 



--------------------------------------------------------------------------------
### Notes as I build this out: 

* Why are the tracks not where I expect them to be? It seems like maybe I didn't merge the daq branch in???
	* Ok it's a rendering issue I think. My plotting is all weird.  
	* Two issues: One is that the tracks I'm making are lasting for 35 ms not 6. The second is that the extent seems to be limiting the freq bins to 1200... 
	* Ok I resolved both of these issues. Things are looking good now. 
