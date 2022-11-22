
<p align="center"><img width="30%" src="/demo/readme_imgs/he6-cres_logo.png" /></p>

--------------------------------------------------------------------------------
# he6_cres_deep_learning

This repo contains modules and scripts for development of a deep learning event detection pipeline for the He6-CRES experiment at the University of Washington. 

--------------------------------------------------------------------------------
### Run an analysis then make interactive plots of cres track features!

#### Look at track and event classification overlaid on top of raw data: 

<p align="center"><img width="42%" src="/demo/readme_imgs/sparse_spec.png" />              <img width="42%" src="/demo/readme_imgs/track_overlay.png" /><img width="80%" src="/demo/readme_imgs/event_overlay.png" /></p>


--------------------------------------------------------------------------------
### Instructions as I build things out: 

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
			* `python3 ./build_snr_oscillation_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 1 -n_events 10 -seed 34234 -sanity_check True`
			* Omit the sanity_check flag to no longer make plots to verify things are working. 
	* Here we've built a simple set of spec files with corresponding labels. 

* **Step 3**: 
	* `python3 ./build_sideband_ds.py -c "/media/drew/T7 Shield/cres_deep_learning/training_data/config/base_daq_config.yaml" -gn "/media/drew/T7 Shield/cres_deep_learning/training_data/gain_noise/base_gain_noise.csv" -n_files 1 -n_events 2 -seed 34234 -sanity_check True`


--------------------------------------------------------------------------------
### TODOS: 

* Want to illustrate the model's performance on all 3 cases. Very simple dataset, snr fluctuations, sidebands. First work on making scripts for building these three datasets. Then work on the modelling and iterate. 



--------------------------------------------------------------------------------
### Notes as I build this out: 

* Why are the tracks not where I expect them to be? It seems like maybe I didn't merge the daq branch in???
	* Ok it's a rendering issue I think. My plotting is all weird.  
	* Two issues: One is that the tracks I'm making are lasting for 35 ms not 6. The second is that the extent seems to be limiting the freq bins to 1200... 
	* Ok I resolved both of these issues. Things are looking good now. 
