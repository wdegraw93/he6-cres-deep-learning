#### Contains all the functionality needed to simulate CRES events with scattering
#### for the purpose of training a FasterRCNN model. Produces spec files and target dicts
#### to be sent to CRES_DS

# Imports
import numpy as np
import pathlib
import pandas as pd
import re
import argparse
import yaml
import shutil
import json

# Sim specific imports - may need to change on AWS
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))
from he6_cres_deep_learning.daq import DAQ, Config

def main():
    # Parse command line arguments.
    par = argparse.ArgumentParser()
    arg = par.add_argument
    arg(
        "-d",
        "--dir_name",
        type=str,
        default="fasterRCNN_test",
        help='Name of directory in which to put spec and label files'
    )
    arg(
        "-c",
        "--config_path",
        type=str,
        default='/config/base_daq_config.yaml',
        help="Path to the base daq config file to copy.",
    )
    arg(
        "-gn",
        "--gain_noise_path",
        type=str,
        default='/gain_noise/base_gain_noise.csv',
        help="Path to the gain_noise file to copy for setting the gain and noise mean of the data file.",
    )
    arg(
        "-n_files",
        "--n_files",
        type=int,
        default=1,
        help="number of files to build in dataset. ",
    )
    arg(
        "-n_events",
        "--n_events_per_file",
        type=int,
        default=1,
        help="average number of events per file.",
    )
    arg(
        "-len",
        "--spec_length",
        type=float,
        default=.035,
        help="length of spec file in seconds.",
    )
    arg(
        "-seed",
        "--random_seed",
        type=int,
        default=1234,
        help="random seed used to generate the spec files.",
    )
    arg(
        "-sanity_check",
        "--sanity_check",
        type=bool,
        default=False,
        help="if true plots of the first file created will be presented.",
    )
    arg(
        '-slope_mean',
        '--slope_mean',
        type=float,
        default=2e8,
        help='Mean slope value for track slope distribution.'
    )
    arg(
        '-slope_std',
        '--slope_std',
        type=float,
        default=1e7,
        help='Standard deviation for track slope distribution.'
    )

    # put command line args into object
    args = par.parse_args()

    # Generate spec files and targets
    fasterrcnn_spec(dir_name = args.dir_name,
                    config_path = sys.path[0]+'/'+args.config_path, 
                    gain_noise_path = sys.path[0]+'/'+args.gain_noise_path, 
                    n_files = args.n_files, 
                    n_events_per_file = args.n_events_per_file,
                    spec_length = args.spec_length,
                    slope_mean = args.slope_std,
                    slope_std = args.random_seed,
                    random_seed = args.random_seed)

    return None



def fasterrcnn_spec(dir_name,
                    config_path, 
                    gain_noise_path, 
                    n_files, 
                    n_events_per_file,
                    spec_length, 
                    slope_mean,
                    slope_std,
                    random_seed):
    
    # ---- Copy and rename base config ----
    name = dir_name
    config_path = Path(config_path)
    config_path_rcnn = config_path.with_name(name + config_path.suffix)
    shutil.copyfile(str(config_path), str(config_path_rcnn))

    # ---- Copy then alter base noise_gain file to make it simpler. ----
    # Step 0: make a copy of the gain noise file.
    gain_noise_path = Path(gain_noise_path)
    gain_noise_path_rcnn = gain_noise_path.with_name(
                gain_noise_path.stem + f"_{name}" + gain_noise_path.suffix)
    shutil.copyfile(str(gain_noise_path), str(gain_noise_path_rcnn))
    
    # Step 1: alter the gain_noise file.
    congifure_gain_noise_csv(gain_noise_path_rcnn)

    # ---- Build spec files ----
    config = Config(config_path_rcnn)
    
    # Change default settings of config to match input args.
    config.daq.gain_noise_csv_path = gain_noise_path_rcnn
    config.daq.random_seed = random_seed
    config.daq.spec_length = spec_length

    # Build the track set to be simulated.
    tracks, target_dict = fasterrcnn_df(n_files=n_files,
                                        n_events_per_file=n_events_per_file,
                                        spec_length=spec_length,
                                        slope_mean=slope_mean,
                                        random_seed=random_seed)
    print(tracks)
    # Build the simulated spec files.
    daq = DAQ(config)
    daq.run(tracks)
    
    # Now that spec files are written, convert bboxes to pixel height, width
    delta_f = config.daq.freq_bw / config.daq.freq_bins #height conversion
    slices_in_spec = config.daq.spec_length * delta_f / config.daq.roach_avg #width conversion
    
    for bbox_dict in target_dict.values():
        for bbox in bbox_dict.values():
            bbox[0] *= slices_in_spec/spec_length
            bbox[2] *= slices_in_spec/spec_length
            bbox[1] /= delta_f
            bbox[3] /= delta_f
    
    # write to disk
    labels_dir = Path(config_path).parent.joinpath(name, 'label_files')
    write_label_files(target_dict, labels_dir, config.daq.spec_prefix)
    
    return None
    
    
    
def fasterrcnn_df(n_files, n_events_per_file, spec_length=.035, freq_bw=1200e6, slope_mean=2e8, slope_std=1e7, random_seed=1234):
    '''
    Define the DataFrame describing the simulation of various events over several spec files. Entries are track-level description of
    the electron's motion in the CRES apparatus. Also define the boundary boxes that will serve as the target for the FasterRCNN model
    Returns the DataFrame of track information and the dictionary of the bounding boxes for each file simulated
    '''
    track_set = {
            "file_in_acq": [],
            "event_num": [],
            "time_start": [],
            "time_stop": [],
            "freq_start": [],
            "freq_stop": [],
            "slope": [],
            "band_power_start": [],
            "band_power_stop": [],
            "band_num": [],
        }
    
    # Dictionary to store event boundary boxes for each file
    event_bboxes = {
        0: # key will be file number
            {0: # value will be a dict, where key is event number (this will grow through sim)
                 []} # value of the inner dict will be the boundary box coordinates 
                     # in format [start_time, start_freq, stop_time, stop_freq]
    }
    
    # Define Generator object for pulling from distributions
    rng = np.random.default_rng(random_seed)
    
    # n_events in a spec file can be reasonably estimated from a Poisson distribution
    n_events = rng.poisson(lam=3, size=n_files) 
    
    # Loop over files
    for file, events in enumerate(n_events): 
        # check if file in keys for bboxes, if not then initialize dict
        if file not in event_bboxes.keys():
            event_bboxes[file] = {0: []}

        # Pull from uniform distribution to determine start time of event
        event_start_times = rng.uniform(low=0, high=spec_length, size=events)
        
#-------------------------------------------------------------------------------------------------------------------------
        # Get start frequencies
        # TODO: implement beta spectrum sampling for start frequencies
        event_start_frequencies = rng.uniform(low=150e6, high=freq_bw, size=events)
#-------------------------------------------------------------------------------------------------------------------------

        # Loop over each event in file
        for event, (start_time, start_freq) in enumerate(zip(event_start_times, event_start_frequencies)):
            # check if event in keys for bboxes, if not then initialize list
            if event not in event_bboxes[file].keys():
                event_bboxes[file][event] = []
                
#-------------------------------------------------------------------------------------------------------------------------
            # Define number of scatters per event
            # TODO: Implement a physical model for scattering
            n_tracks = rng.integers(low=1, high = 10)
#-------------------------------------------------------------------------------------------------------------------------

            # Loop over each track in event
            for track in range(n_tracks):
                # Defining break parameter if need to leave loop because we hit edge of spec file
                _break = False
                
                # Pull track length from exponential distribution with tau=10ms
                track_len = rng.exponential(scale=.01)
                
                # Check to see if track outside of time window
                if start_time+track_len > spec_length:
                    stop_time = spec_length
                    track_len = spec_length-start_time
                    _break = True
                else:
                    stop_time = start_time+track_len
    
#-------------------------------------------------------------------------------------------------------------------------
                # Slope of track is assumed to be normally distributed around the mean of 2e8Hz in Kr83 events
                # TODO: Implement empirical model for track slopes based on energy of the electron
                slope = rng.normal(loc=slope_mean, scale=slope_std)
#-------------------------------------------------------------------------------------------------------------------------
                
                # Check if track outside of bandwidth
                if start_freq + slope*track_len > freq_bw:
                    stop_freq = freq_bw
                    _break = True
                else:
                    stop_freq = start_freq+slope*track_len

                # Store parameters in dict
                track_set["file_in_acq"].append(file)
                track_set["event_num"].append(event)
                track_set["time_start"].append(start_time)
                track_set["time_stop"].append(stop_time)
                track_set["slope"].append(slope)
                track_set["freq_start"].append(start_freq)
                track_set["freq_stop"].append(stop_freq)
                
                # If first track in event, add start_time, start_freq to bboxes dict
                if track==0:
                    event_bboxes[file][event].append(start_time)
                    event_bboxes[file][event].append(start_freq)
                
                # If hit edge then break out of track loop for this event, append bbox values
                if _break:
                    event_bboxes[file][event].append(stop_time)
                    event_bboxes[file][event].append(stop_freq)
                    break
                # Else update start time and freq. 
                # Frequency jump will be pulled from normal dist mean 10MHZ std 2MHz
                
                # If reached number of tracks in event then append bbox values
                elif track==n_tracks-1:
                    event_bboxes[file][event].append(stop_time)
                    event_bboxes[file][event].append(stop_freq)
                
#-------------------------------------------------------------------------------------------------------------------------
                # Else update start time and freq. 
                # Frequency jump will be pulled from normal dist mean 10MHZ std 2MHz
                # TODO: Frequency jumps will also have to be pulled from physical scattering model
                else:
                    start_time = stop_time
                    start_freq = stop_freq + rng.normal(loc=10e6, scale=2e6)
#-------------------------------------------------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------------------------------------------------   
    # TODO: implement proper power information
    track_set["band_power_start"] = [1e-14]*len(track_set["slope"])
    track_set["band_power_stop"] = [1e-14]*len(track_set["slope"])
    track_set["band_num"] = np.zeros(len(track_set["slope"]))
#-------------------------------------------------------------------------------------------------------------------------
    
     # convert track_set dict to DataFrame
    track_set = pd.DataFrame(track_set)
    
    return track_set, event_bboxes



def congifure_gain_noise_csv(csv_path):
    """
    Note that if you don't change the noise here you will end up with the default noise
    floor of the apparatus (I-side).
    """
    # Sinusoidal gain:
    col = "gain"

    array = np.array([1.0] * 4096)
    update_gain_noise_csv(csv_path, col, array)

    # Flat noise:
    col = "noise_mean"

    array = np.array([1.0] * 4096)

    update_gain_noise_csv(csv_path, col, array)
    return None
    
    
def update_gain_noise_csv(csv_path, col, array):
    """
    Helper function for editing gain_noise.csv.
    """
    noise_gain_df = pd.read_csv(csv_path)
    noise_gain_df[col] = array
    noise_gain_df.to_csv(csv_path, index=False)

    return None



def write_label_files(target_dict, path_to_target_dir, spec_prefix):
    if not path_to_target_dir.is_dir():
        path_to_target_dir.mkdir()
        print("created directory : ", path_to_target_dir)
    with open(f'{path_to_target_dir}/{spec_prefix}_labels.json', 'w') as file:
        json.dump(target_dict, file)
    return None
        
        
if __name__ == "__main__":
    main()