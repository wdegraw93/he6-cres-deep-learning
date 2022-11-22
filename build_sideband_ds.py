#!/usr/bin/env python3

# Imports.
import numpy as np
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, fixed
import seaborn as sns
import sys
import yaml
from time import process_time
import argparse
from pathlib import Path
import shutil
import scipy.special as ss

# Additional settings.
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

from he6_cres_deep_learning.daq import DAQ, Config


def main():
    """
    A script for running an experiment based on a dictionary input (in
    the form of a json file). See
    `/he6-cres-spec-sims/config_files/rocks_exp_config_example.json` for
    an example of what it needs to contain.

    Args:
        local_dir (str): where to put experiment results. Ideally on a
            harddrive.
        sim_exp_name (str): name of the experiment (identical to name of
            .json that defines experiment).
    """

    # Parse command line arguments.
    par = argparse.ArgumentParser()
    arg = par.add_argument
    arg(
        "-c",
        "--config_path",
        type=str,
        help="Path to the base daq config file to copy.",
    )
    arg(
        "-gn",
        "--gain_noise_path",
        type=str,
        help="Path to the gain_noise file to copy for setting the gain and noise mean of the data file.",
    )
    arg(
        "-n_files",
        "--n_files",
        type=int,
        help="number of files to build in dataset. ",
    )
    arg(
        "-n_events",
        "--n_events_per_file",
        type=int,
        help="average number of events per file.",
    )
    arg(
    "-len",
    "--spec_length",
    type=float,
    help="length of spec file in seconds.",
    )
    arg(
        "-seed",
        "--random_seed",
        type=int,
        default=123456,
        help="random seed used to generate the spec files.",
    )

    arg(
        "-sanity_check",
        "--sanity_check",
        type=bool,
        default=False,
        help="if true plots of the first file created will be presented.",
    )

    args = par.parse_args()

    build_snr_oscillation_training_ds(
        args.config_path,
        args.gain_noise_path,
        args.n_files,
        args.n_events_per_file,
        args.spec_length,
        args.random_seed,
        args.sanity_check,
    )

    return None


def build_snr_oscillation_training_ds(
    config_path, gain_noise_path, n_files, n_events_per_file, spec_length random_seed, sanity_check
):

    print(f"\n\n\n Building snr oscillation dataset.\n\n\n")

    # ---- Copy base config ----
    name = "snr"
    config_path = Path(config_path)
    config_path_snr = config_path.with_name(
        config_path.stem + f"_{name}" + config_path.suffix
    )
    shutil.copyfile(str(config_path), str(config_path_snr))

    # ---- Copy then alter base noise_gain file to make it simpler. ----

    # Step 0: make a copy of the gain noise file.
    gain_noise_path = Path(gain_noise_path)
    gain_noise_path_snr = gain_noise_path.with_name(
        gain_noise_path.stem + f"_{name}" + gain_noise_path.suffix
    )
    shutil.copyfile(str(gain_noise_path), str(gain_noise_path_snr))

    # Step 1: alter the gain_noise file.
    congifure_gain_noise_csv_snr(gain_noise_path_snr)

    # ---- Build spec files ----
    config = Config(config_path_snr)

    # Change default settings of config to match input args.
    config.daq.gain_noise_csv_path = gain_noise_path_snr
    config.daq.spec_length = spec_length
    config.daq.random_seed = random_seed


    # Extract necessary parameters from config.
    spec_length = config.daq.spec_length
    freq_bw = config.daq.freq_bw

    # Build the track set to be simulated.
    tracks = build_sideband_track_set(
        n_files, n_events_per_file, spec_length, freq_bw, random_seed
    )

    # Build the simulated spec files.
    daq = DAQ(config)
    daq.run(tracks)

    # ---- Visuzlize first spec file ----
    file_in_acq = 0
    spec_path = daq.spec_file_paths[file_in_acq]
    spec_array = daq.spec_to_array(spec_path, slices=-1)

    print(f"sc: {sanity_check}")
    if sanity_check:
        plot_sparse_spec(spec_array, spec_length, freq_bw)
        plot_tracks(tracks, file_in_acq, freq_bw)
        plot_noise_gain(config.daq.gain_noise_csv_path)

    print(f"\n\n\n Done building simple dataset.")

    return None


def build_sideband_track_set(n_files, n_events_per_file, spec_length, freq_bw, seed):

    rng = np.random.default_rng(seed)

    n_events = n_files * n_events_per_file

    file_in_acq = rng.integers(low=0, high=n_files, size=n_events)
    event_num = np.arange(0, n_events, 1)
    time_start = rng.uniform(low=0, high=spec_length, size=n_events)
    time_stop = np.array([spec_length] * n_events)

    freq_start = rng.uniform(low=100e6, high=freq_bw, size=n_events)
    slope = rng.normal(loc=1e11, scale=1e10, size=n_events)
    freq_stop = freq_start + slope * (time_stop - time_start)

    band_power_start = rng.normal(loc=20e-15, scale=3e-15, size=n_events)
    band_power_stop = band_power_start
    band_num = np.array([0] * n_events)

    h = rng.uniform(low=0, high=1, size=n_events)
    axial_freq = rng.uniform(low=60e6, high=100e6, size=n_events)

    segments = pd.DataFrame(
        {
            "file_in_acq": file_in_acq,
            "event_num": event_num,
            "time_start": time_start,
            "time_stop": time_stop,
            "freq_start": freq_start,
            "freq_stop": freq_stop,
            "slope": slope,
            "band_power_start": band_power_start,
            "band_power_stop": band_power_stop,
            "band_num": band_num,
            "h": h,
            "axial_freq": axial_freq,
        }
    )

    tracks = process_segments(segments)
    print(sideband_calc(18e9, 100e6, h=1, num_sidebands=1))

    return tracks


def plot_noise_gain(gain_noise_csv_path):
    pd.read_csv(gain_noise_csv_path).set_index("freq").plot.line()
    plt.show()
    return None


def plot_sparse_spec(spec_array, spec_length, freq_bw, snr_cut=5):

    cut_condition = np.array(
        (spec_array > spec_array.mean(axis=0) * snr_cut).T, dtype=float
    )
    extent = [0, spec_length, 0, freq_bw]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(
        1 - cut_condition,
        origin="lower",
        aspect="auto",
        interpolation=None,
        cmap="gray",
        extent=extent,
    )

    ax.set_title("Sparse Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (Hz)")
    plt.show()

    return None


def plot_tracks(tracks, file_in_acq, freq_bw):

    condition = tracks.file_in_acq == file_in_acq

    fig, ax = plt.subplots(figsize=(12, 8))

    for index, row in tracks[condition].iterrows():

        time_coor = np.array([row["time_start"], row["time_stop"]])
        freq_coor = np.array([row["freq_start"], row["freq_stop"]])
        if np.abs(row.band_num) == 1:
            ax.plot(
                time_coor,
                freq_coor,
                "ro-",
                markersize=0.5,
                alpha=0.5,
            )
        if np.abs(row.band_num) == 0:
            ax.plot(
                time_coor,
                freq_coor,
                "yo-",
                markersize=0.5,
                alpha=0.5,
            )
    ax.set_ylim(0, freq_bw)
    ax.set_title("tracks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (Hz)")
    plt.show()

    return None


def process_segments(segments):

    sideband_num = 1
    # Build the segments into tracks.
    band_list = []
    for segment_index, row in segments.iterrows():

        # sideband_calc(avg_cycl_freq, axial_freq, h, num_sidebands=sideband_num)
        sideband_amplitudes = sideband_calc(
            row.freq_start, row.axial_freq, row.h, num_sidebands=sideband_num
        )[0]
        print(sideband_amplitudes)

        for i, band_num in enumerate(range(-sideband_num, sideband_num + 1)):

            # copy segment in order to fill in band specific values
            row_copy = row.copy()

            # fill in new avg_cycl_freq, band_power, band_num
            row_copy["freq_start"] = sideband_amplitudes[i][0]
            row_copy["freq_stop"] = row_copy["freq_start"] + row_copy["slope"] * (
                row_copy["time_stop"] - row_copy["time_start"]
            )
            # Note that the sideband amplitudes need to be squared to give power.
            row_copy["band_power_start"] = (
                sideband_amplitudes[i][1] ** 2 * row.band_power_start
            )
            row_copy["band_power_stop"] = row_copy["band_power_start"]
            row_copy["band_num"] = band_num

            print(row_copy)
            # append to band_list, as it's better to grow a list than a df
            band_list.append(row_copy.tolist())

    bands_df = pd.DataFrame(band_list, columns=segments.columns)

    return bands_df


def congifure_gain_noise_csv_snr(csv_path):

    # Sinusoidal gain:
    col = "gain"

    array = np.array([1.0] * 4096)
    update_gain_noise_csv(csv_path, col, array)

    # Flat noise:
    col = "noise_mean"

    array = np.array([1.0] * 4096)

    update_gain_noise_csv(csv_path, col, array)


def update_gain_noise_csv(csv_path, col, array):
    """
    Helper function for editing gain_noise.csv.
    """
    noise_gain_df = pd.read_csv(csv_path)
    noise_gain_df[col] = array
    noise_gain_df.to_csv(csv_path, index=False)

    return None


# ---- Functions for sideband creation ----


def format_sideband_array(
    sidebands_one, avg_cyc_freq, axial_freq, mod_index=np.nan, num_sidebands=1
):
    """Does formatting for array with list of sideband magnitudes (normalized), and their start frequencies.
    Takes in 1-sided list of sideband magnitudes
    """
    # Calculate (2-sided) list of (frequency, amplitude) of sidebands
    sidebands = []

    for k in range(-num_sidebands, num_sidebands + 1):
        freq = avg_cyc_freq + k * axial_freq
        magnitude = sidebands_one[abs(k)]
        pair = (freq, magnitude)
        sidebands.append(pair)

    ### Intentionally returns modulation index of nan as it is only (meaningfully) defined for harmonic traps
    return sidebands, mod_index


def sideband_calc(avg_cycl_freq, axial_freq, h, num_sidebands=1):

    """Calculates relative magnitudes of num_sidebands sidebands from
    average cyclotron frequency (avg_cycl_freq), axial frequency
    (axial_freq), and maximum axial amplitude (zmax).
    """

    sidebands = [abs(ss.jv(k, h)) for k in range(num_sidebands + 1)]
    return format_sideband_array(sidebands, avg_cycl_freq, axial_freq, h, num_sidebands)


if __name__ == "__main__":
    main()
