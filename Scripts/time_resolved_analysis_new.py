"""
@author: Annabelle Brauel (based on work by Alexander Niedrig)
Research Group Heilemann
Institute for Physical and Theoretical Chemistry, Goethe University Frankfurt a.M.
Extracts time course of single-particle tracking measurements and plots it
"""

import configparser
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import h5py

#import tifffile
#from tifffile import TiffFile
#import warnings
#from fileinput import filename
#from ftplib import all_errors
#import matplotlib.pyplot as plt
#import scipy.stats as scy
# from TrackAnalysis import cover_slip


class IncorrectConfigException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


def get_matching_files(directory, target, exclusion_string):
    """
    Recursively searches for files containing a target substring in their names,
    excluding any that contain substrings from the exclusion list.

    :param directory: Root directory to search
    :param target: Substring that must be present in the filename
    :param exclusion_string: List of substrings; filenames containing any of them are excluded
    :return: List of matching file paths
    """
    matching_files = []
    for path, subdirectories, files in os.walk(directory):
        for name in files:
            if target.lower() in name.lower():
                if not any([True for string in exclusion_string if string.lower() in name.lower()]):
                    matching_files.append(os.path.join(path, name))
    return matching_files


def sort_cells(cells):
    """
    Sorts a list of cell names in alphanumeric order, taking into account numeric parts.
    For example: ["Cell_1", "Cell_10", "Cell_2"] -> ["Cell_1", "Cell_2", "Cell_10"]

    :param cells: List of cell name strings
    :return: List of alphanumerically sorted cell names
    """


def insert_error(mean_frame, sd_frame, sem_frame):
    """
    Inserts standard deviation (SD) and standard error of the mean (SEM) columns
    into a dataframe containing mean values, directly after their corresponding mean column.

    :param mean_frame: Dataframe with mean values
    :param sd_frame: Dataframe with standard deviations
    :param sem_frame: Dataframe with standard errors of the mean
    :return: Modified dataframe with SD and SEM inserted
    """


def calc_mean_over_cs(binned_data, attribute):
    """
    Calculates the mean, standard deviation, and standard error of a specified attribute
    across a list of binned dataframes (e.g. from different coverslips).

    :param binned_data: List of binned dataframes
    :param attribute: Name pattern of the attribute to aggregate
    :return: Numpy array with [mean, std, sem] per row
    """


def plot_by_time(dataframes, attribute, t_lig, ligand_name, binned_data, error_type, clr):
    """
    Plots time-dependent data from multiple dataframes as individual dots and average bars with error shading.

    :param dataframes: List of raw dataframes to be plotted (dots)
    :param attribute: Attribute to be plotted:
            P: Percentage of mobile molecules
            D: Diffusion coefficient
            L: Track length
            N: Number of tracks/particles
            confinement_radius: confinement radius
    :param t_lig: Time of ligand addition (e.g. "180s")
    :param ligand_name: Name of the ligand to label the vertical line
    :param binned_data: List of binned data per coverslip for calculating statistics
    :param error_type: 'SEM' or 'SD' - determines which error to use
    :param clr: Color of the raw data dots
    :return: The generated plot
    """


def plot_by_cells(dataframes, attribute, t_lig, ligand_name, binned_data, error_type, clr):
    """
    Plots attribute values per cell number across multiple dataframes,
    with optional ligand marking and binned mean/error overlays.

    :param dataframes: List of raw dataframes to be plotted
    :param attribute: Attribute to be plotted
            P: Percentage of mobile molecules
            D: Diffusion coefficient
            L: Track length
            N: Number of tracks/particles
            confinement_radius: confinement radius
    :param t_lig: Ligand addition time (e.g. '40c')
    :param ligand_name: Name of the ligand
    :param binned_data: List of binned data per coverslip
    :param error_type: 'SEM' or 'SD' to determine error bars
    :param clr: Color used for individual data points
    :return: The generated matplotlib plot
    """


def bin_data_time(frame, bin_size):
    """
    Groups rows of a dataframe into time bins, calculates statistical
    summaries for each bin, and returns a new dataframe with added SD and SEM columns.

    :param frame: Input dataframe containing single-cell data with time information
    :param bin_size: Size of time bins (in the same units as column 1 in the dataframe)
    :return: Dataframe containing the mean, SD, and SEM of each attribute per time bin
    """


def test_by_cell(frames, bin_size, ligand, alpha, p1, p2, p3):
    """
    Runs statistical tests over bins of a certain number of cells.

    :param frames: List of dataframes to test
    :param bin_size: Number of cells per bin
    :param ligand: If True, run significance tests between bins
    :param alpha: Significance level for normality tests
    :param p1, p2, p3: thresholds for statistical significance (*, **, ***)
    :return: Dictionary of normality test results, and optionally significance test results
    """


def test_by_time(frames, bin_size, ligand, alpha, p1, p2, p3):
    """
    Runs statistical tests over time-based bins.

    :param frames: List of dataframes to test
    :param bin_size: Duration of each bin (in seconds)
    :param ligand: If True, run significance tests between bins
    :param alpha: Significance level for normality tests
    :param p1, p2, p3: thresholds for statistical significance (*, **, ***)
    :return: Dictionary of normality test results, and optionally significance test results
    """


def normality_tests(dataframe, attribute):
    """
    Run two normality tests on a selected attribute.

    :param dataframe: Dataframe with relevant data
    :param attribute: Column name of the variable to test
    :return: Statistics and p-values from Shapiro and Kolmogorov-Smirnov tests
    """


def significance_tests(reference_frame, data_frame, attribute_to_compare):
    """
    Compares two distributions using the appropriate paired test (t-test or Wilcoxon),
    depending on whether data is normally distributed.

    :param reference_frame: Reference distribution (pandas DataFrame)
    :param data_frame:  Comparison distribution (pandas DataFrame)
    :param attribute_to_compare: Column name to compare, e.g. D_confined (str)
    :return: Test statistics and p-values for paired t-test and Wilcoxon test
    """


def compile_columns(frames, columns, rename_cols):
    """
    Merges columns with matching names from multiple dataframes.

    :param frames: List of input dataframes
    :param columns: List of substrings to match in column names
    :param rename_cols: Whether to prefix columns with date to avoid duplication
    :return: A combined DataFrame of selected columns
    """


def generate_shortname(value):
    """
    Cleans up values for HDF5 writing by removing trailing colons.

    :param value: Value to clean
    :return: Sanitized string representation
    """


def cut_cell_names(name):
    """
    Extracts cell number from full identifier.

    :param name: Full name string (e.g., "sample_cell_42")
    :return: Extracted cell number (e.g., "42")
    """


def output_folder(file, folder, datasets):
    """
    Writes datasets to a group in an HDF5 file.

    :param file: Open h5 py file handle
    :param folder: Name of group (folder) to write into
    :param datasets: List of [name, DataFrame] pairs to write
    """


def rename_columns(old_col_names):
    """
    Appends units to column names based on their semantic label.

    :param old_col_names: List of original column names
    :return: List of renamed columns with units
    """


def four_set_output(save_dir, outputfile, dataset, coverslip_data, use_timestamps, plot_color, t_lig,
                    ligand_name, error_type, binned):
    """
    Outputs analysis data for four diffusion types: 'global', 'immobile', 'confined', 'free'.

    :param save_dir: directory where plots and data will be saved
    :param outputfile: HDF5 file to write the output into
    :param dataset: dataset inside the HDF5 file
    :param coverslip_data: list of pandas DataFrames with coverslip data
    :param use_timestamps: whether time-based plotting is used
    :param plot_color: color used for plots
    :param t_lig: ligand addition time (in seconds or relative time)
    :param ligand_name: name of the ligand
    :param error_type: error display type ('SD' or 'SEM')
    :param binned: whether to output data as binned time response
    """


def stack_data(dataframes):
    """
    Stacks multiple DataFrames (from different coverslips) vertically into a single DataFrame.

    :param dataframes: list of dataframes
    :return: a single-element list containing one DataFrame with all input frames stacked
    """
    # Define the expected column names in the correct order


def load_user_input(config_path):
    """
        Loads all user-defined settings and paths from the configuration file.
        Returns a dictionary containing:
            - Paths to .h5, .tif, and coverslip directories
            - Analysis settings (binning, timestamps, ligand use)
            - Plot and statistical parameters
            - Output directory structure
        """

    # --- Initialize variables ---
    cs_paths = []  # Coverslip directories
    h5_paths = []  # Global data directories
    tif_files = []  # All relevant .tif files
    cs_names = []  # Coverslip identifiers
    h5_files = []  # .h5 files
    use_timestamps = False
    ligand_exists = False
    run_stats = False

    # --- Load configuration ---
    config = configparser.ConfigParser()
    config.read(config_path)

    # --- GLOBAL_DIR section ---
    try:
        if not config["GLOBAL_DIR"]:
            raise IncorrectConfigException("No GLOBAL_DIR section found in config.")
        for key in config["GLOBAL_DIR"]:
            h5_paths.append(config["GLOBAL_DIR"][key])
    except KeyError:
        raise IncorrectConfigException("Section [GLOBAL_DIR] missing in config file.")
    # print("h5_paths:", h5_paths)

    # --- CS_DIRS section ---
    try:
        if not config["CS_DIRS"]:
            raise IncorrectConfigException("No coverslip directories defined in [CS_DIRS].")
        for key in config["CS_DIRS"]:
            cs_paths.append(config["CS_DIRS"][key])
    except KeyError:
        raise IncorrectConfigException("Section [CS_DIRS] missing in config file.")
    # print("cs paths:\n", cs_paths)

    # Collect TIFF metadata files
    for directory in cs_paths:
        tif_files += get_matching_files(os.path.join(directory, "cells", "tifs"), "cell", ["_dl", "metadata"])
    # print("tif files:\n", tif_files)

    # Extract coverslip names from TIFF files
    for cs in cs_paths:
        tifs = os.listdir(os.path.join(cs, "cells", "tifs"))
        if not tifs:
            raise IncorrectConfigException(f"No TIFF files found in {cs}\\cells\\tifs")
        first_tif = tifs[0]
        cs_name = '_'.join(first_tif.split("_")[:-2])
        cs_names.append(cs_name)
    # print("cs names:\n", cs_names)

    # --- BINNING section ---
    try:
        use_timestamps = config.getboolean("BINNING", "use_timestamps")
    except KeyError as e:
        raise IncorrectConfigException(f"Missing parameter in [BINNING]: {e}")
    try:
        bin_size = float(config["BINNING"]["bin_size"])
    except KeyError:
        raise IncorrectConfigException("Section [BIN_SIZE] missing in config file.")
    if use_timestamps == True:
        bin_size_time = bin_size
        bin_size_cells = 0
    elif use_timestamps == False:
        bin_size_cells = bin_size
        bin_size_time = 0
    else:
        raise IncorrectConfigException(f"Check 'use_timestamps' in [BINNING]")
        # print("use timestamps:", use_timestamps)
    # print("bin size cells:", bin_size_cells)
    # print("bin size time:", bin_size_time)

    # --- PLOT_SETTINGS section ---
    try:
        plot_color = config["PLOT_SETTINGS"].get("dot_color", "#FFA500") or "#FFA500"
        ligand_exists = config.getboolean("PLOT_SETTINGS", "ligand", fallback=False)
        if ligand_exists:
            t_lig = config["PLOT_SETTINGS"]["ligand_index"]
        else:
            t_lig = 0
        ligand_name = config["PLOT_SETTINGS"]["ligand_name"]
        error_type = config["PLOT_SETTINGS"]["error_type"]
    except KeyError as e:
        raise IncorrectConfigException(f"Missing parameter in [PLOT_SETTINGS]: {e}")
    # print("dot/plot color:", plot_color)
    # print("ligand (exists):", ligand_exists)
    # print("ligand index:", t_lig)
    # print("ligand name:", ligand_name)
    # print("error type:", error_type)

    # --- STAT_SETTINGS section ---
    try:
        run_stats = config.getboolean("STAT_SETTINGS", "run_stats", fallback=False)
        if run_stats:
            alpha = float(config["STAT_SETTINGS"]["alpha_norm"])
            p1 = float(config["STAT_SETTINGS"]["alpha_sign_1"])
            p2 = float(config["STAT_SETTINGS"]["alpha_sign_2"])
            p3 = float(config["STAT_SETTINGS"]["alpha_sign_3"])
        else:
            alpha = p1 = p2 = p3 = None
    except KeyError:
          raise IncorrectConfigException("Section [STAT_SETTINGS] missing in config file.")

    # print(run_stats)
    # print(alpha)
    # print(p1)
    # print(p2)
    # print(p3)

    # --- SAVE_DIR section ---
    try:
        save_dir = config["SAVE_DIR"]["save_dir"]
    except KeyError:
        raise IncorrectConfigException("Parameter save_dir missing in [SAVE_DIR].")

    # print("save dir:\n", save_dir)

    # --- Collect all .h5 files ---
    for path in h5_paths:
        if isinstance(path, float):
            raise IncorrectConfigException("Invalid .h5 file path (float value found).")
        h5_files += get_matching_files(path, ".h5", ["statistics.h5"])

    # print("h5 files:\n", h5_files)

     # --- Prepare output directory structure ---
    os.makedirs(save_dir, exist_ok=True)
    time_dir = os.path.join(save_dir, "timeResolvedAnalysis")
    if os.path.exists(time_dir):
        shutil.rmtree(time_dir)
    os.mkdir(time_dir)

    # print("\nOutput folder prepared at:", time_dir, "\n")

    # --- Final structured return ---
    return {
        # File system
        "save_dir": time_dir,
        "cs_paths": cs_paths,
        "cs_names": cs_names,
        "h5_files": h5_files,
        "tif_files": tif_files,

        # Time and binning settings
        "use_timestamps": use_timestamps,
        "bin_size_cells": bin_size_cells,
        "bin_size_time": bin_size_time,

        # Plot configuration
        "plot_color": plot_color,
        "ligand_exists": ligand_exists,
        "t_lig": t_lig,
        "ligand_name": ligand_name,
        "error_type": error_type,

        # Statistics
        "run_stats": run_stats,
        "alpha": alpha,
        "p1": p1,
        "p2": p2,
        "p3": p3,
    }


def load_and_sort_input_data(h5_files, tif_files, cs_names):
    """
    Match each .h5 file to its corresponding .tif file based on filename substring matching.
    Returns a dictionary mapping each coverslip name to a list of .h5 filenames.
    """

    # Helper function to find the matching .tif file for a given .h5 file
    def find_matching_tif(filename, tif_files, exclusion_str="metadata"):
        """Return the .tif file that contains 'filename' and does not contain 'exclusion_str'."""
        matches = [tif for tif in tif_files if filename in tif and exclusion_str not in tif]
        if len(matches) > 1:
            raise ValueError(f"Multiple matches found for '{filename}'")
        elif len(matches) == 0:
            raise FileNotFoundError(f"No matching .tif file found for '{filename}'")
        return matches[0]

    # --- Helper function: natural sorting for filenames: cell_1, cell_10, cell_2, ... -> cell_1, cell_2, ..., cell_10
    def sort_cells(cells):
        def natural_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        return sorted(cells, key=natural_key)

    # Initialize output dictionary
    coverslip_dict = {c: [] for c in cs_names}

    # Assign .h5 files to their coverslip
    for h5 in h5_files:
        filename = h5.split("\\")[-1][:-2]
        tif = find_matching_tif(filename, tif_files, exclusion_str="metadata")  # remove path and last two chars (.h5)
        coverslip_name = '_'.join(tif.split("\\")[-1].split('_')[:-2])
        coverslip_dict[coverslip_name].append(filename)

    # Sort the cell lists for each coverslip
    for cs in coverslip_dict:
        coverslip_dict[cs] = sort_cells(coverslip_dict[cs])

    return coverslip_dict


def load_cell_data(coverslip_name, coverslip_cells, h5_files, tif_files):
    """
    Loads all cells of a coverslip into a DataFrame.

    :param coverslip_name: name of the coverslip
    :param coverslip_cells: a correctly sorted list of all cell names for this coverslip
    :param h5_files: list of h5 files
    :param tif_files: list of tif files (used for timestamp retrieval)
    :return: a DataFrame with all cell data for this coverslip
    """

    # Extract acquisition timestamp from metadata text file
    def get_acquisition_time_from_meta(tif_path):
        meta_path = os.path.splitext(tif_path)[0] + "_MMStack_Pos0_metadata.txt"
        if not os.path.exists(meta_path):
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if '"Time"' in line:
                    # Extract timestamp from metadata
                    match = re.search(r'"Time"\s*:\s*"([^"]+)"', line)
                    if match:
                        return match.group(1)
        return None

    # Helper function: find matching file
    def find_file(substring, files, exclusion="metadata"):
        matches = [f for f in files if substring in f and exclusion not in f]
        if len(matches) == 0:
            raise FileNotFoundError(f"No file found for {substring}")
        if len(matches) > 1:
            raise ValueError(f"Multiple files found for {substring}")
        return matches[0]

    # Prepare empty DataFrame for each coverslip
    columns = [
        "Cell Name", "Time",
        "P_immobile", "P_confined", "P_free",
        "D_global", "D_immobile", "D_confined", "D_free",
        "L_global", "L_immobile", "L_confined", "L_free",
        "N_global", "N_immobile", "N_confined", "N_free",
        "confinement_radius",
        "DE_immobile", "DE_confined", "DE_free",
        "LE_immobile", "LE_confined", "LE_free"
    ]
    coverslip_data = pd.DataFrame(columns=columns)

    cs_start_time = None

    # Iterate over all cells
    for cell in coverslip_cells:

        # print("Loading cell: ", cell)

        h5_path = find_file(cell, h5_files)
        tif_path = find_file(cell, tif_files)

        # Retrieve acquisition time
        time_str = get_acquisition_time_from_meta(tif_path)
        if not time_str:
            raise ValueError(f"--- No acquisition time found in metadata file for: {tif_path} ---")

        # Parse datetime (example format: "2025-04-30 08:59:34 +0200")
        try:
            cell_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S %z")
        except ValueError:
            # Fallback: if no timezone info
            cell_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

        # Set start time if first cell
        if cs_start_time is None:
            cs_start_time = cell_time
            print("Start time:", cs_start_time)

        # Calculate elapsed time in minutes
        time_elapsed = (cell_time - cs_start_time).total_seconds() / 60.0
        time_elapsed = round(time_elapsed, 2)
        print(f"{cell}: {time_elapsed} min")

        # Load diffusion data
        with h5py.File(h5_path, "r") as hdf:

            stats_global = hdf["statistics"]["statistics_3"][()]  # structured array with dtype contents
            # diffusion_infos = hdf['diffusion']['diffusionInfos'][()]
            # print(stats_global.dtype.names)
            # print(diffusion_infos.dtype.names)

            # timestamp of first cell
            if cs_start_time is None:
                cs_start_time = os.path.getmtime(tif_path)
                print("start time: ", cs_start_time)

            # diffusion modes
            p_immobile = float(stats_global['immobile+notype [%]'])
            p_confined = float(stats_global['confined [%]'])
            p_free = float(stats_global['free [%]'])

            # diffusion coefficients
            dg_immobile = 0 if math.isnan(
                stats_global['mean D immobile+notype [μm²/s]']) else float(stats_global['mean D immobile+notype [μm²/s]'])
            dg_confined = 0 if math.isnan(
                stats_global['mean D confined [μm²/s]']) else float(stats_global['mean D confined [μm²/s]'])
            dg_free = 0 if math.isnan(
                stats_global['mean D free [μm²/s]']) else float(stats_global['mean D free [μm²/s]'])
            d_global = (dg_immobile * p_immobile + dg_confined * p_confined + dg_free * p_free) * 0.01
            # d_global = np.nanmean(diffusion_infos['diffusion coefficient [μm²/s]'])  # yields exactly the same as line below

            # diffusion coefficient errors
            de_immobile = 0 if math.isnan(
                stats_global['Δ mean D immobile+notype [μm²/s]']) else float(stats_global['Δ mean D immobile+notype [μm²/s]'])
            de_confined = 0 if math.isnan(
                stats_global['Δ mean D confined [μm²/s]']) else float(stats_global['Δ mean D confined [μm²/s]'])
            de_free = 0 if math.isnan(
                stats_global['Δ mean D free [μm²/s]']) else float(stats_global['Δ mean D free [μm²/s]'])

            # segment lengths
            lg_immobile = 0 if math.isnan(
                stats_global['mean length immobile+notype [frames]']
            ) else float(stats_global['mean length immobile+notype [frames]'])
            lg_confined = 0 if math.isnan(
                stats_global['mean length confined [frames]']
            ) else float(stats_global['mean length confined [frames]'])
            lg_free = 0 if math.isnan(
                stats_global['mean length free [frames]']
            ) else float(stats_global['mean length free [frames]'])
            l_global = (lg_immobile * p_immobile + lg_confined * p_confined + lg_free * p_free) * 0.01

            # segment lengths errors
            le_immobile = 0 if math.isnan(
                stats_global['Δ mean length immobile+notype [frames]']) else float(stats_global['Δ mean length immobile+notype [frames]'])
            le_confined = 0 if math.isnan(
                stats_global['Δ mean length confined [frames]']) else float(stats_global['Δ mean length confined [frames]'])
            le_free = 0 if math.isnan(
                stats_global['Δ mean length free [frames]']) else float(stats_global['Δ mean length free [frames]'])

            # confinement radius & number of segments (?)
            n_confined = n_free = n_immobile = 0
            confinement_radii_sum = 0

            for j in hdf["rossier"]["rossierStatistics"][()]: # TODO: not sure what this is for?
                if j[2] == 1:
                    n_confined += 1
                    confinement_radii_sum += j[7]
                elif j[3] == 1:
                    n_free += 1
                elif j[1] == 1 or j[4] == 1:
                    n_immobile += 1

            n_global = n_immobile + n_confined + n_free

            confinement_radius = confinement_radii_sum / n_confined if n_confined > 0 else 0

            # append a row for each cell
            coverslip_data.loc[len(coverslip_data)] = (
                cell, time_elapsed,
                p_immobile, p_confined, p_free,
                d_global, dg_immobile, dg_confined, dg_free,
                l_global, lg_immobile, lg_confined, lg_free,
                n_global, n_immobile, n_confined, n_free,
                confinement_radius,
                de_immobile, de_confined, de_free,
                le_immobile, le_confined, le_free
            )

    if coverslip_data.empty:
        raise ValueError(f"Coverslip '{coverslip_name}' resulted in an empty DataFrame!")

    coverslip_data.sort_values("Time", inplace=True)
    coverslip_data.reset_index(drop=True, inplace=True)

    return coverslip_data


def bin_input_data(all_coverslips_data, use_timestamps, bin_size_time, bin_size_cells, allow_empty_time_bins=True):
    """
    Bins all cells from *all* coverslips together (shared bin edges) by time or cell number.
    Calculates mean, SD, SEM per bin and returns same structure as before.
    """

    # print("\n", all_coverslips_data, "\n")
    # TODO: add behavior with negative times! (before ligand addition)

    def determine_bins(frame, use_timestamps, bin_size_time, bin_size_cells, allow_empty_time_bins=True):
        """
        If use_timestamps==True: returns fixed time windows anchored at global min(Time).
        If use_timestamps==False: returns index-based bins of size bin_size_cells.
        """
        if frame.empty:
            print("  determine_bins: empty global frame -> no bins")
            return []

        if use_timestamps:
            print(f"  determine_bins: GLOBAL time-based binning with window {bin_size_time} min (fixed windows).")
            # compute minutes from global start
            if pd.api.types.is_datetime64_any_dtype(frame['Time']):
                start_time = frame['Time'].min()
                minutes = (frame['Time'] - start_time).dt.total_seconds() / 60.0
            elif pd.api.types.is_numeric_dtype(frame['Time']):
                start_time = frame['Time'].min()
                minutes = frame['Time'].astype(float) - float(start_time)
            else:
                raise TypeError("Column 'Time' must be datetime or numeric (minutes).")

            max_min = minutes.max()
            n_windows = int(np.ceil((max_min + 1e-9) / bin_size_time))  # number of full windows
            bins = []
            print(f"\nAssigning {len(frame)} entries into {n_windows + 1} time bins ({bin_size_time} min each):")

            for w in range(n_windows + 1):
                ws = w * bin_size_time
                we = ws + bin_size_time
                mid_time = (ws + we) / 2
                idx = list(minutes[(minutes >= ws) & (minutes < we)].index)

                # Prepare detailed info
                if idx:
                    details = "\n".join(
                        [f"      - | {frame.loc[i, 'Cell Name']:<35} | {minutes[i]:6.2f} min" for i in idx])
                else:
                    details = "      <empty>"

                print(f"  Window {w:02d}: {ws:6.2f}–{we:6.2f} min → {len(idx)} entries\n{details}")
                if idx or allow_empty_time_bins:
                    bins.append(idx)

            print("-" * 60)
            return bins

        else:
            print(f"  determine_bins: GLOBAL cell-count binning with bin_size {bin_size_cells}")
            bins = [list(range(i, min(i + bin_size_cells, len(frame))))
                    for i in range(0, len(frame), bin_size_cells)]
            for bi, b in enumerate(bins):
                names = f"{frame.iloc[b[0], 0]} - {frame.iloc[b[-1], 0]}" if b else "empty"
                print(f"    cell-bin {bi}: indices {b} -> {names}")
            return bins

    def aggregate_bins(frame, bins):
        """
        Aggregates rows according to provided bins (list of lists of row indices)
        Returns a DataFrame with mean, SD, SEM for all numeric columns.
        """

        binned_rows = []
        numeric_cols = frame.select_dtypes(include='number').columns

        for bin_idx, bin_indices in enumerate(bins):
            if not bin_indices:
                # empty bin
                empty_series = pd.Series({col: np.nan for col in numeric_cols})
                for col in numeric_cols:
                    empty_series[f"{col}_sd"] = np.nan
                    empty_series[f"{col}_sem"] = np.nan
                empty_series['Cell_range'] = "empty"
                empty_series['Num_cells'] = 0
                binned_rows.append(empty_series)
                continue

            bin_df = frame.iloc[bin_indices]
            mean_vals = bin_df[numeric_cols].mean()
            sd_vals = bin_df[numeric_cols].std()
            sem_vals = bin_df[numeric_cols].sem()

            combined = mean_vals.copy()
            for col in numeric_cols:
                combined[f"{col}_sd"] = sd_vals[col]
                combined[f"{col}_sem"] = sem_vals[col]

            combined['Cell_range'] = f"{bin_indices[0]}-{bin_indices[-1]}"
            combined['Num_cells'] = len(bin_indices)
            binned_rows.append(combined)

            print(f"Bin {bin_idx}: Cells {bin_indices[0]}-{bin_indices[-1]} ({len(bin_indices)} cells)")
            print("-" * 40)

        binned_df = pd.DataFrame(binned_rows)
        print(f"Completed aggregation. Total bins: {len(binned_df)}\n")
        return binned_df

    # --- Combine all coverslips into one DataFrame ---
    print("\nCombining all coverslips for global binning...")
    global_df = pd.concat(all_coverslips_data.values(), ignore_index=True)
    print(f"Global dataset size: {len(global_df)} rows from {len(all_coverslips_data)} coverslips.")

    # --- Determine global bins ---
    bins = determine_bins(global_df, use_timestamps, bin_size_time, bin_size_cells, allow_empty_time_bins)

    # --- Aggregate globally ---
    global_binned = aggregate_bins(global_df, bins)
    print("--------------------\n", global_binned, "----------------------\n")

    # --- Assemble return structures ---
    all_coverslips_binned = {"GLOBAL": global_binned}
    stacked_data = global_binned.copy()
    largest_bindex = 0  # only one entry

    print(f"\nGlobal stacked data: {len(stacked_data)} rows in total.")
    print("Binning completed successfully (GLOBAL).")

    return all_coverslips_binned, stacked_data, largest_bindex


def main(config_path):

    start_time = time.time()

    # Load configuration
    config = load_user_input(config_path)   # returns dictionary!

    # print(config)
    # print(config["tif_files"], "\n----------------------")

    # Load and sort input data
    coverslip_dict = load_and_sort_input_data(
        h5_files=config["h5_files"],
        tif_files=config["tif_files"],
        cs_names=config["cs_names"]
    )

    # Load cell data for each coverslip into a dictionary & assign a relative timestamp
    all_coverslips_data = {}  # key = coverslip_name, value = DataFrame with values for each cell in the coverslip
    for cs_name, cs_cells in coverslip_dict.items():
        print(f"\nLoading data for coverslip {cs_name} with {len(cs_cells)} cells...")
        all_coverslips_data[cs_name] = load_cell_data(
            coverslip_name=cs_name,
            coverslip_cells=cs_cells,
            h5_files=config["h5_files"],
            tif_files=config["tif_files"]
        )

        # print(all_coverslips_data, "pppppppppppppppppppppp")

        # Zugriff z.B. auf DataFrame von einem Coverslip:
        # df = all_coverslips_data["250430_CS2_CHO_HT7FGFR1c_SiRHTL_FGF1"]
        # coverslip_data[cs_name] = coverslip_data

    # Process and bin input data
    binned_data, stacked_data, largest_bindex = bin_input_data(
        all_coverslips_data = all_coverslips_data,
        use_timestamps=config["use_timestamps"],
        bin_size_time = config["bin_size_time"],
        bin_size_cells = config["bin_size_cells"]
    )

    # Print execution time
    print("--- %s seconds ---" % (time.time() - start_time))

# Entry point for script execution
if __name__ == "__main__":

    try:
        cfg_path = sys.argv[1]
        main(cfg_path)
    except FileExistsError:
        print("Usage: python timeResolvedAnalysis.py your_config_file.ini")
