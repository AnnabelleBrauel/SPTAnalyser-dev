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
import matplotlib.pyplot as plt
import itertools

#import tifffile
#from tifffile import TiffFile
#import warnings
#from fileinput import filename
#from ftplib import all_errors
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
        all_files = os.listdir(os.path.join(cs, "cells", "tifs"))
        tifs = [f for f in all_files if f.lower().endswith(('.tif', '.tiff'))]
        if not tifs:
            raise IncorrectConfigException(f"No TIFF files found in {cs}\\cells\\tifs")
        first_tif = tifs[0]
        cs_name = '_'.join(first_tif.split("_")[:-2])
        cs_names.append(cs_name)
    # print("cs names:\n", cs_names)

    # --- LIGAND_ADDITION section ---
    try:
        ligand_time = config["LIGAND_ADDITION"]["ligand_time"]
    except KeyError as e:
        raise IncorrectConfigException(f"Missing parameter in [LIGAND_ADDITION]: {e}")

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

        # Ligand configuration
        "ligand_time": ligand_time,

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


def load_and_sort_input_data(h5_files, tif_files, cs_names, cs_paths, ligand_time):
    """
    Match each .h5 file to its corresponding .tif file based on filename substring matching.
    Optionally, find a .txt file per coverslip to get the ligand_time.
    Returns a dictionary mapping each coverslip name to:
        {'cells': [sorted .h5 filenames], 'ligand_time': time_str or None}
    """

    # Helper function to find the matching .tif file for a given .h5 file
    def find_matching_tif(filename, tif_files):
        # Return the .tif file that contains 'filename' and does not contain 'exclusion_str'.
        exclusion_strs = ["metadata", ".txt", "_dl"]
        matches = [
            tif for tif in tif_files
            if filename in tif and all(excl not in tif for excl in exclusion_strs)
        ]
        if len(matches) > 1:
            raise ValueError(f"Multiple matches found for '{filename}'")
        elif len(matches) == 0:
            raise FileNotFoundError(f"No matching .tif file found for '{filename}'")
        return matches[0]


    # Helper function to sort filenames naturally: cell_1, cell_10, cell_2, ... -> cell_1, cell_2, ..., cell_10
    def sort_cells(cells):
        def natural_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        return sorted(cells, key=natural_key)

    # Initialize output dictionary: each coverslip has a list of cells and a time of ligand addition
    coverslip_dict = {c: {'cells': [], 'ligand_time': None} for c in cs_names}

    # Assign .h5 files to their coverslip
    for h5 in h5_files:
        filename = h5.split("\\")[-1][:-2]
        tif = find_matching_tif(filename, tif_files)  # remove path and last two chars (.h5)
        coverslip_name = '_'.join(tif.split("\\")[-1].split('_')[:-2])
        coverslip_dict[coverslip_name]['cells'].append(filename)

    # Sort the cell lists for each coverslip
    for cs in coverslip_dict:
        coverslip_dict[cs]['cells'] = sort_cells(coverslip_dict[cs]['cells'])

    print("\n")

    # Extract ligand_time from .txt files if requested
    if ligand_time == "txt":
        for cs_name, cs_path in zip(cs_names, cs_paths):
            found = False
            for root, dirs, files in os.walk(cs_path):
                for f in files:
                    if f.endswith('.txt'):
                        # Use search to match HH_MM anywhere in filename
                        match = re.search(r'(\d{1,2})_(\d{2})\.txt', f)
                        if match:
                            coverslip_dict[cs_name]['ligand_time'] = f"{match.group(1)}:{match.group(2)}"
                            print(f"Found ligand addition time for {cs_name}: {coverslip_dict[cs_name]['ligand_time']}")
                            found = True
                            break
                if found:
                    break
                if not found:
                    continue

    return coverslip_dict


def load_cell_data(coverslip_name, coverslip_cells, ligand_time, h5_files, tif_files):
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

        if ligand_time == None:

            # Set start time if first cell
            if cs_start_time is None:
                cs_start_time = cell_time
                print("Start time:", cs_start_time)

            # Calculate elapsed time in minutes
            time_elapsed = (cell_time - cs_start_time).total_seconds() / 60.0
            time_elapsed = round(time_elapsed, 2)
            print(f"{cell}: {time_elapsed} min")

        else:

            # Convert ligand_time string (e.g. "19:09") to datetime on the same day as first cell
            ligand_hour, ligand_minute = map(int, ligand_time.split(":"))
            ligand_datetime = cell_time.replace(hour=ligand_hour, minute=ligand_minute, second=0, microsecond=0)

            # Calculate relative time in minutes to ligand_time
            time_elapsed = (cell_time - ligand_datetime).total_seconds() / 60.0
            time_elapsed = round(time_elapsed, 2)

            print(f"{cell}: {time_elapsed} min relative to ligand addition")

        # Load diffusion data
        with h5py.File(h5_path, "r") as hdf:

            stats_global = hdf["statistics"]["statistics_3"][()]  # structured array with dtype contents
            # diffusion_infos = hdf['diffusion']['diffusionInfos'][()]
            # print(stats_global.dtype.names)
            # print(diffusion_infos.dtype.names)

            # timestamp of first cell
            #if cs_start_time is None:
            #    cs_start_time = os.path.getmtime(tif_path)
            #    print("start time: ", cs_start_time)

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
    Returns one dataframe with binned data from all coverslips
    """

    def determine_bins(frame, use_timestamps, bin_size_time, bin_size_cells, allow_empty_time_bins=True):
        """
        If use_timestamps==True: returns fixed time windows anchored at global min(Time).
        If use_timestamps==False: returns index-based bins of size bin_size_cells.
        """
        if frame.empty:
            print("  determine_bins: empty global frame -> no bins")
            return []

        if use_timestamps:
            print(f"\nDetermine_bins: GLOBAL time-based binning with window {bin_size_time} min fixed windows...")

            # Use Time column from all_coverslips_data
            if pd.api.types.is_numeric_dtype(frame['Time']):
                minutes = frame['Time'].astype(float)
            else:
                raise TypeError("Column 'Time' must be numeric (minutes).")

            # Bin edges from min to max, including negative times
            min_bin = np.floor(minutes.min() / bin_size_time) * bin_size_time
            max_bin = np.ceil(minutes.max() / bin_size_time) * bin_size_time
            bin_edges = np.arange(min_bin, max_bin + bin_size_time, bin_size_time)

            bins = []
            for i in range(len(bin_edges) - 1):
                ws = bin_edges[i]
                we = bin_edges[i + 1]
                mid_time = (ws + we) / 2
                idx = list(minutes[(minutes >= ws) & (minutes < we)].index)

                # Print detailed info
                if idx:
                    details = "\n".join(
                        [f"      - | {frame.loc[i, 'Cell Name']:<35} | {minutes[i]:6.2f} min" for i in idx])
                else:
                    details = "      <empty>"
                print(f"  Bin {i:02d}: {ws:6.2f} – {we:6.2f} min → {len(idx)} entries\n{details}")

                if idx or allow_empty_time_bins:
                    bins.append((idx, mid_time))

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
        exclude = {'Time', 'Frame', 'Cell_ID'}
        numeric_cols = [c for c in frame.select_dtypes('number') if c not in exclude]

        for bin_idx, (bin_indices, mid_time) in enumerate(bins):
            if not bin_indices:

                # empty bins -> fill with NaNs, no calculations
                empty_series = pd.Series({col: np.nan for col in numeric_cols})
                for col in numeric_cols:
                    empty_series[f"{col}_sd"] = np.nan
                    empty_series[f"{col}_sem"] = np.nan
                empty_series['Time'] = mid_time
                empty_series['Cell_range'] = "empty"
                empty_series['Num_cells'] = 0
                binned_rows.append(empty_series)
                continue

            bin_df = frame.iloc[bin_indices]

            combined = pd.Series(dtype=float)

            # percent values -> compute weighted mean to account for different cell counts per row
            for col in ['P_immobile', 'P_confined', 'P_free']:
                n_col = f"N_{col.split('_')[1]}"
                if col in bin_df.columns and n_col in bin_df.columns:
                    total_cells = bin_df['N_global'].sum()
                    if total_cells > 0:
                        combined[col] = bin_df[n_col].sum() / total_cells * 100
                    else:
                        combined[col] = np.nan

            # fit parameters -> use median for robustness & calculate SD and SEM
            for col in ['D_global', 'D_immobile', 'D_confined', 'D_free']:
                if col in bin_df.columns:
                    combined[col] = np.median(bin_df[col])
                    combined[f"{col}_sd"] = bin_df[col].std()
                    combined[f"{col}_sem"] = bin_df[col].sem()

            # count values -> sum across all cells in the bin
            for col in ['N_global', 'N_immobile', 'N_confined', 'N_free']:
                if col in bin_df.columns:
                    combined[col] = bin_df[col].sum()

            # time & metadata
            combined['Time'] = bin_df['Time'].mean() if 'Time' in bin_df else mid_time
            combined['Cell_range'] = f"{bin_indices[0]}-{bin_indices[-1]}"
            combined['Num_cells'] = len(bin_indices)

            binned_rows.append(combined)

        binned_df = pd.DataFrame(binned_rows)
        return binned_df

    # Combine all coverslips into one global DataFrame
    global_df = pd.concat(all_coverslips_data.values(), ignore_index=True)

    bins = determine_bins(global_df, use_timestamps, bin_size_time, bin_size_cells, allow_empty_time_bins)
    global_binned = aggregate_bins(global_df, bins)

    print("\nBinning completed successfully.\n")

    return global_binned


def plot_by_time(binned_data, data_for_each_cell, bin_size_time, ligand_exists):
    """
    Plots D_free vs Time as scatter points with error bars using D_free_sem.
    """

    #TODO: add plot for percentage of immobile fraction (or second y-axis in existing plot)

    # Ensure required columns exist
    required_cols = ["Time", "D_free", "D_free_sem"]
    for col in required_cols:
        if col not in binned_data.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")

    plt.figure(figsize=(8, 4))

    # Draw horizontal lines per bin
    for _, row in binned_data.iterrows():
        row_time = row['Time']

        # Determine actual bin edges (floor/ceil to nearest multiple of bin_size_time)
        start = bin_size_time * np.floor(row_time / bin_size_time)
        end = start + bin_size_time  # bin width is fixed

        y = row['D_free']
        yerr = row['D_free_sem']

        # Plot shaded error box
        plt.fill_between([start, end], y - yerr, y + yerr, color='lightgray', alpha=0.5)
        # Plot horizontal line at median
        plt.hlines(y=y, xmin=start, xmax=end, color='blue', lw=2)

    # Plot individual cell data with different colors per coverslip
    colors = itertools.cycle(['red', 'green', 'blue', 'orange', 'purple', 'cyan'])
    for coverslip, df in data_for_each_cell.items():
        color = next(colors)
        plt.scatter(df["Time"], df["D_free"], label=coverslip, color=color, alpha=0.7, s=20)

    if ligand_exists:
        # Vertical line at 0 min with text for ligand addition
        plt.axvline(x=0, color='red', linestyle='--', lw=1.5)
        ylim = plt.ylim()
        plt.text(
            x=0.2,
            y=ylim[0] + 0.05*(ylim[1]-ylim[0]),  # text is 5% above x-axis
            s="+ ligand",
            color='red',
            fontsize=10,
            verticalalignment='bottom')

    # Axis labels
    plt.xlabel("time / min")
    plt.ylabel("D_free")
    plt.title(f"free diffusion coefficient ({bin_size_time} min bins)")
    plt.tight_layout()
    plt.show()


def main(config_path):

    start_time = time.time()

    # Load configuration
    config = load_user_input(config_path)   # returns dictionary!

    # Load and sort input data
    coverslip_dict = load_and_sort_input_data(
        h5_files=config["h5_files"],
        tif_files=config["tif_files"],
        cs_names=config["cs_names"],
        cs_paths=config["cs_paths"],
        ligand_time=config["ligand_time"],
    )

    # Load cell data for each coverslip into a dictionary & assign a relative timestamp
    all_coverslips_data = {}  # key = coverslip_name, value = DataFrame with values for each cell in the coverslip
    for cs_name, cs_data in coverslip_dict.items():  # cs_data = {"cells": [...], "ligand_time": "..."}
        coverslip_cells = cs_data["cells"]
        ligand_time = cs_data.get("ligand_time")  # None, if no ligand
        print(f"\nLoading data for coverslip {cs_name} with {len(coverslip_cells)} cells and ligand_time {ligand_time}...")
        all_coverslips_data[cs_name] = load_cell_data(
            coverslip_name=cs_name,
            coverslip_cells=coverslip_cells,
            ligand_time=ligand_time,
            h5_files=config["h5_files"],
            tif_files=config["tif_files"]
        )

    # Process and bin input data
    binned_data = bin_input_data(
        all_coverslips_data=all_coverslips_data,
        use_timestamps=config["use_timestamps"],
        bin_size_time=config["bin_size_time"],
        bin_size_cells=config["bin_size_cells"]
    )

    if config["use_timestamps"] == True:
        plot_by_time(
            binned_data=binned_data,
            data_for_each_cell=all_coverslips_data,
            bin_size_time=config["bin_size_time"],
            ligand_exists=config["ligand_exists"]
        )

    # Print execution time
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":

    try:
        cfg_path = sys.argv[1]
        main(cfg_path)
    except FileExistsError:
        print("Usage: python timeResolvedAnalysis.py your_config_file.ini")
