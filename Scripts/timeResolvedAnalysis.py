"""
@author: Alexander Niedrig
Research group Heilemann
Institute for Physical and Theoretical Chemistry, Goethe University Frankfurt a.M.
Calculates mean values over timeframes, plots them, runs statistical tests, and rearranges input data in an output file
"""
import configparser
import math
import os
import shutil
import sys
import time
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scy


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


def parent_string(substring, parents, exclusion_str):
    """
    Searches a list of strings ('parents') for entries that contain a specific substring
    but do NOT contain a given exclusion string.

    :param substring: Substring to search for
    :param parents: List of potential parent strings
    :param exclusion_str: List of strings that must not be present in the parent string
    :return: A single matching string (raises exception if ambiguous or not found)
    """
    parent = [a for a in parents if
              (substring in a and exclusion_str not in a)]
    if len(parent) > 1:
        raise IncorrectConfigException("multiple options when looking for " + substring)
    elif len(parent) == 0:
        raise IncorrectConfigException("no file found when looking for " + substring)
    elif len(parent) == 1:
        parent = parent[0]
    return parent


def sort_cells(cells):
    """
    Sorts a list of cell names in alphanumeric order, taking into account numeric parts.
    For example: ["Cell_1", "Cell_10", "Cell_2"] -> ["Cell_1", "Cell_2", "Cell_10"]

    :param cells: List of cell name strings
    :return: List of alphanumerically sorted cell names
    """
    digits = [[]]
    for cell in cells:
        number = cell.split("_")[-1].split(".")[0]  # Extracts numeric part from the name
        while True:
            try:
                digits[len(number)].append(cell)
                break
            except IndexError:
                digits.append([])  # Add new sublist if needed
                continue
    sorted_list = []
    for dig in digits:
        dig.sort()
        sorted_list += dig
    return sorted_list


def insert_error(mean_frame, sd_frame, sem_frame):
    """
    Inserts standard deviation (SD) and standard error of the mean (SEM) columns
    into a dataframe containing mean values, directly after their corresponding mean column.

    :param mean_frame: Dataframe with mean values
    :param sd_frame: Dataframe with standard deviations
    :param sem_frame: Dataframe with standard errors of the mean
    :return: Modified dataframe with SD and SEM inserted
    """
    # Remove metadata columns (assumes first two are not data columns)
    sd_frame = sd_frame.iloc[:, 2:]
    sem_frame = sem_frame.iloc[:, 2:]
    i = 0
    while i < sd_frame.shape[1]:
        name = sd_frame.columns[i]
        sd = sd_frame.iloc[:, i]
        sem = sem_frame.iloc[:, i]
        # Rename error columns to indicate type
        sd.rename(name + '_SD', inplace=True)
        sem.rename(name + '_SEM', inplace=True)
        # Insert error columns right after the mean column
        mean_frame.insert(i * 3 + 3, name + '_SD', sd)
        mean_frame.insert(i * 3 + 4, name + '_SEM', sem)
        i += 1
    return mean_frame


def calc_mean_over_cs(binned_data, attribute):
    """
    Calculates the mean, standard deviation, and standard error of a specified attribute
    across a list of binned dataframes (e.g. from different coverslips).

    :param binned_data: List of binned dataframes
    :param attribute: Name pattern of the attribute to aggregate
    :return: Numpy array with [mean, std, sem] per row
    """
    attributes = pd.DataFrame()
    for i, df in enumerate(binned_data):
        # Select columns matching attribute name, but exclude error columns
        columns = [col for col in df.columns if attribute in col and not col.endswith(("_SD", '_SEM'))]
        for col in columns:
            attributes = pd.concat([attributes, df[col]], axis=1)

    # Stack results into a Numpy array
    stats = np.column_stack([
        attributes.mean(axis=1),
        attributes.std(axis=1),
        attributes.sem(axis=1)
    ])
    return stats


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
    fix, ax = plt.subplots()

    # Plot individual data points from each coverslip
    for frame in dataframes:
        color = ax.plot(
            frame.iloc[:, 1] / 60,
            frame[attribute],
            marker='o', ms=2, linestyle='None', color=clr
        )[0].get_color()

    # Calculate mean, std, sem across coverslips
    stats = calc_mean_over_cs(binned_data, attribute)

    left = 0
    longest_df = ''
    # Determine the longest dataframe for time bin labels
    for df in binned_data:
        if len(df) > len(longest_df):
            longest_df = df

    # Plot horizontal bars for mean values with error shading
    for i, row in enumerate(stats):
        if error_type == 'SEM':
            error = row[2]
        else:
            error = row[1]

        # Calculate right edge of the time bin (convert from seconds to minutes)
        if i == len(stats) - 1:
            right = float(longest_df.iloc[i, 1].split('-')[1]) / 60
        else:
            right = (((float(longest_df.iloc[i + 1, 1].split('-')[0]) - float(
                longest_df.iloc[i, 1].split('-')[1])) / 2) + float(longest_df.iloc[i, 1].split('-')[1])) / 60

        # Draw the mean bar
        ax.barh(row[0],  # mean_value? TODO
                width=right - left,
                height=ax.get_ylim()[1] * 0.005,  # Thin horizontal bar
                left=left,
                alpha=1,
                align='center',
                color='black')

        # Draw the error region
        ax.barh(row[0],  # mean_value? TODO
                width=right - left,
                height=error * 2,  # Full error width
                left=left,
                alpha=0.5,
                align='center',
                color='grey')
        left = right

    # Set Y-axis label based on attribute type
    atr = attribute.split('_')[0]
    if atr == 'P':
        ax.set_ylabel(attribute + ' [%]')
    elif atr == 'D':
        ax.set_ylabel(attribute + r' [$\mu m^2 s^{-1}$]')
    elif atr == 'L':
        ax.set_ylabel(attribute + ' [frames]')
    elif atr == 'N':
        ax.set_ylabel(attribute)
    elif atr == 'confinement_radius':
        ax.set_ylabel('confinement_radius' + r'$\mu m$')

    ax.set_xlim(left=-0.2)
    ax.set_xlabel('Time [s]')

    # Add vertical line for ligand addition
    if ligand_name == '':
        ligand_name = '+LIG'
    if len(t_lig) > 0 and t_lig[-1] == 's':
        ax.axvline(float(t_lig[:-1]) / 60, color='black')  # Convert to minutes
        ax.text(float(t_lig[:-1]) / 60, ax.get_ylim()[1],
                ligand_name, ha='center', va='bottom', color='black')

    return plt


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
    fig, ax = plt.subplots()

    # Plot individual dots (raw per-cell values)
    for frame in dataframes:
        ax.plot([float(cell_name.split("_")[-1]) for cell_name in frame.iloc[:, 0]], frame[attribute], marker='o',
                linestyle='None', color=clr)[0].get_color()

    # Calculate binned stats across coverslips
    stats = calc_mean_over_cs(binned_data, attribute)
    left = 0

    # Find the binned dataset with the most cells (for x-range info)
    lens = [int(j.iloc[-1, 0].split('-')[1]) for j in binned_data]
    max_index = lens.index(max(lens))
    maxframe = binned_data[max_index]

    # Draw horizontal bars for means and shaded error bars
    for i, row in enumerate(stats):
        if error_type == 'SEM':
            error = row[2]
        else:
            error = row[1]
        right = float(maxframe.iloc[i, 0].split('-')[1]) + 0.5
        ax.barh(row[0],
                width=right - left,
                height=ax.get_ylim()[1] * 0.005,
                left=left,
                alpha=1,
                align='center',
                color='black')
        ax.barh(row[0],
                width=right - left,
                height=error * 2, left=left,
                alpha=0.5,
                align='center',
                color='grey')
        left = right

    # Label Y-axis according to attribute type
    atr = attribute.split('_')[0]
    if atr == 'P':
        ax.set_ylabel(attribute + ' [%]')
    elif atr == 'D':
        ax.set_ylabel(attribute + r' [$\mu m^2 s^{-1}$]')
    elif atr == 'L':
        ax.set_ylabel(attribute + ' [frames]')
    elif atr == 'N':
        ax.set_ylabel(attribute)
    elif atr == 'confinement_radius':
        ax.set_ylabel('confinement_radius' + r' [$\mu m$]')

    ax.set_xlim(left=-0.2)
    ax.set_xlabel('Cells')

    # Add ligand marker if defined
    if len(t_lig) > 0 and t_lig[-1] == 'c':
        ax.axvline((float(t_lig[:-1])), color='black')
        if ligand_name == '':
            ligand_name = '+LIG'
        ax.text((float(t_lig[:-1])), ax.get_ylim()[1],
                ligand_name, ha='center', va='bottom', color='black')

    return plt


def bin_data_cells(frame, bin_size):
    """
    Bins a dataframe's rows by cell number, calculates mean, SD, and SEM per bin, and
    merges all results into a single dataframe with inserted error columns.

    :param frame: Raw dataframe with per-cell data
    :param bin_size: Bin size in number of cells
    :return: Dataframe with per-bin means and corresponding SD/SEM columns
    """
    mean_frame = pd.DataFrame(columns=frame.columns)
    sd_frame = pd.DataFrame(columns=frame.columns)
    sem_frame = pd.DataFrame(columns=frame.columns)

    # Derive base name for bin labels (e.g., CS2_P3_)
    cs_name = frame.iloc[0, 0].split('_')[0] + '_' + frame.iloc[0, 0].split('_')[-3] + '_'
    binnumber = 0
    row_index = 0
    complete = False

    # Iterate through the dataframe and collect bins
    while not complete:
        current_bin = pd.DataFrame(columns=frame.columns)
        # Collect rows within the current bin range
        while float(frame.iloc[row_index, 0].split('_')[-1]) <= binnumber * bin_size:
            current_bin.loc[len(current_bin)] = frame.iloc[row_index, :]
            row_index += 1
            if row_index == len(frame):  # End of DataFrame reached
                complete = True
                break
        # Skip empty bins (e.g., if no data points fall into this time range)
        if len(current_bin) == 0:
            binnumber += 1
            continue

        # Calculate and label mean per bin
        mean_frame.loc[len(mean_frame)] = current_bin.iloc[:, 2:].mean()
        mean_frame.iloc[-1, 0] = cs_name + "cell_" + str((binnumber - 1) * bin_size) + "-" + str(binnumber * bin_size)
        mean_frame.iloc[-1, 1] = str(int(current_bin.iloc[0, 1])) + "-" + str(int(current_bin.iloc[-1, 1]))

        # Calculate and label standard deviation per bin
        sd_frame.loc[len(sd_frame)] = current_bin.iloc[:, 2:].std()
        sd_frame.iloc[-1, 0] = cs_name + "cell_" + str((binnumber - 1) * bin_size) + "-" + str(binnumber * bin_size)
        sd_frame.iloc[-1, 1] = str(int(current_bin.iloc[0, 1])) + "-" + str(int(current_bin.iloc[-1, 1]))

        # Calculate and label standard error per bin
        sem_frame.loc[len(sem_frame)] = current_bin.iloc[:, 2:].sem()
        sem_frame.iloc[-1, 0] = cs_name + "cell_" + str((binnumber - 1) * bin_size) + "-" + str(binnumber * bin_size)
        sem_frame.iloc[-1, 1] = str(int(current_bin.iloc[0, 1])) + "-" + str(int(current_bin.iloc[-1, 1]))

    binnumber += 1  # Shouldn't this have one more tab? TODO

    # Merge all results and insert error columns
    outframe = insert_error(mean_frame, sd_frame, sem_frame)
    return outframe


def bin_data_time(frame, bin_size):
    """
    Groups rows of a dataframe into time bins, calculates statistical
    summaries for each bin, and returns a new dataframe with added SD and SEM columns.

    :param frame: Input dataframe containing single-cell data with time information
    :param bin_size: Size of time bins (in the same units as column 1 in the dataframe)
    :return: Dataframe containing the mean, SD, and SEM of each attribute per time bin
    """
    # Initialize empty dataframes to store the statistical summaries
    mean_frame = pd.DataFrame(columns=frame.columns)
    sd_frame = pd.DataFrame(columns=frame.columns)
    sem_frame = pd.DataFrame(columns=frame.columns)

    # Extract coverslip and condition name from the first entry for naming consistency
    cs_name = frame.iloc[0, 0].split('_')[0] + '_' + frame.iloc[0, 0].split('_')[-3] + '_'

    complete = False
    bin_number = 1
    row_index = 0

    while not complete:
        start = row_index
        current_bin = pd.DataFrame(columns=frame.columns)  # temporary dataframe for one bin

        # Accumulate rows into the current bin based on the time column
        while frame.iloc[
            row_index, 1] < bin_number * bin_size:
            current_bin.loc[len(current_bin)] = frame.iloc[row_index, :]
            row_index += 1
            if row_index == len(frame):
                complete = True
                break
        # Skip empty bins (e.g., if no data points fall into this time range)
        if len(current_bin) == 0:
            bin_number += 1
            continue

        # Calculate mean of all numeric columns and append to result dataframe
        mean_frame.loc[len(mean_frame)] = current_bin.iloc[:, 2:].mean()
        mean_frame.iloc[-1, 0] = cs_name + "cell_" + str(start) + "-" + str(start + len(current_bin))

        # Calculate standard deviation and append to result dataframe
        sd_frame.loc[len(sd_frame)] = current_bin.iloc[:, 2:].std()
        sd_frame.iloc[-1, 0] = cs_name + "cell_" + str(start) + "-" + str(start + len(current_bin))

        # Calculate standard error of the mean and append
        sem_frame.loc[len(sem_frame)] = current_bin.iloc[:, 2:].sem()
        sem_frame.iloc[-1, 0] = cs_name + "cell_" + str(start) + "-" + str(start + len(current_bin))

        # Add time interval label to column index 1
        if len(current_bin) > 0:
            mean_frame.iloc[-1, 1] = str((bin_number - 1) * bin_size) + "-" + str(bin_number * bin_size)
            sd_frame.iloc[-1, 1] = str((bin_number - 1) * bin_size) + "-" + str(bin_number * bin_size)
            sem_frame.iloc[-1, 1] = str((bin_number - 1) * bin_size) + "-" + str(bin_number * bin_size)

        bin_number += 1

    # Insert SD and SEM into the mean frame before returning
    outframe = insert_error(mean_frame, sd_frame, sem_frame)
    return outframe


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
    normality_frames = {}  # stores results of normality tests
    significance_frames = {}  # stores results of significance tests if enabled
    if not ligand:
        significance_frames = None
    compare_frame = pd.DataFrame()

    cell_count=[]

    # Determine number of bins based on cell count in first frame
    bin_number = int(len(frames[0]) / bin_size)
    if len(frames[0]) % bin_size != 0:
        bin_number += 1

    for j in range(bin_number):
        temp_frame = pd.DataFrame()  # accumulates all data for current bin

        # Collect all rows from each frame that fall into the current cell bin
        for frame in frames:
            bin_frame = pd.DataFrame(columns=temp_frame.columns)
            for index,row in frame.iterrows():
                cell_id = float(row[0].split('_')[-1])
                if j * bin_size < cell_id <= (j + 1) * bin_size:
                    bin_frame = bin_frame.append(row, ignore_index=True)
            temp_frame = pd.concat([temp_frame, bin_frame], axis=0)

        # Record number of valid cells in this bin
        try:
            cell_count[j]+= len(temp_frame)
        except IndexError:
            cell_count.append(len(temp_frame))

        # Skip bins with fewer than 3 cells
        if len(temp_frame) < 3:
            for name in temp_frame.columns[2:]:
                normality_frames[name].loc[len(normality_frames[name])] = pd.Series(
                    [j + 1, str(j * bin_size + 1) + '-' + str((j + 1) * bin_size), 'NaN', 'NaN',
                    f'Dataset too small ({len(temp_frame)})', 'NaN', 'NaN',
                    f'Dataset too small ({len(temp_frame)})']).values
                if ligand:
                    significance_frames[name].loc[len(significance_frames[name])] = pd.Series(
                        [f'1-{j + 1}', f"{j * bin_size + 1}-{(j + 1) * bin_size}", 'NaN', 'NaN',
                        f'Dataset too small ({len(temp_frame)})', 'NaN', 'NaN',
                        f'Dataset too small ({len(temp_frame)})']).values
            continue


        for name in temp_frame.columns[2:]:  # run tests on all measurement columns
            temp_frame2 = temp_frame.dropna(subset=[name])  # drop NaNs for this column

            # Create new result table for this attribute if needed
            if name not in normality_frames.keys():
                if compare_frame.empty:
                    compare_frame = temp_frame2.copy()
                norm_frame = pd.DataFrame(columns=[
                    'bin number', 'cell range', 'number of cells','Shapiro statistic', 'Shapiro p', 'Shapiro result',
                    'Kolmogorov-Smirnov statistic', 'Kolmogorov-Smirnov p', 'Kolmogorov-Smirnov result'])
                normality_frames[name] = norm_frame

            # Run normality tests (Shapiro-Wilk and Kolmogorov-Smirnov)
            statistics_shapiro, p_value_shapiro, statistics_kolmogorov, p_value_kolmogorov = normality_tests(temp_frame2, name)
            sr = 'not norm' if p_value_shapiro < alpha else 'norm'
            kr = 'not norm' if p_value_kolmogorov < alpha else 'norm'

            normality_frames[name].loc[len(normality_frames[name])] = pd.Series([
                j + 1, f"{j * bin_size + 1}-{(j + 1) * bin_size}",
                f"{len(temp_frame2)} {cell_count[j] - len(temp_frame2)} were dropped due to NaN entries",
                statistics_shapiro, p_value_shapiro, sr,
                statistics_kolmogorov, p_value_kolmogorov, kr
            ]).values

            # If ligand is used, compare this bin to the first one using significance tests
            if ligand:
                if name not in significance_frames.keys():
                    sign_frame = pd.DataFrame(columns=[
                        'compared bins', 'cell range', 'number of cells (control: '+str(cell_count)+' cells)',
                        'paired tTest statistic', 'paired tTest p', 'paired tTest result',
                        'Wilcoxon-signed-rank statistic', 'Wilcoxon p', 'Wilcoxon result'])
                    significance_frames[name] = sign_frame
                else:
                    stat_t_test, p_value_t_test, statistics_wilcoxon, p_value_wilcoxon = significance_tests(temp_frame2, compare_frame, name)

                    # Determine significance levels for each test
                    if p_value_t_test == 'NaN':
                        tr = 'test not applicable'
                    elif p_value_t_test < p3:
                        tr = '***'
                    elif p_value_t_test < p2:
                        tr = '**'
                    elif p_value_t_test < p1:
                        tr = '*'
                    else:
                        tr = 'no significant difference'

                    if p_value_wilcoxon == 'NaN':
                        wr = 'test not applicable'
                    elif p_value_wilcoxon < p3:
                        wr = '***'
                    elif p_value_wilcoxon < p2:
                        wr = '**'
                    elif p_value_wilcoxon < p1:
                        wr = '*'
                    else:
                        wr = 'no significant difference'

                    significance_frames[name].loc[len(significance_frames[name])] = pd.Series([
                        f'1-{j + 1}', f"{j * bin_size + 1}-{(j + 1) * bin_size}",
                        f"{len(temp_frame2)} {cell_count[j] - len(temp_frame2)} were dropped due to NaN entries",
                        stat_t_test, p_value_t_test, tr,
                        statistics_wilcoxon, p_value_wilcoxon, wr
                    ]).values

    # Return both result sets if ligand was used, else only normality results
    return (normality_frames, significance_frames) if ligand else normality_frames


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
    normality_frames = {}  # store results of normality tests
    significance_frames = {}  # stores results of significance tests if enabled
    if not ligand:
        significance_frames = None
    compare_frame = pd.DataFrame()

    # Determine the maximum time across all frames
    max_times = []
    for frame in frames:
        max_times.append(max(frame.iloc[:, 1]))
    max_time = max(max_times)

    # Determine the number of time bins needed
    bin_number = int(max_time / bin_size + 1)
    if max_time % bin_size != 0:
        bin_number += 1


    for i in range(1, bin_number):
        temp_frame = pd.DataFrame()

        # Collect all rows that fall into the current time bin
        for frame in frames:
            row_index = 0
            while frame.iloc[row_index, 1] < i * bin_size:
                if frame.iloc[row_index, 1] >= (i - 1) * bin_size:
                    temp_frame = temp_frame.append(frame.iloc[row_index, :])
                row_index += 1
                if row_index == len(frame):
                    break

        # Skip bins with insufficient data
        if len(temp_frame) < 3:
            for name in temp_frame.columns[2:]:
                normality_frames[name].loc[len(normality_frames[name])] = pd.Series(
                    [i + 1, f"{min(temp_frame['Time'])}-{max(temp_frame['Time'])}", 'NaN', 'NaN',
                     'Dataset too small', 'NaN', 'NaN', 'Dataset too small']).values
                if ligand:
                    significance_frames[name].loc[len(significance_frames[name])] = pd.Series(
                        [f'1-{i}', f"{min(temp_frame['Time'])}-{max(temp_frame['Time'])}", 'NaN', 'NaN',
                         'Dataset too small', 'NaN', 'NaN', 'Dataset too small']).values
            continue

        for name in temp_frame.columns[2:]:  # analyze each measurement column
            if name not in normality_frames.keys():
                compare_frame = temp_frame  # store the first bin for comparison
                norm_frame = pd.DataFrame(columns=[
                    'BinNumber', 'Timerange', 'Shapiro statistic', 'Shapiro p', 'Shapiro result',
                    'Kolmogorov-Smirnov statistic', 'Kolmogorov-Smirnov p', 'Kolmogorov-Smirnov result'])
                normality_frames[name] = norm_frame

            # Run normality tests
            stats_shapiro, p_value_shapiro, statistics_kolmogorov, p_value_kolmogorov = normality_tests(temp_frame.iloc[:, 2:], name)
            sr = 'not norm' if p_value_shapiro < alpha else 'norm'
            kr = 'not norm' if p_value_kolmogorov < alpha else 'norm'

            normality_frames[name].loc[len(normality_frames[name])] = pd.Series([
                i + 1, f"{min(temp_frame['Time'])}-{max(temp_frame['Time'])}", stats_shapiro, p_value_shapiro, sr,
                statistics_kolmogorov, p_value_kolmogorov, kr]).values

            # Run significance tests against first bin if enabled
            if ligand:
                if name not in significance_frames.keys():
                    sign_frame = pd.DataFrame(columns=[
                        'Compared Bins', 'Timerange',
                        'Mann-Whitney U statistic', 'Mann-Whitney U p', 'Mann-Whitney U result',
                        'Wilcoxon-signed-rank statistic', 'Wilcoxon p', 'Wilcoxon result'])
                    significance_frames[name] = sign_frame
                else:
                    statistics_t_test, p_value_t_test, statistics_wilcoxon, p_value_wilcoxon = significance_tests(
                        temp_frame.iloc[:, 2:], compare_frame, name)

                    # Determine significance annotations
                    if p_value_t_test == 'NaN':
                        tr = 'test not applicable'
                    elif p_value_t_test < p3:
                        tr = '***'
                    elif p_value_t_test < p2:
                        tr = '**'
                    elif p_value_t_test < p1:
                        tr = '*'
                    else:
                        tr = 'no significant difference'

                    if p_value_wilcoxon == 'NaN':
                        wr = 'test not applicable'
                    elif p_value_wilcoxon < p3:
                        wr = '***'
                    elif p_value_wilcoxon < p2:
                        wr = '**'
                    elif p_value_wilcoxon < p1:
                        wr = '*'
                    else:
                        wr = 'no significant difference'

                    significance_frames[name].loc[len(significance_frames[name])] = pd.Series([
                        f'1-{i}', f"{min(temp_frame['Time'])}-{max(temp_frame['Time'])}",
                        statistics_t_test, p_value_t_test, tr, statistics_wilcoxon, p_value_wilcoxon, wr]).values

    # Return both result sets if ligand was used, else only normality results
    return (normality_frames, significance_frames) if ligand else normality_frames


def normality_tests(dataframe, attribute):
    """
    Run two normality tests on a selected attribute.

    :param dataframe: Dataframe with relevant data
    :param attribute: Column name of the variable to test
    :return: Statistics and p-values from Shapiro and Kolmogorov-Smirnov tests
    """
    statistics_shapiro, p_value_shapiro = scy.shapiro(dataframe[attribute])
    statistics_kolmogorov, p_value_kolmogorov = scy.kstest(dataframe[attribute], 'norm', args=(dataframe[attribute].mean(), dataframe[attribute].std()))  # this was not the right test! test if it works now! TODO
    return statistics_shapiro, p_value_shapiro, statistics_kolmogorov, p_value_kolmogorov


def significance_tests(reference_frame, data_frame, attribute_to_compare):
    """
    Compares two distributions using the appropriate paired test (t-test or Wilcoxon),
    depending on whether data is normally distributed.

    :param reference_frame: Reference distribution (pandas DataFrame)
    :param data_frame:  Comparison distribution (pandas DataFrame)
    :param attribute_to_compare: Column name to compare, e.g. D_confined (str)
    :return: Test statistics and p-values for paired t-test and Wilcoxon test
    """
    if len(reference_frame) == len(data_frame):
        try:
            # paired t-test
            statistics_t_test, p_value_t_test = scy.ttest_rel(reference_frame[attribute_to_compare], data_frame[attribute_to_compare])
        except ValueError:
            statistics_t_test, p_value_t_test = 'not applicable to data', 'NaN'
        try:
            # wilcoxon signed-rank
            statistics_wilcoxon, p_value_wilcoxon = scy.wilcoxon(reference_frame[attribute_to_compare], data_frame[attribute_to_compare])
        except ValueError:
            statistics_wilcoxon, p_value_wilcoxon = 'not applicable to data', 'NaN'
    else:
        # For mismatched sample sizes, return explanation and NaNs
        statistics_t_test = statistics_wilcoxon = f'uneven sample size: {len(reference_frame)} vs {len(data_frame)}'
        p_value_t_test = p_value_wilcoxon = 'NaN'
    return statistics_t_test, p_value_t_test, statistics_wilcoxon, p_value_wilcoxon


def compile_columns(frames, columns, rename_cols):
    """
    Merges columns with matching names from multiple dataframes.

    :param frames: List of input dataframes
    :param columns: List of substrings to match in column names
    :param rename_cols: Whether to prefix columns with date to avoid duplication
    :return: A combined DataFrame of selected columns
    """
    compiled_frame = pd.DataFrame()
    comp_df = pd.DataFrame()

    for df in frames:
        # Extract identifier from filename (e.g., date and well position)
        date = df.iloc[0, 0].split('_')[0] + '_' + df.iloc[0, 0].split('_')[-3] + ': '
        # Select relevant columns
        common_columns = [column for column in df.columns if any(sub in column for sub in columns)]
        comp_df = df[common_columns]

        # Optionally rename columns to avoid duplicates
        if rename_cols:
            comp_df = comp_df.rename(columns={col: date + col for col in comp_df.columns})
        compiled_frame = pd.concat([compiled_frame, comp_df], axis=1)

    return comp_df if not rename_cols else compiled_frame


def generate_shortname(value):
    """
    Cleans up values for HDF5 writing by removing trailing colons.

    :param value: Value to clean
    :return: Sanitized string representation
    """
    if type(value) != str:
        return str(value)
    elif value == '':
        return 'None'
    elif value[-1] == '.':
        return value[:-1]
    else:
        return value


def cut_cell_names(name):
    """
    Extracts cell number from full identifier.

    :param name: Full name string (e.g., "sample_cell_42")
    :return: Extracted cell number (e.g., "42")
    """
    return name.split("_")[-1]


def output_folder(file, folder, datasets):
    """
    Writes datasets to a group in an HDF5 file.

    :param file: Open h5 py file handle
    :param folder: Name of group (folder) to write into
    :param datasets: List of [name, DataFrame] pairs to write
    """
    fold = file.create_group(folder)
    for i, dataset in enumerate(datasets):
        if folder != 'metadata':
            # Add units to column names
            dataset[1].columns = rename_columns(dataset[1].columns.tolist())
        # Sanitize each cell value for writing
        for col in dataset[1].columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset[1][col] = dataset[1][col].apply(generate_shortname)

        # Define variable-length string dtype for each column
        compound_dtype = np.dtype([(a, h5py.special_dtype(vlen=str)) for a in dataset[1].columns])
        tab = fold.create_dataset(dataset[0], (len(dataset[1]),), dtype=compound_dtype)

        # Write each column's data as a string array
        for col_index in range(dataset[1].shape[1]):
            data_array = np.array(dataset[1].iloc[:, col_index], dtype=compound_dtype)
            tab[dataset[1].columns[col_index]] = data_array


def rename_columns(old_col_names):
    """
    Appends units to column names based on their semantic label.

    :param old_col_names: List of original column names
    :return: List of renamed columns with units
    """
    new_col_names = []
    for col in old_col_names:
        if col.split(' ')[-1].startswith('Time'):
            new_col_names.append(col + ' [s]')
        elif col.split(' ')[-1].startswith('P'):
            new_col_names.append(col + ' [%]')
        elif col.split(' ')[-1].startswith('D'):
            new_col_names.append(col + ' [um^2s^-1]')
        elif col.split(' ')[-1].startswith('L'):
            new_col_names.append(col + ' [frames]')
        elif col.split(' ')[-1].startswith('confinement'):
            new_col_names.append(col + ' [um]')
        else:
            new_col_names.append(col)
    return new_col_names

# stopped annotating here TODO


def load_cs(sorted_list, coverslip, file, tif_files, coverslip_data):
    """
    Loads all cells of a coverslip into the variable coverslip_data.

    :param sorted_list: a correctly sorted list of all cell names
    :param coverslip: a list of cell names belonging to one coverslip
    :param file: list of h5 files
    :param tif_files: list of tif files (used for timestamp retrieval)
    :param coverslip_data: list of dataframes to which the new data will be appended
    :return: updated coverslip_data
    """

    cs_start_time = 0  # initialize start time

    for cell in coverslip:
        hdf5global = h5py.File(parent_string(cell, file, "metadata"), "r")

        # Get timestamp of first cell in coverslip
        try:
            float(coverslip_data[sorted_list.index(coverslip)].iloc[0, 1])
        except IndexError:
            cs_start_time = os.path.getmtime(parent_string(cell, tif_files, "metadata"))

        stats_global = hdf5global["statistics"]["statistics_3"][0, 0]

        # Probabilities of motion types
        p_immobile = float(stats_global[0])
        p_confined = float(stats_global[1])
        p_free = float(stats_global[2])

        # Diffusion coefficients (individual and global)
        dg_immobile = 0 if math.isnan(stats_global[5]) else float(stats_global[5])
        dg_confined = 0 if math.isnan(stats_global[6]) else float(stats_global[6])
        dg_free = 0 if math.isnan(stats_global[7]) else float(stats_global[7])
        d_global = (dg_immobile * p_immobile + dg_confined * p_confined + dg_free * p_free) * 0.01

        # Segment lengths (individual and global)
        lg_immobile = 0 if math.isnan(stats_global[11]) else float(stats_global[11])
        lg_confined = 0 if math.isnan(stats_global[12]) else float(stats_global[12])
        lg_free = 0 if math.isnan(stats_global[13]) else float(stats_global[13])
        l_global = (lg_immobile * p_immobile + lg_confined * p_confined + lg_free * p_free) * 0.01

        # Segment counts and confinement radius
        n_confined = 0
        n_free = 0
        n_immobile = 0
        confinement_radii_sum = 0

        de_immobile = float(stats_global[8])
        de_confined = float(stats_global[9])
        de_free = float(stats_global[10])
        le_immobile = float(stats_global[14])
        le_confined = float(stats_global[15])
        le_free = float(stats_global[16])

        for j in hdf5global["rossier"]["rossierStatistics"][()]:
            if j[2] == 1:
                n_confined += 1
                confinement_radii_sum += j[7]
            elif j[3] == 1:
                n_free += 1
            elif j[1] == 1 or j[4] == 1:
                n_immobile += 1

        confinement_radii = confinement_radii_sum / n_confined if n_confined > 0 else 0
        n_global = n_immobile + n_confined + n_free

        # Append data row
        coverslip_data[sorted_list.index(coverslip)].loc[
            len(coverslip_data[sorted_list.index(coverslip)])
        ] = (
            cell,
            os.path.getmtime(parent_string(cell, tif_files, 'metadata')) - cs_start_time,
            p_immobile, p_confined, p_free,
            d_global, dg_immobile, dg_confined, dg_free,
            l_global, lg_immobile, lg_confined, lg_free,
            n_global, n_immobile, n_confined, n_free,
            confinement_radii,
            de_immobile, de_confined, de_free,
            le_immobile, le_confined, le_free
        )

    # Raise error if DataFrame for this coverslip is empty
    if coverslip_data[sorted_list.index(coverslip)].empty:
        raise IncorrectConfigException(
             f'Coverslip number {sorted_list.index(coverslip) + 1} created an empty dataframe. Please check your data.'
        )

    return coverslip_data


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

    for folder in ['D', 'L', 'N']:
        # Determine attribute name
        if folder == 'D':
            name = 'diffusion_coefficients'
        elif folder == 'L':
            name = 'segment_lengths'
        elif folder == 'N':
            name = 'number_of_segments'
        else:
            raise ValueError(f"Unknown folder name: {folder}")

        data = []

        # Create output directories for plots
        if binned:
            for fmt in ['pngs', 'svgs', 'pdfs']:
                os.makedirs(os.path.join(save_dir, 'time_resolved_plots', name, fmt))

        # Loop over all signal types
        for signal_type in ['global', 'immobile', 'confined', 'free']:
            if binned:
                data.append(compile_columns(
                    dataset,
                    ["Cell Name", "Time", f"{folder}_{signal_type}",
                     f"{folder}_{signal_type}_SD", f"{folder}_{signal_type}_SEM"],
                    True
                ))

                # Generate plots depending on timestamp usage
                if use_timestamps:
                    plot = plot_by_time(coverslip_data, f"{folder}_{signal_type}", t_lig,
                                        ligand_name, dataset, error_type, plot_color)

                else:
                    plot = plot_by_cells(coverslip_data, f"{folder}_{signal_type}", t_lig,
                                         ligand_name, dataset, error_type, plot_color)

                for fmt in ['png', 'svg', 'pdf']:
                    plot.savefig(os.path.join(save_dir, 'time_resolved_plots', name, f"{fmt}s", f"{signal_type}.{fmt}"), dpi=300)

            else:
                data.append(compile_columns(
                    dataset,
                    ["Cell Name", "Time", f"{folder}_{signal_type}", f"{folder}E_{signal_type}"],
                    True
                ))

        # Write data into the HDF5 file
        output_folder(outputfile, name, [
            ['global', data[0]],
            ['immobile', data[1]],
            ['confined', data[2]],
            ['free', data[3]]
        ])


def stack_data(dataframes):
    """
    Stacks multiple DataFrames (from different coverslips) vertically into a single DataFrame.

    :param dataframes: list of dataframes
    :return: a single-element list containing one DataFrame with all input frames stacked
    """
    # Define the expected column names in the correct order
    columns = ["Cell Name", "Time", "P_immobile", "P_confined", "P_free", "D_global", "D_immobile", "D_confined",
               "D_free", "L_global", "L_immobile", "L_confined", "L_free", "N_global", "N_immobile", "N_confined",
               "N_free", 'confinement_radius', 'DE_immobile', 'DE_confined', 'DE_free', 'LE_immobile', 'LE_confined',
               'LE_free']

    # Attempt to rename columns of each DataFrame based on the defined header
    for df in dataframes:
        new_col = {}
        for i, col in enumerate(df.columns):
            new_col[col] = columns[i]
        df.rename(columns=new_col, inplace=True)

    # Concatenate all DataFrames vertically and return inside a list
    return [pd.concat(dataframes, axis=0)]


def main(config_path):
    start_time = time.time()
    cs_paths = []                # List of directories containing coverslip-specific data
    paths = []                  # List of directories containing global data
    files = []                  # List of .h5 files (will be filled later)
    use_timestamps = False      # Whether to use timestamps for time alignment
    tif_files = []

    # Load configuration file
    config = configparser.ConfigParser()
    config.sections()   # TODO: not necessary?
    config.read(config_path)

    # Check if timestamps should be used
    try:
        if config["USE_TIMESTAMPS"]["use_timestamps"].lower() == "true":
            use_timestamps = True
    except KeyError:
        raise IncorrectConfigException("Section USE_TIMESTAMPS missing in config.")

    # Load binning settings (how to group data by time or cell count)
    try:
        bin_size_cells = int(config["BIN_SIZE"]["cells"])
        bin_size_time = float(config["BIN_SIZE"]["minutes"]) * 60 + float(config["BIN_SIZE"]["seconds"])
    except KeyError:
        raise IncorrectConfigException("Section BIN_SIZE missing in config.")

    # Load all coverslip directories
    try:
        if len([key for key in config["CS_DIRS"]]):
            for key in config["CS_DIRS"]:
                cs_paths.append(config["CS_DIRS"][key])
        else:
            raise IncorrectConfigException("No coverslip directory defined in config.")
    except KeyError:
        raise IncorrectConfigException("Section CS_DIRS missing in config.")

    # Collect TIFF metadata files (needed for timestamps)
    for directory in cs_paths:
        tif_files += get_matching_files(directory + "\\cells\\tifs", "cell", ["_dl", 'metadata'])

    # Extract base names for each coverslip
    cs_names = []
    for cs in cs_paths:
        cs_names.append('_'.join(os.listdir(cs + "\\cells\\tifs")[0].split("\\")[-1].split('_')[:-2]))

    # Load global .h5 file paths
    try:
        if len([key for key in config["GLOBAL_DIR"]]):
            for key in config["GLOBAL_DIR"]:
                paths.append(config["GLOBAL_DIR"][key])
        else:
            raise IncorrectConfigException("No GLOBAL directory defined in config.")
    except KeyError:
        raise IncorrectConfigException("Section GLOBAL_DIR missing in config.")

    # Where to save the output plots and files
    try:
        save_dir = config["SAVE_DIR"]["save_dir"]
    except KeyError:
        raise IncorrectConfigException("Parameter save_dir missing in config.")

    # Load statistical threshold for significance (used for normalization)
    try:
        alpha = float(config["STAT_SETTINGS"]["alpha_norm"])
    except KeyError:
        raise IncorrectConfigException("Parameter alpha_norm missing in config.")

    # Determine whether statistical tests should be run
    try:
        if config["STAT_SETTINGS"]["run_stats"].lower() == "true":
            run_stats = True
        else:
            run_stats = False
    except KeyError:
        raise IncorrectConfigException("Section USE_TIMESTAMPS missing in config.")

    # Load significance levels for up to three comparison groups
    try:
        p1 = float(config["STAT_SETTINGS"]["alpha_sign_1"])
    except KeyError:
        raise IncorrectConfigException("Parameter alpha_sign_1 missing in config.")
    try:
        p2 = float(config["STAT_SETTINGS"]["alpha_sign_2"])
    except KeyError:
        raise IncorrectConfigException("Parameter alpha_sign_1 missing in config.")
    try:
        p3 = float(config["STAT_SETTINGS"]["alpha_sign_3"])
    except KeyError:
        raise IncorrectConfigException("Parameter alpha_sign_1 missing in config.")

    # Get dot color for plots; fallback to orange if empty
    try:
        plot_color = config["PLOT_SETTINGS"]["dot_color"]
        if plot_color == '':
            plot_color = '#FFA500'
    except KeyError:
        raise IncorrectConfigException("Parameter dot_color missing in config.")

    # Determine ligand addition time index and whether to use it
    try:
        t_lig = config["PLOT_SETTINGS"]["ligand_index"]
        if t_lig == '':
            ligand = False
        else:
            ligand = True
            # Add identifier for timestamped ('s') or cell-based ('c') data
            if use_timestamps:
                t_lig += 's'
            else:
                t_lig += 'c'
    except KeyError:
        raise IncorrectConfigException("Parameter t_lig missing in config.")

    # Load ligand name for use in plot labeling
    try:
        ligand_name = config["PLOT_SETTINGS"]["ligand_name"]
    except KeyError:
        raise IncorrectConfigException("Parameter ligand_name missing in config.")

    # Load error type (e.g., SD or SEM)
    try:
        error_type = config["PLOT_SETTINGS"]["error_type"]
    except KeyError:
        raise IncorrectConfigException("Parameter error_type missing in config.")

    # Check each path in the config; collect matching .h5 files unless entry is a float (indicates an error)
    for path in paths:
        if type(path) is float:
            raise IncorrectConfigException("mismatched number of files")
        else:
            files = get_matching_files(path, ".h5", ["statistics.h5"])

    # Prepare the output directory structure
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass

    # (Re)create subfolder for time-resolved analysis output
    try:
        os.mkdir(save_dir + '\\timeResolvedAnalysis')
    except FileExistsError:
        shutil.rmtree(save_dir + '\\timeResolvedAnalysis')
        os.mkdir(save_dir + '\\timeResolvedAnalysis')

    save_dir += '\\timeResolvedAnalysis'

    # Match each .h5 file to its corresponding .tif file by coverslip name
    input_files = {c: [] for c in cs_names}
    for h5 in files:
        filename = h5.split("\\")[-1][:-2]
        tif = parent_string(filename, tif_files,
                            "metadata")  # find matching .tif file based on metadata
        coverslip_name = '_'.join(tif.split("\\")[-1].split('_')[:-2]) # extract coverslip identifier
        input_files[coverslip_name].append(filename)

    # Sort files naturally (e.g., cell_2 comes before cell_10)
    sorted_list = []
    for cs in input_files.keys():
        sorted_list.append(sort_cells(input_files[cs]))

    # Prepare empty dataframes for each coverslip to store time-resolved statistics
    coverslip_data = [pd.DataFrame(
        columns=["Cell Name", "Time", "P_immobile", "P_confined", "P_free", "D_global", "D_immobile", "D_confined",
                 "D_free", "L_global", "L_immobile", "L_confined", "L_free", "N_global", "N_immobile", "N_confined",
                 "N_free", "confinement_radius", "DE_immobile", "DE_confined", "DE_free", "LE_immobile", "LE_confined",
                 "LE_free"]) for cs in cs_names]

    # Process and bin data for each coverslip; optionally use time-based or cell-based binning
    binned_data = []
    for coverslip in sorted_list:  # coverslip is a sorted list of all cells in a coverslip
        coverslip_data = load_cs(sorted_list, coverslip, files, tif_files, coverslip_data)
        if use_timestamps:
            binned_mean = bin_data_time(coverslip_data[sorted_list.index(coverslip)], bin_size_time)
        else:
            binned_mean = bin_data_cells(coverslip_data[sorted_list.index(coverslip)], bin_size_cells)
        binned_data += [binned_mean]

    # Stack all coverslip data into one dataframe
    stacked_data = stack_data(coverslip_data)

    # Find the index of the longest binned dataset to use for global averaging
    largest_bindex = 0
    for i, current_bin in enumerate(binned_data):
        if len(current_bin) > len(binned_data[largest_bindex]):
            largest_bindex = i

    # Calculate global mean across all coverslips for each parameter
    global_mean = pd.DataFrame(binned_data[largest_bindex].iloc[:, 0:2]) # base columns: Cell Name and Time
    for attribute in ["P_immobile", "P_confined", "P_free", "D_global", "D_immobile", "D_confined",
                      "D_free", "L_global", "L_immobile", "L_confined", "L_free", "N_global", "N_immobile",
                      "N_confined",
                      "N_free", "confinement_radius"]:
        global_mean = pd.concat([global_mean, pd.DataFrame(calc_mean_over_cs(binned_data, attribute),
                                                           columns=[attribute, attribute + "_SD", attribute + "_SEM"])],
                                axis=1)

    # Clean up cell names to just the range identifier
    for i, row in enumerate(global_mean["Cell Name"]):
        global_mean.iloc[i, 0] = row.split("_")[-1]
    global_mean.rename(columns={'Cell Names': 'cell range'}) # TODO: likely intended to rename "Cell Name"

    # Create output file structure and save results
    output_file = h5py.File(save_dir + '\\stats.h5', 'w')
    output_file_bin = output_file.create_group('bin')
    output_file_raw = output_file.create_group('raw')
    output_file_stacked = output_file.create_group('stacked')

    # Save global means
    output_folder(output_file, 'global means', [['global means', global_mean]])

    # Save raw fractions data
    output_folder(output_file_raw, 'fractions',
                  [['immobile', compile_columns(coverslip_data, ["Cell Name", "Time", "P_immobile"], True)],
                   ['confined', compile_columns(coverslip_data, ["Cell Name", "Time", "P_confined"], True)],
                   ['free', compile_columns(coverslip_data, ["Cell Name", "Time", "P_free"], True)]])

    # Save binned fractions data
    output_folder(output_file_bin, 'fractions',
                  [['immobile', compile_columns(binned_data, ["Cell Name", "Time", "P_immobile"], True)],
                   ['confined', compile_columns(binned_data, ["Cell Name", "Time", "P_confined"], True)],
                   ['free', compile_columns(binned_data, ["Cell Name", "Time", "P_free"], True)]])

    # Save stacked fractions data
    output_folder(output_file_stacked, 'fractions',
                  [['immobile', compile_columns(stacked_data, ["Cell Name", "Time", "P_immobile"], False)],
                   ['confined', compile_columns(stacked_data, ["Cell Name", "Time", "P_confined"], False)],
                   ['free', compile_columns(stacked_data, ["Cell Name", "Time", "P_free"], False)]])

    # Prepare folder structure for saving plots
    name = 'fractions'
    os.mkdir(save_dir + '\\time_resolved_plots')
    os.mkdir(save_dir + '\\time_resolved_plots\\' + name)
    os.mkdir(save_dir + '\\time_resolved_plots\\' + name + '\\pngs')
    os.mkdir(save_dir + '\\time_resolved_plots\\' + name + '\\svgs')
    os.mkdir(save_dir + '\\time_resolved_plots\\' + name + '\\pdfs')

    # Generate and save time-resolved plots (for each population type)
    for signal_type in ['immobile', 'confined', 'free']:
        if use_timestamps:
            plot = plot_by_time(coverslip_data, 'P_' + signal_type, t_lig, ligand_name, binned_data, error_type, plot_color)
        else:
            plot = plot_by_cells(coverslip_data, 'P_' + signal_type, t_lig, ligand_name, binned_data, error_type, plot_color)
        plot.savefig(save_dir + '\\time_resolved_plots\\' + name + '\\pngs\\' + signal_type + '.png', dpi=300)
        plot.savefig(save_dir + '\\time_resolved_plots\\' + name + '\\svgs\\' + signal_type + '.svg', dpi=300)
        plot.savefig(save_dir + '\\time_resolved_plots\\' + name + '\\pdfs\\' + signal_type + '.pdf', dpi=300)

    # Save raw, binned, and stacked confinement radius data
    output_folder(output_file_raw, 'confinement_radii',
                  [['confinement_radii', compile_columns(coverslip_data, ["Cell Name", "Time", 'confinement_radius'], True)]])
    output_folder(output_file_bin, 'confinement_radii',
                  [['confinement_radii', compile_columns(binned_data, ["Cell Name", "Time", 'confinement_radius'], True)]])
    output_folder(output_file_stacked, 'confinement_radii',
                  [['confinement_radii', compile_columns(stacked_data, ["Cell Name", "Time", 'confinement_radius'], False)]])

    # Create folder and generate confinement radius plots
    name = 'confinement_radii'
    os.mkdir(save_dir + '\\time_resolved_plots\\' + name)
    if use_timestamps:
        plot = plot_by_time(coverslip_data, 'confinement_radius', t_lig, ligand_name, binned_data, error_type, plot_color)
    else:
        plot = plot_by_cells(coverslip_data, 'confinement_radius', t_lig, ligand_name, binned_data, error_type, plot_color)

    # Save plot in multiple formats
    plot.savefig(save_dir + '\\time_resolved_plots\\' + name + '\\confinement_radii.png', dpi=300)
    plot.savefig(save_dir + '\\time_resolved_plots\\' + name + '\\confinement_radii.svg', dpi=300)
    plot.savefig(save_dir + '\\time_resolved_plots\\' + name + '\\confinement_radii.pdf', dpi=300)

    # Save outputs for diffusion coefficients, segment lengths, and number of segments
    # raw, binned, and stacked
    four_set_output(save_dir, output_file_raw, coverslip_data, coverslip_data, use_timestamps, plot_color, t_lig, ligand_name, error_type, False)
    four_set_output(save_dir, output_file_bin, binned_data, coverslip_data, use_timestamps, plot_color, t_lig, ligand_name, error_type, True)
    four_set_output(save_dir, output_file_stacked, stacked_data, coverslip_data, use_timestamps, plot_color, t_lig, ligand_name, error_type, False)

    # Prepare metadata DataFrame with parameters used in the analysis
    metadata = pd.DataFrame(
        columns=['use_timestamps', 'n_cells', 'Plot_Error_type', 'Plot_Ligand_time', 'stat_alpha', 'stat_p1', 'stat_p2', 'stat_p3'])
    metadata.loc[len(metadata)] = pd.Series(
        [use_timestamps, str(int(bin_size_time / 60)) + 'm' + str(bin_size_time % 60) + 's', error_type, t_lig, alpha, p1, p2, p3]).values

    if use_timestamps:
        metadata = pd.DataFrame(columns=['use_timestamps', 'n_cells', 'Plot_Error_type', 'Plot_Ligand_time',
                                         'stat_alpha', 'stat_p1', 'stat_p2', 'stat_p3'])
        metadata.loc[len(metadata)] = pd.Series(
            [use_timestamps, str(int(bin_size_time / 60)) + 'm' + str(bin_size_time % 60) + 's', error_type, t_lig,
             alpha, p1, p2, p3]).values
    else:
        metadata = pd.DataFrame(
            columns=['use_timestamps', 'n_cells', 'Plot_Error_type', 'Plot_Ligand_index', 'stat_alpha',
                     'stat_p1', 'stat_p2', 'stat_p3'])
        metadata.loc[len(metadata)] = pd.Series(
            [use_timestamps, str(bin_size_cells), error_type, t_lig, alpha, p1, p2, p3]).values

    # Save metadata
    output_folder(output_file, 'metadata',[['metadata', metadata]])
    try:
        output_folder(output_file, 'metadata', [['metadata', metadata]])
    except ValueError:
        pass

    # Finalize and close output file
    output_file.close()

    # Run statistical tests if required
    if use_timestamps:
        run_stats = False  # override if timestamps are used

    if run_stats:
        # Create folders for test results
        os.mkdir(save_dir + '\\tests')
        os.mkdir(save_dir + '\\tests\\normality')

        # Run normality (and optionally significance) tests
        if use_timestamps:
            if ligand:
                norm_frames, sign_frames = test_by_time(coverslip_data, bin_size_time, ligand, alpha, p1, p2, p3)
            else:
                norm_frames = test_by_time(coverslip_data, bin_size_time, ligand, alpha, p1, p2, p3)
        else:
            if ligand:
                norm_frames, sign_frames = test_by_cell(coverslip_data, bin_size_cells, ligand, alpha, p1, p2, p3)
            else:
                norm_frames = test_by_cell(coverslip_data, bin_size_cells, ligand, alpha, p1, p2, p3)

        # Save normality test results
        for key in norm_frames.keys():
            key2 = key
            if key.split('_')[0] == 'P':
                name = 'fractions'
            elif key.split('_')[0] == 'D':
                name = 'diffusion_coefficients'
            elif key.split('_')[0] == 'L':
                name = 'segment_lengths'
            elif key.split('_')[0] == 'N':
                name = 'number_of_segments'
            elif key.split('_')[0] == 'confinement':
                name = 'confinement_radius'
                key2 = ""
            else:
                continue
            try:
                os.mkdir(save_dir + '\\tests\\normality\\' + name)
            except FileExistsError:
                pass
            norm_frames[key].to_csv(
                save_dir + '\\tests\\normality\\' + name + '\\test_normality_' + name + "_" + key2.split('_')[
                    -1] + '.csv', index=False)

        # Save significance test results (if ligand present)
        if ligand:
            os.mkdir(save_dir + '\\tests\\significance')
            for key in sign_frames.keys():
                key2 = key
                if key.split('_')[0] == 'P':
                    name = 'fractions'
                elif key.split('_')[0] == 'D':
                    name = 'diffusion_coefficients'
                elif key.split('_')[0] == 'L':
                    name = 'segment_lengths'
                elif key.split('_')[0] == 'N':
                    name = 'number_of_segments'
                elif key.split('_')[0] == 'confinement':
                    name = 'confinement_radius'
                    key2 = ""
                else:
                    continue
                try:
                    os.mkdir(save_dir + '\\tests\\significance\\' + name)
                except FileExistsError:
                    pass
                if key == key2:
                    sign_frames[key].to_csv((
                            save_dir + '\\tests\\significance\\' + name + '\\test_significance_' + name + "_" +
                            key2.split('_')[-1] + '.csv'), index=False)
                else:
                    sign_frames[key].to_csv((save_dir + '\\tests\\significance\\' + name + '\\test_significance_' + name + '.csv'), index=False)

    # Print execution time
    print("--- %s seconds ---" % (time.time() - start_time))

# Entry point for script execution
if __name__ == "__main__":
    try:
        cfg_path = sys.argv[1]
        main(cfg_path)
    except FileExistsError:
        print("Usage: python timeResolvedAnalysis.py your_config_file.ini")
