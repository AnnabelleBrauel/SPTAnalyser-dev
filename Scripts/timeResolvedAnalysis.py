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
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if target.lower() in name.lower():
                if not any([True for string in exclusion_string if string.lower() in name.lower()]):
                    matching_files.append(os.path.join(path, name))
    return matching_files


def parent_string(substring, parents, exclusionstr):
    """
    Searches a list of strings ('parents') for entries that contain a specific substring
    but do NOT contain a given exclusion string.

    :param substring: Substring to search for
    :param parents: List of potential parent strings
    :param exclusionstr: List of strings that must not be present in the parent string
    :return: A single matching string (raises exception if ambiguous or not found)
    """
    parent = [a for a in parents if
              (substring in a and exclusionstr not in a)]
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
    sorted = []
    for dig in digits:
        dig.sort()
        sorted += dig
    return sorted


def insert_error(meanframe, sdframe, semframe):
    """
    Inserts standard deviation (SD) and standard error of the mean (SEM) columns
    into a dataframe containing mean values, directly after their corresponding mean column.

    :param meanframe: Dataframe with mean values
    :param sdframe: Dataframe with standard deviations
    :param semframe: Dataframe with standard errors of the mean
    :return: Modified dataframe with SD and SEM inserted
    """
    # Remove metadata columns (assumes first two are not data columns)
    sdframe = sdframe.iloc[:, 2:]
    semframe = semframe.iloc[:, 2:]
    i = 0
    while i < sdframe.shape[1]:
        name = sdframe.columns[i]
        sd = sdframe.iloc[:, i]
        sem = semframe.iloc[:, i]
        # Rename error columns to indicate type
        sd.rename(name + '_SD', inplace=True)
        sem.rename(name + '_SEM', inplace=True)
        # Insert error columns right after the mean column
        meanframe.insert(i * 3 + 3, name + '_SD', sd)
        meanframe.insert(i * 3 + 4, name + '_SEM', sem)
        i += 1
    return meanframe


def calc_mean_over_cS(binned_data, attribute):
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

    :param dataframes: List of raw datafames to be plotted (dots)
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
    stats = calc_mean_over_cS(binned_data, attribute)

    left = 0
    longestdf = ''
    # Determine the longest dataframe for time bin labels
    for df in binned_data:
        if len(df) > len(longestdf):
            longestdf = df

    # Plot horizontal bars for mean values with error shading
    for i, row in enumerate(stats):
        if error_type == 'SEM':
            error = row[2]
        else:
            error = row[1]

        # Calculate right edge of the time bin (convert from seconds to minutes)
        if i == len(stats) - 1:
            right = float(longestdf.iloc[i, 1].split('-')[1]) / 60
        else:
            right = (((float(longestdf.iloc[i + 1, 1].split('-')[0]) - float(
                longestdf.iloc[i, 1].split('-')[1])) / 2) + float(longestdf.iloc[i, 1].split('-')[1])) / 60

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
    stats = calc_mean_over_cS(binned_data, attribute)
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
    meanframe = pd.DataFrame(columns=frame.columns)
    sdframe = pd.DataFrame(columns=frame.columns)
    semframe = pd.DataFrame(columns=frame.columns)

    # Derive base name for bin labels (e.g., CS2_P3_)
    cs_name = frame.iloc[0, 0].split('_')[0] + '_' + frame.iloc[0, 0].split('_')[-3] + '_'
    binnumber = 0
    rowindex = 0
    complete = False

    # Iterate through the dataframe and collect bins
    while not complete:
        start = rowindex
        bin = pd.DataFrame(columns=frame.columns)
        # Collect rows within the current bin range
        while float(frame.iloc[rowindex, 0].split('_')[-1]) <= binnumber * bin_size:
            bin.loc[len(bin)] = frame.iloc[rowindex, :]
            rowindex += 1
            if rowindex == len(frame):  # End of DataFrame reached
                complete = True
                break
        # Skip empty bins (e.g., if no data points fall into this time range)
        if len(bin) == 0:
            binnumber += 1
            continue

        # Calculate and label mean per bin
        meanframe.loc[len(meanframe)] = bin.iloc[:, 2:].mean()
        meanframe.iloc[-1, 0] = cs_name + "cell_" + str((binnumber - 1) * bin_size) + "-" + str(binnumber * bin_size)
        meanframe.iloc[-1, 1] = str(int(bin.iloc[0, 1])) + "-" + str(int(bin.iloc[-1, 1]))

        # Calculate and label standard deviation per bin
        sdframe.loc[len(sdframe)] = bin.iloc[:, 2:].std()
        sdframe.iloc[-1, 0] = cs_name + "cell_" + str((binnumber - 1) * bin_size) + "-" + str(binnumber * bin_size)
        sdframe.iloc[-1, 1] = str(int(bin.iloc[0, 1])) + "-" + str(int(bin.iloc[-1, 1]))

        # Calculate and label standard error per bin
        semframe.loc[len(semframe)] = bin.iloc[:, 2:].sem()
        semframe.iloc[-1, 0] = cs_name + "cell_" + str((binnumber - 1) * bin_size) + "-" + str(binnumber * bin_size)
        semframe.iloc[-1, 1] = str(int(bin.iloc[0, 1])) + "-" + str(int(bin.iloc[-1, 1]))

    binnumber += 1  # Shouldn't this have one more tab? TODO

    # Merge all results and insert error columns
    outframe = insert_error(meanframe, sdframe, semframe)
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
    meanframe = pd.DataFrame(columns=frame.columns)
    sdframe = pd.DataFrame(columns=frame.columns)
    semframe = pd.DataFrame(columns=frame.columns)

    # Extract coverslip and condition name from the first entry for naming consistency
    cs_name = frame.iloc[0, 0].split('_')[0] + '_' + frame.iloc[0, 0].split('_')[-3] + '_'

    complete = False
    binnumber = 1
    rowindex = 0

    while not complete:
        start = rowindex
        bin = pd.DataFrame(columns=frame.columns)  # temporary dataframe for one bin

        # Accumulate rows into the current bin based on the time column
        while frame.iloc[
            rowindex, 1] < binnumber * bin_size:
            bin.loc[len(bin)] = frame.iloc[rowindex, :]
            rowindex += 1
            if rowindex == len(frame):
                complete = True
                break
        # Skip empty bins (e.g., if no data points fall into this time range)
        if len(bin) == 0:
            binnumber += 1
            continue

        # Calculate mean of all numeric columns and append to result dataframe
        meanframe.loc[len(meanframe)] = bin.iloc[:, 2:].mean()
        meanframe.iloc[-1, 0] = cs_name + "cell_" + str(start) + "-" + str(start + len(bin))

        # Calculate standard deviation and append to result dataframe
        sdframe.loc[len(sdframe)] = bin.iloc[:, 2:].std()
        sdframe.iloc[-1, 0] = cs_name + "cell_" + str(start) + "-" + str(start + len(bin))

        # Calculate standard error of the mean and append
        semframe.loc[len(semframe)] = bin.iloc[:, 2:].sem()
        semframe.iloc[-1, 0] = cs_name + "cell_" + str(start) + "-" + str(start + len(bin))

        # Add time interval label to column index 1
        if len(bin) > 0:
            meanframe.iloc[-1, 1] = str((binnumber - 1) * bin_size) + "-" + str((binnumber) * bin_size)
            sdframe.iloc[-1, 1] = str((binnumber - 1) * bin_size) + "-" + str((binnumber) * bin_size)
            semframe.iloc[-1, 1] = str((binnumber - 1) * bin_size) + "-" + str((binnumber) * bin_size)

        binnumber += 1

    # Insert SD and SEM into the meanframe before returning
    outframe = insert_error(meanframe, sdframe, semframe)
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
    if ligand:
        significance_frames = {}  # stores results of significance tests if enabled

    cell_count=[]

    # Determine number of bins based on cell count in first frame
    bin_number = int(len(frames[0]) / bin_size)
    if len(frames[0]) % bin_size != 0:
        bin_number += 1

    for j in range(bin_number):
        tempFrame = pd.DataFrame()  # accumulates all data for current bin

        # Collect all rows from each frame that fall into the current cell bin
        for frame in frames:
            binframe = pd.DataFrame(columns=tempFrame.columns)
            for index,row in frame.iterrows():
                cell_id = float(row[0].split('_')[-1])
                if j * bin_size < cell_id <= (j + 1) * bin_size:
                    binframe = binframe.append(row, ignore_index=True)
            tempFrame = pd.concat([tempFrame, binframe], axis=0)

        # Record number of valid cells in this bin
        try:
            cell_count[j]+= len(tempFrame)
        except IndexError:
            cell_count.append(len(tempFrame))

        # Skip bins with fewer than 3 cells
        if len(tempFrame) < 3:
            for name in tempFrame.columns[2:]:
                normality_frames[name].loc[len(normality_frames[name])] = pd.Series(
                    [j + 1, str(j * bin_size + 1) + '-' + str((j + 1) * bin_size), 'NaN', 'NaN',
                    f'Dataset too small ({len(tempFrame)})', 'NaN', 'NaN',
                    f'Dataset too small ({len(tempFrame)})']).values
                if ligand:
                    significance_frames[name].loc[len(significance_frames[name])] = pd.Series(
                        [f'1-{j + 1}', f"{j * bin_size + 1}-{(j + 1) * bin_size}", 'NaN', 'NaN',
                        f'Dataset too small ({len(tempFrame)})', 'NaN', 'NaN',
                        f'Dataset too small ({len(tempFrame)})']).values
            continue


        for name in tempFrame.columns[2:]:  # run tests on all measurement columns
            tempFrame2 = tempFrame.dropna(subset=[name])  # drop NaNs for this column

            # Create new result table for this attribute if needed
            if name not in normality_frames.keys():
                compare_frame = tempFrame2
                norm_frame = pd.DataFrame(columns=[
                    'BinNumber', 'Cellrange', 'number of cells','Shapiro statistic', 'Shapiro p', 'Shapiro result',
                    'Kolmogorov-Smirnov statistic', 'Kolmogorov-Smirnov p', 'Kolmogorov-Smirnov result'])
                normality_frames[name] = norm_frame

            # Run normality tests (Shapiro-Wilk and Kolmogorov-Smirnov)
            shapstat, ps_value, kolstat, pk_value = normality_tests(tempFrame2, name)
            sr = 'not norm' if ps_value < alpha else 'norm'
            kr = 'not norm' if pk_value < alpha else 'norm'

            normality_frames[name].loc[len(normality_frames[name])] = pd.Series([
                j + 1, f"{j * bin_size + 1}-{(j + 1) * bin_size}",
                f"{len(tempFrame2)} {cell_count[j] - len(tempFrame2)} were dropped due to NaN entries",
                shapstat, ps_value, sr,
                kolstat, pk_value, kr
            ]).values

            # If ligand is used, compare this bin to the first one using significance tests
            if ligand:
                if name not in significance_frames.keys():
                    sign_frame = pd.DataFrame(columns=[
                        'Compared Bins', 'Cellrange', 'number of cells (control: '+str(cell_count)+' cells)',
                        'paired tTest statistic', 'paired tTest p', 'paired tTest result',
                        'Wilcoxon-signed-rank statistic', 'Wilcoxon p', 'Wilcoxon result'])
                    significance_frames[name] = sign_frame
                else:
                    mstat, pm_value, wilstat, pw_value = significance_tests(tempFrame2, compare_frame, name)

                    # Determine significance levels for each test
                    if pm_value == 'NaN':
                        tr = 'test not applicable'
                    elif pm_value < p3:
                        tr = '***'
                    elif pm_value < p2:
                        tr = '**'
                    elif pm_value < p1:
                        tr = '*'
                    else:
                        tr = 'no significant difference'

                    if pw_value == 'NaN':
                        wr = 'test not applicable'
                    elif pw_value < p3:
                        wr = '***'
                    elif pw_value < p2:
                        wr = '**'
                    elif pw_value < p1:
                        wr = '*'
                    else:
                        wr = 'no significant difference'

                    significance_frames[name].loc[len(significance_frames[name])] = pd.Series([
                        f'1-{j + 1}', f"{j * bin_size + 1}-{(j + 1) * bin_size}",
                        f"{len(tempFrame2)} {cell_count[j] - len(tempFrame2)} were dropped due to NaN entries",
                        mstat, pm_value, tr,
                        wilstat, pw_value, wr
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
    if ligand:
        significance_frames = {}  # store significance test results if enabled

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
        tempFrame = pd.DataFrame()

        # Collect all rows that fall into the current time bin
        for frame in frames:
            rowindex = 0
            while frame.iloc[rowindex, 1] < i * bin_size:
                if frame.iloc[rowindex, 1] >= (i - 1) * bin_size:
                    tempFrame = tempFrame.append(frame.iloc[rowindex, :])
                rowindex += 1
                if rowindex == len(frame):
                    break

        # Skip bins with insufficient data
        if len(tempFrame) < 3:
            for name in tempFrame.columns[2:]:
                normality_frames[name].loc[len(normality_frames[name])] = pd.Series(
                    [i + 1, f"{min(tempFrame['Time'])}-{max(tempFrame['Time'])}", 'NaN', 'NaN',
                     'Dataset too small', 'NaN', 'NaN', 'Dataset too small']).values
                if ligand:
                    significance_frames[name].loc[len(significance_frames[name])] = pd.Series(
                        [f'1-{i}', f"{min(tempFrame['Time'])}-{max(tempFrame['Time'])}", 'NaN', 'NaN',
                         'Dataset too small', 'NaN', 'NaN', 'Dataset too small']).values
            continue

        for name in tempFrame.columns[2:]:  # analyze each measurement column
            if name not in normality_frames.keys():
                compare_frame = tempFrame  # store the first bin for comparison
                norm_frame = pd.DataFrame(columns=[
                    'BinNumber', 'Timerange', 'Shapiro statistic', 'Shapiro p', 'Shapiro result',
                    'Kolmogorov-Smirnov statistic', 'Kolmogorov-Smirnov p', 'Kolmogorov-Smirnov result'])
                normality_frames[name] = norm_frame

            # Run normality tests
            shapstat, ps_value, kolstat, pk_value = normality_tests(tempFrame.iloc[:, 2:], name)
            sr = 'not norm' if ps_value < alpha else 'norm'
            kr = 'not norm' if pk_value < alpha else 'norm'

            normality_frames[name].loc[len(normality_frames[name])] = pd.Series([
                i + 1, f"{min(tempFrame['Time'])}-{max(tempFrame['Time'])}", shapstat, ps_value, sr,
                kolstat, pk_value, kr]).values

            # Run significance tests against first bin if enabled
            if ligand:
                if name not in significance_frames.keys():
                    sign_frame = pd.DataFrame(columns=[
                        'Compared Bins', 'Timerange',
                        'Mann-Whitney U statistic', 'Mann-Whitney U p', 'Mann-Whitney U result',
                        'Wilcoxon-signed-rank statistic', 'Wilcoxon p', 'Wilcoxon result'])
                    significance_frames[name] = sign_frame
                else:
                    tstat, pt_value, wilstat, pw_value = significance_tests(
                        tempFrame.iloc[:, 2:], compare_frame, name)

                    # Determine significance annotations
                    if pt_value == 'NaN':
                        tr = 'test not applicable'
                    elif pt_value < p3:
                        tr = '***'
                    elif pt_value < p2:
                        tr = '**'
                    elif pt_value < p1:
                        tr = '*'
                    else:
                        tr = 'no significant difference'

                    if pw_value == 'NaN':
                        wr = 'test not applicable'
                    elif pw_value < p3:
                        wr = '***'
                    elif pw_value < p2:
                        wr = '**'
                    elif pw_value < p1:
                        wr = '*'
                    else:
                        wr = 'no significant difference'

                    significance_frames[name].loc[len(significance_frames[name])] = pd.Series([
                        f'1-{i}', f"{min(tempFrame['Time'])}-{max(tempFrame['Time'])}",
                        tstat, pt_value, tr, wilstat, pw_value, wr]).values

    # Return both result sets if ligand was used, else only normality results
    return (normality_frames, significance_frames) if ligand else normality_frames


def normality_tests(dataframe, attribute):
    """
    Run two normality tests on a selected attribute.

    :param dataframe: Dataframe with relevant data
    :param attribute: Column name of the variable to test
    :return: Statistics and p-values from Shapiro and Kolmogorov-Smirnov tests
    """
    shapstat, ps_value = scy.shapiro(dataframe[attribute])
    kolstat, pk_value = scy.kstest(dataframe[attribute], 'norm', args=(dataframe[attribute].mean(), dataframe[attribute].std()))  # this was not the right test! test if it works now! TODO
    return shapstat, ps_value, kolstat, pk_value


def significance_tests(frame1, frame2, attribute):
    """
    Compares two distributions using paired tests.

    :param frame1: Reference distribution
    :param frame2: Comparison distribution
    :param attribute: Column to compare
    :return: Test statistics and p-values for paired t-test and Wilcoxon test
    """
    if len(frame1) == len(frame2):
        try:
            mstat, pm_value = scy.ttest_rel(frame1[attribute], frame2[attribute])  # paired t-test
        except ValueError:
            mstat, pm_value = 'not applicable to data', 'NaN'
        try:
            wilstat, pw_value = scy.wilcoxon(frame1[attribute], frame2[attribute])  # Wilcoxon signed-rank
        except ValueError:
            wilstat, pw_value = 'not applicable to data', 'NaN'
    else:
        # For mismatched sample sizes, return explanation and NaNs
        mstat = wilstat = f'uneven sample size: {len(frame1)} vs {len(frame2)}'
        pm_value = pw_value = 'NaN'
    return mstat, pm_value, wilstat, pw_value


def compile_columns(frames, columns, renameColumns):
    """
    Merges columns with matching names from multiple dataframes.

    :param frames: List of input dataframes
    :param columns: List of substrings to match in column names
    :param renameColumns: Whether to prefix columns with date to avoid duplication
    :return: A combined DataFrame of selected columns
    """
    compiled_frame = pd.DataFrame()
    for df in frames:
        # Extract identifier from filename (e.g., date and well position)
        date = df.iloc[0, 0].split('_')[0] + '_' + df.iloc[0, 0].split('_')[-3] + ': '
        # Select relevant columns
        common_columns = [column for column in df.columns if any(sub in column for sub in columns)]
        comp_df = df[common_columns]

        # Optionally rename columns to avoid duplicates
        if renameColumns:
            comp_df = comp_df.rename(columns={col: date + col for col in comp_df.columns})
        compiled_frame = pd.concat([compiled_frame, comp_df], axis=1)

    return comp_df if not renameColumns else compiled_frame


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


def cut_cellnames(name):
    """
    Extracts cell number from full identifier.

    :param name: Full name string (e.g., "sample_cell_42")
    :return: Extracted cell number (e.g., "42")
    """
    return name.split("_")[-1]


def output_folder(file, folder, datasets):
    """
    Writes datasets to a group in an HDF5 file.

    :param file: Open h5py file handle
    :param folder: Name of group (folder) to write into
    :param datasets: List of [name, DataFrame] pairs to write
    """
    fold = file.create_group(folder)
    for i, set in enumerate(datasets):
        if folder != 'metadata':
            # Add units to column names
            set[1].columns = rename_columns(set[1].columns.tolist())
        # Sanitize each cell value for writing
        for col in set[1].columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                set[1][col] = set[1][col].apply(generate_shortname)

        # Define variable-length string dtype for each column
        compound_dtype = np.dtype([(a, h5py.special_dtype(vlen=str)) for a in set[1].columns])
        tab = fold.create_dataset(set[0], (len(set[1]),), dtype=compound_dtype)

        # Write each column's data as a string array
        for i in range(set[1].shape[1]):
            data_array = np.array(set[1].iloc[:, i], dtype=compound_dtype)
            tab[set[1].columns[i]] = data_array


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


def load_cs(sorted, coverslip, file, tiffiles, coverslip_data):
    """
    loads the data of a coverslip into the coverslip_data variable
    :param sorted: a correctly sorted list of cell names
    :param coverslip: a correctly sorted list of cell names in a coverslip
    :param file: list of h5 files
    :param tiffiles: list of the tif files for the timestamps
    :param coverslip_data: list of already loaded data to add to
    :return:coverslip_data with the data added to it
    """
    for cell in coverslip:
        hdf5global = h5py.File(parent_string(cell, file, "metadata"), "r")
        try:
            float(coverslip_data[sorted.index(coverslip)].iloc[0, 1])
        except IndexError:
            csStartTime = os.path.getmtime(parent_string(cell, tiffiles, "metadata"))
        statglobal = hdf5global["statistics"]["statistics_3"][0, 0]
        P_immobile = float(statglobal[0])
        P_confined = float(statglobal[1])
        P_free = float(statglobal[2])
        D_immobile = float(statglobal[5])
        if math.isnan(D_immobile):
            DG_immobile = 0
        else:
            DG_immobile = D_immobile
        D_confined = float(statglobal[6])
        if math.isnan(D_confined):
            DG_confined = 0
        else:
            DG_confined = D_confined
        D_free = float(statglobal[7])
        if math.isnan(D_free):
            DG_free = 0
        else:
            DG_free = D_free
        D_global = (DG_immobile * P_immobile + DG_confined * P_confined + DG_free * P_free) * 0.01
        L_immobile = float(statglobal[11])
        L_confined = float(statglobal[12])
        L_free = float(statglobal[13])
        if math.isnan(L_immobile):
            LG_immobile = 0
        else:
            LG_immobile = L_immobile
        if math.isnan(D_confined):
            LG_confined = 0
        else:
            LG_confined = L_confined
        if math.isnan(D_free):
            LG_free = 0
        else:
            LG_free = L_free
        L_global = (LG_immobile * P_immobile + LG_confined * P_confined + LG_free * P_free) * 0.01
        N_confined = 0
        N_free = 0
        N_immobile = 0
        confinementRadiiSum = 0
        DE_immobile = float(statglobal[8])
        DE_confined = float(statglobal[9])
        DE_free = float(statglobal[10])
        LE_immobile = float(statglobal[14])
        LE_confined = float(statglobal[15])
        LE_free = float(statglobal[16])
        for i, j in enumerate(hdf5global["rossier"]["rossierStatistics"][()]):
            if j[2] == 1:
                N_confined += 1
                confinementRadiiSum += j[7]
            elif j[3] == 1:
                N_free += 1
            elif j[1] == 1:
                N_immobile += 1
            elif j[4] == 1:
                N_immobile += 1
        if N_confined > 0:
            confinementRadii = confinementRadiiSum / N_confined
        else:
            confinementRadii = 0
        N_global = N_immobile + N_confined + N_free
        coverslip_data[sorted.index(coverslip)].loc[
            len(coverslip_data[sorted.index(coverslip)])] = cell, os.path.getmtime(parent_string(cell,
                                                                                                 tiffiles,
                                                                                                 'metadata')) - csStartTime, P_immobile, P_confined, P_free, D_global, D_immobile, D_confined, D_free, L_global, L_immobile, L_confined, L_free, N_global, N_immobile, N_confined, N_free, confinementRadii, DE_immobile, DE_confined, DE_free, LE_immobile, LE_confined, LE_free
    if coverslip_data[sorted.index(coverslip)].empty:
        raise IncorrectConfigException('Coverslide number ' + str(
            sorted.index(coverslip) + 1) + ' created an empty dataframe. Please check your data')
    return coverslip_data


def fourset_output(save_dir, outputfile, dataset, coverslip_data, use_timestamps, plotcolor, t_lig, ligand_name,
                   error_type, binned):
    """
    :param save_dir: where to save
    :param outputfile: h5 file to output into
    :param dataset: the set inside the h5 file to output into
    :param coverslip_data: the data to output
    :param use_timestamps: whether timestamps were used
    :param plotcolor: what color to plot
    :param t_lig: when the ligand was added
    :param ligand_name: the name of the ligand
    :param error_type: whether to use SD or SEM
    :param binned: whether you're outputting binned data
    outputs data that has 4 diffusion types ('global', 'immobile', 'confined', 'free')
    """
    # determins attribute name
    for folder in ['D', 'L', 'N']:
        if folder == 'D':
            name = 'diffusion_coefficients'
        if folder == 'L':
            name = 'segment_lengths'
        if folder == 'N':
            name = 'number_of_segments'
        data = []
        if binned:
            os.mkdir(save_dir + '\\TRplots\\' + name)
            os.mkdir(save_dir + '\\TRplots\\' + name + '\\pngs')
            os.mkdir(save_dir + '\\TRplots\\' + name + '\\svgs')
            os.mkdir(save_dir + '\\TRplots\\' + name + '\\pdfs')
        # writes output by signal type
        for signal_type in ['global', 'immobile', 'confined', 'free']:
            if binned:
                data.append(compile_columns(dataset, ["Cell Name", "Time", folder + "_" + signal_type,
                                                      folder + '_' + signal_type + '_SD',
                                                      folder + '_' + signal_type + '_SEM'], True))
                if use_timestamps:
                    plot = plot_by_time(coverslip_data, folder + '_' + signal_type, t_lig, ligand_name, dataset,
                                        error_type, plotcolor)
                else:
                    plot = plot_by_cells(coverslip_data, folder + '_' + signal_type, t_lig, ligand_name, dataset,
                                         error_type,
                                         plotcolor)
                plot.savefig(save_dir + '\\TRplots\\' + name + '\\pngs\\' + signal_type + '.png', dpi=300)
                plot.savefig(save_dir + '\\TRplots\\' + name + '\\svgs\\' + signal_type + '.svg', dpi=300)
                plot.savefig(save_dir + '\\TRplots\\' + name + '\\pdfs\\' + signal_type + '.pdf', dpi=300)

            else:
                data.append(compile_columns(dataset, ["Cell Name", "Time", folder + "_" + signal_type,
                                                      folder + 'E_' + signal_type], True))
        output_folder(outputfile, name,
                      [['global', data[0]], ['immobile', data[1]], ['confined', data[2]], ['free', data[3]]])


def stack_data(dataframes):
    """
    :param dataframes: list of dataframes
    :return: a dataframe consisting of the frames stacked atop each other
    """
    columns = ["Cell Name", "Time", "P_immobile", "P_confined", "P_free", "D_global", "D_immobile", "D_confined",
               "D_free", "L_global", "L_immobile", "L_confined", "L_free", "N_global", "N_immobile", "N_confined",
               "N_free", 'confinement_radius', 'DE_immobile', 'DE_confined', 'DE_free', 'LE_immobile', 'LE_confined',
               'LE_free']
    for df in dataframes:
        new_col = {}
        for i, col in enumerate(df.columns):
            new_col[col] = columns[i]
        df.rename(columns=new_col)
    return [pd.concat(dataframes, axis=0)]


def main(config_path):
    start_time = time.time()
    csPaths = []
    paths = []
    files = []
    use_timestamps = False
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_path)

    try:
        if config["USE_TIMESTAMPS"]["use_timestamps"].lower() == "true":
            use_timestamps = True
    except KeyError:
        raise IncorrectConfigException("Section USE_TIMESTAMPS missing in config.")
    try:
        bin_size_cells = int(config["BIN_SIZE"]["cells"])
        bin_size_time = float(config["BIN_SIZE"]["minutes"]) * 60 + float(config["BIN_SIZE"]["seconds"])
    except KeyError:
        raise IncorrectConfigException("Section BIN_SIZE missing in config.")
    try:
        if len([key for key in config["CS_DIRS"]]):
            for key in config["CS_DIRS"]:
                csPaths.append(config["CS_DIRS"][key])
        else:
            raise IncorrectConfigException("No coverslip directory defined in config.")
    except KeyError:
        raise IncorrectConfigException("Section CS_DIRS missing in config.")

    tiffiles = []
    for dir in csPaths:
        tiffiles += get_matching_files(dir + "\\cells\\tifs", "cell", ["_dl", 'metadata'])

    csNames = []
    for cs in csPaths:
        csNames.append('_'.join(os.listdir(cs + "\\cells\\tifs")[0].split("\\")[-1].split('_')[:-2]))

    try:
        if len([key for key in config["GLOBAL_DIR"]]):
            for key in config["GLOBAL_DIR"]:
                paths.append(config["GLOBAL_DIR"][key])
        else:
            raise IncorrectConfigException("No GLOBAL directory defined in config.")
    except KeyError:
        raise IncorrectConfigException("Section GLOBAL_DIR missing in config.")

    try:
        save_dir = config["SAVE_DIR"]["svdir"]
    except KeyError:
        raise IncorrectConfigException("Parameter svdir missing in config.")

    try:
        alpha = float(config["STAT_SETTINGS"]["alpha_norm"])
    except KeyError:
        raise IncorrectConfigException("Parameter alpha_norm missing in config.")

    try:
        if config["STAT_SETTINGS"]["run_stats"].lower() == "true":
            run_stats = True
        else:
            run_stats = False
    except KeyError:
        raise IncorrectConfigException("Section USE_TIMESTAMPS missing in config.")

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

    try:
        plotcolor = config["PLOT_SETTINGS"]["dot_color"]
        if plotcolor == '':
            plotcolor = '#FFA500'
    except KeyError:
        raise IncorrectConfigException("Parameter dot_color missing in config.")

    try:
        t_lig = config["PLOT_SETTINGS"]["ligand_index"]
        if t_lig == '':
            ligand = False
        else:
            ligand = True
            # adds identificatior to the end of the ligand index
            if use_timestamps:
                t_lig += 's'
            else:
                t_lig += 'c'
    except KeyError:
        raise IncorrectConfigException("Parameter t_lig missing in config.")
    try:
        ligand_name = config["PLOT_SETTINGS"]["ligand_name"]
    except KeyError:
        raise IncorrectConfigException("Parameter ligand_name missing in config.")
    try:
        error_type = config["PLOT_SETTINGS"]["error_type"]
    except KeyError:
        raise IncorrectConfigException("Parameter error_type missing in config.")

    # writes the location of the h5 files into columns
    for path in paths:
        if type(path) is float:
            raise IncorrectConfigException("mismatched number of files")
        else:
            files = get_matching_files(path, ".h5", ["statistics.h5"])
    # sets up output folder
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(save_dir + '\\timeResolvedAnalysis')
    except FileExistsError:
        shutil.rmtree(save_dir + '\\timeResolvedAnalysis')
        os.mkdir(save_dir + '\\timeResolvedAnalysis')
    save_dir += '\\timeResolvedAnalysis'

    # writes the .tif files into a dictionary with the key being their coverslip name
    input_files = {c: [] for c in csNames}
    for h5 in files:
        filename = h5.split("\\")[-1][:-2]
        tif = parent_string(filename, tiffiles,
                            "metadata")  # goes through the tif files and looks for the one with the right name
        coverslipname = '_'.join(tif.split("\\")[-1].split('_')[:-2])
        input_files[coverslipname].append(filename)
    # sorts the files in correct order, taking multi digit numbers into account e.g. 10 comes after 2
    sorted = []
    for cs in input_files.keys():
        sorted.append(sort_cells(input_files[cs]))
    # prepares dataframes for each coverslip
    coverslip_data = [pd.DataFrame(
        columns=["Cell Name", "Time", "P_immobile", "P_confined", "P_free", "D_global", "D_immobile", "D_confined",
                 "D_free", "L_global", "L_immobile", "L_confined", "L_free", "N_global", "N_immobile", "N_confined",
                 "N_free", "confinement_radius", "DE_immobile", "DE_confined", "DE_free", "LE_immobile", "LE_confined",
                 "LE_free"]) for cs in csNames]

    # writes all data into dataframes and creates their own binned dataframe, then appends them to the binned_data list
    binned_data = []
    for coverslip in sorted:  # coverslip is a sorted list of all cells in a coverslip
        coverslip_data = load_cs(sorted, coverslip, files, tiffiles, coverslip_data)
        if use_timestamps:
            binned_mean = bin_data_time(coverslip_data[sorted.index(coverslip)], bin_size_time)
        else:
            binned_mean = bin_data_cells(coverslip_data[sorted.index(coverslip)], bin_size_cells)
        binned_data += [binned_mean]
    stacked_data = stack_data(coverslip_data)

    largest_bindex = 0
    for i, bin in enumerate(binned_data):
        if len(bin) > len(binned_data[largest_bindex]):
            largest_bindex = i

    global_mean = pd.DataFrame(binned_data[largest_bindex].iloc[:, 0:2])
    for attribute in ["P_immobile", "P_confined", "P_free", "D_global", "D_immobile", "D_confined",
                      "D_free", "L_global", "L_immobile", "L_confined", "L_free", "N_global", "N_immobile",
                      "N_confined",
                      "N_free", "confinement_radius"]:
        global_mean = pd.concat([global_mean, pd.DataFrame(calc_mean_over_cS(binned_data, attribute),
                                                           columns=[attribute, attribute + "_SD", attribute + "_SEM"])],
                                axis=1)
    for i, row in enumerate(global_mean["Cell Name"]):
        global_mean.iloc[i, 0] = row.split("_")[-1]
    global_mean.rename(columns={'Cell Names': 'cell range'})

    # output
    outputFile = h5py.File(save_dir + '\\stats.h5', 'w')
    outputFile_bin = outputFile.create_group('bin')
    outputFile_raw = outputFile.create_group('raw')
    outputFile_stacked = outputFile.create_group('stacked')
    # sheet with global means output
    output_folder(outputFile, 'global means',
                  [['global means', global_mean]])

    # raw data output
    output_folder(outputFile_raw, 'fractions',
                  [['immobile', compile_columns(coverslip_data, ["Cell Name", "Time", "P_immobile"], True)],
                   ['confined', compile_columns(coverslip_data, ["Cell Name", "Time", "P_confined"], True)],
                   ['free', compile_columns(coverslip_data, ["Cell Name", "Time", "P_free"], True)]])

    # binned data output
    # writes fractions and confinement_radii
    output_folder(outputFile_bin, 'fractions',
                  [['immobile', compile_columns(binned_data, ["Cell Name", "Time", "P_immobile"], True)],
                   ['confined', compile_columns(binned_data, ["Cell Name", "Time", "P_confined"], True)],
                   ['free', compile_columns(binned_data, ["Cell Name", "Time", "P_free"], True)]])
    # stacked output
    output_folder(outputFile_stacked, 'fractions',
                  [['immobile', compile_columns(stacked_data, ["Cell Name", "Time", "P_immobile"], False)],
                   ['confined', compile_columns(stacked_data, ["Cell Name", "Time", "P_confined"], False)],
                   ['free', compile_columns(stacked_data, ["Cell Name", "Time", "P_free"], False)]])

    name = 'fractions'
    os.mkdir(save_dir + '\\TRplots')
    os.mkdir(save_dir + '\\TRplots\\' + name)
    os.mkdir(save_dir + '\\TRplots\\' + name + '\\pngs')
    os.mkdir(save_dir + '\\TRplots\\' + name + '\\svgs')
    os.mkdir(save_dir + '\\TRplots\\' + name + '\\pdfs')
    for signal_type in ['immobile', 'confined', 'free']:
        if use_timestamps:
            plot = plot_by_time(coverslip_data, 'P_' + signal_type, t_lig, ligand_name, binned_data, error_type,
                                plotcolor)
        else:
            plot = plot_by_cells(coverslip_data, 'P_' + signal_type, t_lig, ligand_name, binned_data, error_type,
                                 plotcolor)
        plot.savefig(save_dir + '\\TRplots\\' + name + '\\pngs\\' + signal_type + '.png', dpi=300)
        plot.savefig(save_dir + '\\TRplots\\' + name + '\\svgs\\' + signal_type + '.svg', dpi=300)
        plot.savefig(save_dir + '\\TRplots\\' + name + '\\pdfs\\' + signal_type + '.pdf', dpi=300)

    output_folder(outputFile_raw, 'confinement_radii',
                  [['confinement_radii',
                    compile_columns(coverslip_data, ["Cell Name", "Time", 'confinement_radius'], True)]])
    output_folder(outputFile_bin, 'confinement_radii',
                  [['confinement_radii',
                    compile_columns(binned_data, ["Cell Name", "Time", 'confinement_radius'], True)]])
    output_folder(outputFile_stacked, 'confinement_radii',
                  [['confinement_radii',
                    compile_columns(stacked_data, ["Cell Name", "Time", 'confinement_radius'], False)]])

    name = 'confinement_radii'
    os.mkdir(save_dir + '\\TRplots\\' + name)
    if use_timestamps:
        plot = plot_by_time(coverslip_data, 'confinement_radius', t_lig, ligand_name, binned_data, error_type,
                            plotcolor)
    else:
        plot = plot_by_cells(coverslip_data, 'confinement_radius', t_lig, ligand_name, binned_data, error_type,
                             plotcolor)
    plot.savefig(save_dir + '\\TRplots\\' + name + '\\confinement_radii.png', dpi=300)
    plot.savefig(save_dir + '\\TRplots\\' + name + '\\confinement_radii.svg', dpi=300)
    plot.savefig(save_dir + '\\TRplots\\' + name + '\\confinement_radii.pdf', dpi=300)

    # writes the output files for diff coef, seg lengths, and number of segments
    # raw
    fourset_output(save_dir, outputFile_raw, coverslip_data, coverslip_data, use_timestamps, plotcolor, t_lig,
                   ligand_name,
                   error_type, False)
    # binned
    fourset_output(save_dir, outputFile_bin, binned_data, coverslip_data, use_timestamps, plotcolor, t_lig, ligand_name,
                   error_type, True)

    # stacked
    fourset_output(save_dir, outputFile_stacked, stacked_data, coverslip_data, use_timestamps, plotcolor, t_lig,
                   ligand_name,
                   error_type, False)

    metadata = pd.DataFrame(
        columns=['use_timestamps', 'n_cells', 'Plot_Error_type', 'Plot_Ligand_time', 'stat_alpha', 'stat_p1',
                 'stat_p2', 'stat_p3'])
    metadata.loc[len(metadata)] = pd.Series(
        [use_timestamps, str(int(bin_size_time / 60)) + 'm' + str(bin_size_time % 60) + 's', error_type, t_lig,
         alpha, p1, p2, p3]).values

    if use_timestamps:
        metadata = pd.DataFrame(
            columns=['use_timestamps', 'n_cells', 'Plot_Error_type', 'Plot_Ligand_time', 'stat_alpha', 'stat_p1',
                     'stat_p2', 'stat_p3'])
        metadata.loc[len(metadata)] = pd.Series(
            [use_timestamps, str(int(bin_size_time / 60)) + 'm' + str(bin_size_time % 60) + 's', error_type, t_lig,
             alpha, p1, p2, p3]).values
    else:
        metadata = pd.DataFrame(
            columns=['use_timestamps', 'n_cells', 'Plot_Error_type', 'Plot_Ligand_index', 'stat_alpha', 'stat_p1',
                     'stat_p2', 'stat_p3'])
        metadata.loc[len(metadata)] = pd.Series(
            [use_timestamps, str(bin_size_cells), error_type, t_lig,
             alpha, p1, p2, p3]).values

    output_folder(outputFile, 'metadata',
                  [['metadata', metadata]])
    try:
        output_folder(outputFile, 'metadata',
                      [['metadata', metadata]])

    except ValueError:
        pass
    outputFile.close()
    # runs the statistic tests
    if use_timestamps:
        run_stats = False
    if run_stats:
        os.mkdir(save_dir + '\\tests')
        os.mkdir(save_dir + '\\tests\\normality')
        if use_timestamps:
            if ligand:
                normFrames, signFrames = test_by_time(coverslip_data, bin_size_time, ligand, alpha, p1, p2, p3)
            else:
                normFrames = test_by_time(coverslip_data, bin_size_time, ligand, alpha, p1, p2, p3)
        else:
            if ligand:
                normFrames, signFrames = test_by_cell(coverslip_data, bin_size_cells, ligand, alpha, p1, p2, p3)
            else:
                normFrames = test_by_cell(coverslip_data, bin_size_cells, ligand, alpha, p1, p2, p3)
        for key in normFrames.keys():
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
            normFrames[key].to_csv(
                save_dir + '\\tests\\normality\\' + name + '\\test_normality_' + name + "_" + key2.split('_')[
                    -1] + '.csv', index=False)
        if ligand:
            os.mkdir(save_dir + '\\tests\\significance')
            for key in signFrames.keys():
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
                    signFrames[key].to_csv((
                                                       save_dir + '\\tests\\significance\\' + name + '\\test_significance_' + name + "_" +
                                                       key2.split('_')[
                                                           -1] + '.csv'), index=False)
                else:
                    signFrames[key].to_csv((
                            save_dir + '\\tests\\significance\\' + name + '\\test_significance_' + name + '.csv'), index=False)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    try:
        cfg_path = sys.argv[1]
        main(cfg_path)
    except FileExistsError:
        print("Usage: python timeResolvedAnalysis.py your_config_file.ini")
