"""
@author:
"""

import time
import sys
import configparser
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# import math
# import re
# import shutil
# from datetime import datetime
# import h5py
# from matplotlib.patches import Patch


class IncorrectConfigException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


def load_user_input(config_path):
    """
        Loads all user-defined settings and paths from the configuration file.
        Returns a dictionary.
        conditions returns a list like ['resting', 'FGF1', ...]
        input_directories returns a list to all paths to extracted data
        diff_coeff_global_paths returns a list with all '*/*_diff_coeff.txt' paths
        condition_save_dirs is a dictionary
        """

    # --- Initialize variables ---
    conditions = []
    input_directories = []
    diff_coeff_global_paths = {}
    diff_coeff_free_paths = {}
    diff_modes_paths = {}

    # --- Load configuration ---
    config = configparser.ConfigParser()
    config.read(config_path)

    # --- CONDITIONS section ---
    try:
        if "CONDITIONS" not in config:
            raise IncorrectConfigException("Section [CONDITIONS] missing in config file.")
        if not config["CONDITIONS"]:
            raise IncorrectConfigException("No entries found in [CONDITIONS].")
        for key in config["CONDITIONS"]:
            conditions.append(config["CONDITIONS"][key])
    except KeyError:
        raise IncorrectConfigException("Section [CONDITIONS] missing in config file.")

    # --- DIRECTORIES section ---

    try:
        if "DIRECTORIES" not in config:
            raise IncorrectConfigException("Section [DIRECTORIES] missing in config file.")
        if not config["DIRECTORIES"]:
            raise IncorrectConfigException("No entries found in [DIRECTORIES].")
        for key in config["DIRECTORIES"]:
            input_directories.append(config["DIRECTORIES"][key])
    except KeyError:
        raise IncorrectConfigException("Section [DIRECTORIES] missing in config file.")

    # --- ANALYSIS section ---
    try:
        if "ANALYSIS" not in config:
            raise IncorrectConfigException("Section [ANALYSIS] missing in config file.")
        diff_coeff_global = config.getboolean("ANALYSIS", "diffusion_coefficient_global", fallback=False)
        diff_coeff_free = config.getboolean("ANALYSIS", "diffusion_coefficient_free", fallback=True)
        diff_modes = config.getboolean("ANALYSIS", "diffusion_modes", fallback=True)
        seg_len_global = config.getboolean("ANALYSIS", "segment_lengths_global", fallback=False)
        seg_len_free = config.getboolean("ANALYSIS", "segment_lengths_free", fallback=False)
        seg_no_global = config.getboolean("ANALYSIS", "segment_number_global", fallback=False)
    except KeyError:
        raise IncorrectConfigException("Section [ANALYSIS] missing in config file.")

    # Find global diffusion coefficient input data
    if diff_coeff_global:
        for condition in conditions:
            diff_coeff_global_paths[condition] = [] # initialize dictionary with empty lists
        for directory in input_directories:
            for condition in conditions:
                file_path = os.path.join(
                    directory,
                    condition,
                    f"{condition}_diff_coeff.txt"
                )
                if os.path.exists(file_path):
                    diff_coeff_global_paths[condition].append(file_path)
                else:
                    print("Missing:", file_path)

    # Find free diffusion coefficient input data
    if diff_coeff_free:
        for condition in conditions:
            diff_coeff_free_paths[condition] = []
        for directory in input_directories:
            for condition in conditions:
                file_path = os.path.join(
                    directory,
                    condition,
                    f"{condition}_diff_coeff.txt"
                )
                if os.path.exists(file_path):
                    diff_coeff_free_paths[condition].append(file_path)
                else:
                    print("Missing:", file_path)

    # Find diffusion modes input data
    if diff_modes:
        for condition in conditions:
            diff_modes_paths[condition] = []
        for directory in input_directories:
            for condition in conditions:
                file_path = os.path.join(
                    directory,
                    condition,
                    f"{condition}_diff_modes.txt"
                )
                if os.path.exists(file_path):
                    diff_modes_paths[condition].append(file_path)
                else:
                    print("Missing:", file_path)

    # TODO: Find segment lengths data
    # TODO: Find segment number data

    # --- STATISTICS section ---
    try:
        if "STATISTICS" not in config:
            raise IncorrectConfigException("Section [STATISTICS] missing in config file.")
        run_statistics = config.getboolean(
            "STATISTICS", "run_statistics", fallback=False
        )
        if run_statistics:
            p_value = float(config["STATISTICS"]["p_value"])
            p1 = float(config["STATISTICS"]["significant"])
            p2 = float(config["STATISTICS"]["highly_significant"])
            p3 = float(config["STATISTICS"]["very_highly_significant"])
        else:
            p_value = p1 = p2 = p3 = None
    except KeyError:
        raise IncorrectConfigException("Error in [STATISTICS] section.")

    # --- SAVE_DIRECTORY section ---
    try:
        if "SAVE_DIRECTORY" not in config:
            raise IncorrectConfigException("Section [SAVE_DIRECTORY] missing.")
        save_dir = config["SAVE_DIRECTORY"]["save_dir"]
    except KeyError:
        raise IncorrectConfigException(
            "Parameter save_dir missing in [SAVE_DIRECTORY]."
        )
    os.makedirs(save_dir, exist_ok=True)

    # Create condition-specific subdirectories
    condition_save_dirs = {}
    for condition in conditions:
        condition_path = os.path.join(save_dir, condition)
        os.makedirs(condition_path, exist_ok=True)
        condition_save_dirs[condition] = condition_path

    print("\nSave folders created:")
    for cond, path in condition_save_dirs.items():
        print(f"{cond} -> {path}")

    # --- Final structured return ---
    return {
        # File system
        "conditions": conditions,
        "input_directories": input_directories,
        "diff_coeff_global_paths": diff_coeff_global_paths,
        "diff_coeff_free_paths": diff_coeff_free_paths,
        "diff_modes_paths": diff_modes_paths,
        "save_dir": save_dir,
        "condition_save_dirs": condition_save_dirs,

        # Analysis
        "diff_coeff_global": diff_coeff_global,
        "diff_coeff_free": diff_coeff_free,
        "diff_modes": diff_modes,
        "seg_len_global": seg_len_global,
        "seg_len_free": seg_len_free,
        "seg_no_global": seg_no_global,

        # Statistics
        "run_statistics": run_statistics,
        "p_value": p_value,
        "p1": p1,
        "p2": p2,
        "p3": p3,
    }


def analyse_diff_coeff_global(diff_coeff_global_paths, condition_save_dirs, save_dir):

    def calculate(diff_coeff_global_paths, condition_save_dirs):

        pooled_df_dict = {}
        pooled_stats_dict = {}

        # Iterate through conditions
        for condition, paths in diff_coeff_global_paths.items():

            pooled_dfs = []
            save_dir = condition_save_dirs[condition]

            print(f"\nProcessing condition: {condition}")
            print("Save directory:", save_dir)

            # Iterate trough date files in one condition
            for path in paths:

                if not os.path.exists(path):
                    print("Missing:", path)
                    continue

                df = pd.read_csv(path, sep="\t", header=0, skiprows=[1])

                # Extract cell number
                df["cell_number"] = (
                    df["cell label"]
                    .str.extract(r"_cell_(\d+)$")
                    .astype(int)
                )

                # Extract date (first 6 characters)
                df["date"] = df["cell label"].str[:6]

                # Filter cells > 5
                df_filtered = df[df["cell_number"] > 5].copy()
                # TODO: make filtering variable later
                # df_filtered.index = df_filtered.index.str.strip()
                # df_filtered.columns = df_filtered.column.str.strip()
                pooled_dfs.append(df_filtered)

                # Calculate mean and median values
                value_columns = [
                    col for col in df_filtered.columns
                    if ("Δ" not in col)
                    and (col not in ["cell label", "cell_number", "date"])
                ]

                means = df_filtered[value_columns].mean()
                medians = df_filtered[value_columns].median()
                stds = df_filtered[value_columns].std()
                ns = df_filtered[value_columns].count()  # counts non-NaN values
                sems = stds / (ns ** 0.5)

                results_df = pd.DataFrame({
                    "N": ns,
                    "Mean": means,
                    "Median": medians,
                    "Standard Deviation": stds,
                    "Standard Error of the Mean": sems
                })

                date = df_filtered["date"].iloc[0]
                # output_filename = f"{date}_{condition}_stats.csv"
                output_path = os.path.join(save_dir, f"{date}_{condition}_results_diffusion_coefficient_global.csv")
                results_df.to_csv(output_path)

                # print(df_filtered, "\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            if not pooled_dfs:
                continue

            pooled_df = pd.concat(pooled_dfs, ignore_index=True)
            pooled_df_dict[condition] = pooled_df
            # print(pooled_df, "xxxxxxxxxxxxxxxxxxxxxxx")

            # Save pooled input data
            raw_output_path = os.path.join(save_dir, f"{condition}_pooled_raw_diffusion_coefficient_global.csv")
            pooled_df.to_csv(raw_output_path, index=False)

            # Create pooled output data
            value_columns = [
                col for col in pooled_df.columns
                if ("Δ" not in col)
                   and (col not in ["cell label", "cell_number", "date"])
            ]

            means = pooled_df[value_columns].mean()
            medians = pooled_df[value_columns].median()
            stds = pooled_df[value_columns].std()
            ns = pooled_df[value_columns].count()  # counts non-NaN values
            sems = stds / (ns ** 0.5)

            pooled_results_df = pd.DataFrame({
                "N": ns,
                "Mean": means,
                "Median": medians,
                "Standard Deviation": stds,
                "Standard Error of the Mean": sems
            })
            pooled_results_df.index = pooled_results_df.index.str.strip()
            pooled_results_df.columns = pooled_results_df.columns.str.strip()
            pooled_stats_dict[condition] = pooled_results_df

            stats_output_path = os.path.join(
                save_dir,
                f"{condition}_pooled_results_diffusion_coefficient_global.csv"
            )

            pooled_results_df.to_csv(stats_output_path)

        return pooled_df_dict, pooled_stats_dict

    def plot(pooled_df_dict, pooled_stats_dict, save_dir):

        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['pdf.fonttype'] = 42

        raw_plot_data = pd.concat(
            [df.assign(Condition=cond) for cond, df in pooled_df_dict.items()],
            ignore_index=True
        )

        raw_plot_data.columns = raw_plot_data.columns.str.strip()
        # print(raw_plot_data.columns.tolist())

        conditions = raw_plot_data["Condition"].unique()
        n_conditions = len(conditions)

        # Colors
        violin_color = ["#d4e9f3"]  # color for first violin
        violin_color += ["#a8e0ff"] * (n_conditions - 1) # color for every other violin
        violin_edge_color = "#2b8bc1"
        data_points_color = "white"
        data_points_edge_color = "#0072b2"
        mean_color = "#0072b2"
        median_color = "#0072b2"

        # pastel shades for violins
        # cmap = plt.get_cmap("Set3")
        # palette = [cmap(i / n_conditions) for i in range(n_conditions)]
        # darker shades for edges
        # dark_palette = sns.color_palette("dark", n_colors=n_conditions)

        plt.figure(figsize=(8, 8))

        violin_width = 0.6

        sns.violinplot(
            x="Condition",
            y="mean D global",
            hue="Condition",
            data=raw_plot_data,
            inner=None,
            width=violin_width,
            density_norm='width',
            palette=violin_color,
            linewidth=1,
            edgecolor=violin_edge_color,
            cut=0, # no extrapolation
        )

        # Mean and median
        for i, cond in enumerate(conditions):
            cond_data = raw_plot_data[raw_plot_data["Condition"] == cond]["mean D global"].to_numpy()

            n_points = len(cond_data)
            jitter = np.random.normal(loc=0, scale=0.1, size=n_points)
            jitter = np.clip(jitter, -0.25, 0.25)

            plt.scatter(
                x=i + jitter,  # add jitter to x
                y=cond_data,
                facecolors=data_points_color,  # fill color
                edgecolors=data_points_edge_color,  # border color
                s=15,
                zorder=5,
                linewidths=0.8
            )

            stats_df = pooled_stats_dict[cond]
            mean_val = stats_df.loc["mean D global", "Mean"]
            median_val = stats_df.loc["mean D global", "Median"]

            # Mean as dashed line
            plt.hlines(
                y=mean_val,
                xmin=i - violin_width / 2,
                xmax=i + violin_width / 2,
                color=mean_color,
                linewidth=1.2,
                linestyle='--',
                zorder=10,
                #label="Mean" if i == 0 else ""
            )

            # Median as a star
            plt.scatter(
                i, median_val,
                color=median_color,
                s=300,
                marker="*",
                zorder=12,
                #label="Median" if i == 0 else ""
            )

        plt.xlabel("")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.ylabel(r"D$_\mathrm{global}$ / µm$^\mathrm{2}$s$^\mathrm{-1}$", fontsize=24)
        plt.tight_layout()

        # plt.show()

        plot_path = os.path.join(save_dir, "diff_coeff_global_plot_pooled.pdf")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    pooled_df_dict, pooled_stats_dict = calculate(diff_coeff_global_paths, condition_save_dirs)
    plot(pooled_df_dict, pooled_stats_dict, save_dir)


def analyse_diff_coeff_free(diff_coeff_free_paths, condition_save_dirs, save_dir):

    def pool(diff_coeff_free_paths, condition_save_dirs):

        pooled_df_dict = {}
        pooled_stats_dict = {}

        # Iterate through conditions
        for condition, paths in diff_coeff_free_paths.items():

            pooled_dfs = []
            save_dir = condition_save_dirs[condition]

            print(f"\nProcessing condition: {condition}")
            print("Save directory:", save_dir)

            # Iterate trough date files in one condition
            for path in paths:

                if not os.path.exists(path):
                    print("Missing:", path)
                    continue

                df = pd.read_csv(path, sep="\t", header=0, skiprows=[1])

                # Extract cell number
                df["cell_number"] = (
                    df["cell label"]
                    .str.extract(r"_cell_(\d+)$")
                    .astype(int)
                )

                # Extract date (first 6 characters)
                df["date"] = df["cell label"].str[:6]

                # Filter cells > 5
                df_filtered = df[df["cell_number"] > 5].copy()
                # TODO: make filtering variable later
                # df_filtered.index = df_filtered.index.str.strip()
                # df_filtered.columns = df_filtered.column.str.strip()
                pooled_dfs.append(df_filtered)

                # Calculate mean and median values
                value_columns = [
                    col for col in df_filtered.columns
                    if ("Δ" not in col)
                    and (col not in ["cell label", "cell_number", "date"])
                ]

                # TODO: analyse only D free column here

                means = df_filtered[value_columns].mean()
                medians = df_filtered[value_columns].median()
                stds = df_filtered[value_columns].std()
                ns = df_filtered[value_columns].count()  # counts non-NaN values
                sems = stds / (ns ** 0.5)

                results_df = pd.DataFrame({
                    "N": ns,
                    "Mean": means,
                    "Median": medians,
                    "Standard Deviation": stds,
                    "Standard Error of the Mean": sems
                })

                date = df_filtered["date"].iloc[0]
                # output_filename = f"{date}_{condition}_stats.csv"
                output_path = os.path.join(save_dir, f"{date}_{condition}_results_diffusion_coefficient_free.csv")
                # results_df.to_csv(output_path)

                # print(df_filtered, "\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            if not pooled_dfs:
                continue

            pooled_df = pd.concat(pooled_dfs, ignore_index=True)
            pooled_df_dict[condition] = pooled_df
            # print(pooled_df, "xxxxxxxxxxxxxxxxxxxxxxx")

            # Save pooled input data
            raw_output_path = os.path.join(save_dir, f"{condition}_pooled_raw_diffusion_coefficient_free.csv")
            # pooled_df.to_csv(raw_output_path, index=False)

            # Create pooled output data
            value_columns = [
                col for col in pooled_df.columns
                if ("Δ" not in col)
                   and (col not in ["cell label", "cell_number", "date"])
            ]

            means = pooled_df[value_columns].mean()
            medians = pooled_df[value_columns].median()
            stds = pooled_df[value_columns].std()
            ns = pooled_df[value_columns].count()  # counts non-NaN values
            sems = stds / (ns ** 0.5)

            pooled_results_df = pd.DataFrame({
                "N": ns,
                "Mean": means,
                "Median": medians,
                "Standard Deviation": stds,
                "Standard Error of the Mean": sems
            })
            pooled_results_df.index = pooled_results_df.index.str.strip()
            pooled_results_df.columns = pooled_results_df.columns.str.strip()
            pooled_stats_dict[condition] = pooled_results_df

            stats_output_path = os.path.join(
                save_dir,
                f"{condition}_pooled_results_diffusion_coefficient_free.csv"
            )

            # pooled_results_df.to_csv(stats_output_path)

        return pooled_df_dict, pooled_stats_dict

    def plot(pooled_df_dict, pooled_stats_dict, save_dir):

        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['pdf.fonttype'] = 42

        raw_plot_data = pd.concat(
            [df.assign(Condition=cond) for cond, df in pooled_df_dict.items()],
            ignore_index=True
        )

        raw_plot_data.columns = raw_plot_data.columns.str.strip()
        # print(raw_plot_data.columns.tolist())

        conditions = raw_plot_data["Condition"].unique()
        n_conditions = len(conditions)

        # Colors
        violin_color = ["#d4e9f3"]  # color for first violin
        violin_color += ["#a8e0ff"] * (n_conditions - 1) # color for every other violin
        violin_edge_color = "#2b8bc1"
        data_points_color = "white"
        data_points_edge_color = "#0072b2"
        mean_color = "#0072b2"
        median_color = "#0072b2"

        # pastel shades for violins
        # cmap = plt.get_cmap("Set3")
        # palette = [cmap(i / n_conditions) for i in range(n_conditions)]
        # darker shades for edges
        # dark_palette = sns.color_palette("dark", n_colors=n_conditions)

        plt.figure(figsize=(8, 8))

        violin_width = 0.6

        sns.violinplot(
            x="Condition",
            y="mean D free",
            hue="Condition",
            data=raw_plot_data,
            inner=None,
            width=violin_width,
            density_norm='width',
            palette=violin_color,
            linewidth=1,
            edgecolor=violin_edge_color,
            cut=0, # no extrapolation
        )

        # Mean and median
        for i, cond in enumerate(conditions):
            cond_data = raw_plot_data[raw_plot_data["Condition"] == cond]["mean D free"].to_numpy()

            n_points = len(cond_data)
            jitter = np.random.normal(loc=0, scale=0.1, size=n_points)
            jitter = np.clip(jitter, -0.25, 0.25)

            plt.scatter(
                x=i + jitter,  # add jitter to x
                y=cond_data,
                facecolors=data_points_color,  # fill color
                edgecolors=data_points_edge_color,  # border color
                s=15,
                zorder=5,
                linewidths=0.8
            )

            stats_df = pooled_stats_dict[cond]
            mean_val = stats_df.loc["mean D free", "Mean"]
            median_val = stats_df.loc["mean D free", "Median"]

            # Mean as dashed line
            plt.hlines(
                y=mean_val,
                xmin=i - violin_width / 2,
                xmax=i + violin_width / 2,
                color=mean_color,
                linewidth=1.2,
                linestyle='--',
                zorder=10,
                #label="Mean" if i == 0 else ""
            )

            # Median as a star
            plt.scatter(
                i, median_val,
                color=median_color,
                s=300,
                marker="*",
                zorder=12,
                #label="Median" if i == 0 else ""
            )

        plt.xlabel("")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.ylabel(r"D$_\mathrm{free}$ / µm$^\mathrm{2}$s$^\mathrm{-1}$", fontsize=24)
        plt.tight_layout()

        # plt.show()

        plot_path = os.path.join(save_dir, "diff_coeff_free_plot_pooled.pdf")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    pooled_df_dict, pooled_stats_dict = pool(diff_coeff_free_paths, condition_save_dirs)
    plot(pooled_df_dict, pooled_stats_dict, save_dir)


def analyse_diffusion_modes(diff_modes_paths, condition_save_dirs, save_dir):

    def calculate(diff_modes_paths, condition_save_dirs):

        pooled_df_dict = {}
        pooled_stats_dict = {}

        # Iterate through conditions
        for condition, paths in diff_modes_paths.items():

            pooled_dfs = []
            save_dir = condition_save_dirs[condition]

            print(f"\nProcessing condition: {condition}")
            print("Save directory:", save_dir)

            # Iterate trough date files in one condition
            for path in paths:

                if not os.path.exists(path):
                    print("Missing:", path)
                    continue

                df = pd.read_csv(path, sep="\t", header=0, skiprows=[1])

                # Extract cell number
                df["cell_number"] = (
                    df["cell label"]
                    .str.extract(r"_cell_(\d+)$")
                    .astype(int)
                )

                # Extract date (first 6 characters)
                df["date"] = df["cell label"].str[:6]

                # Filter cells > 5
                df_filtered = df[df["cell_number"] > 5].copy()
                # TODO: make filtering variable later
                # df_filtered.index = df_filtered.index.str.strip()
                # df_filtered.columns = df_filtered.column.str.strip()
                pooled_dfs.append(df_filtered)

                # Calculate mean and median values
                value_columns = ["immobile", "confined", "free"]

                means = df_filtered[value_columns].mean()
                medians = df_filtered[value_columns].median()
                stds = df_filtered[value_columns].std()
                ns = df_filtered[value_columns].count()  # counts non-NaN values
                sems = stds / (ns ** 0.5)

                results_df = pd.DataFrame({
                    "N": ns,
                    "Mean": means,
                    "Median": medians,
                    "Standard Deviation": stds,
                    "Standard Error of the Mean": sems
                })

                date = df_filtered["date"].iloc[0]
                # output_filename = f"{date}_{condition}_stats.csv"
                output_path = os.path.join(save_dir, f"{date}_{condition}_results_diffusion_modes.csv")
                results_df.to_csv(output_path)

                # print(df_filtered, "\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            if not pooled_dfs:
                continue

            pooled_df = pd.concat(pooled_dfs, ignore_index=True)
            pooled_df_dict[condition] = pooled_df
            # print(pooled_df, "xxxxxxxxxxxxxxxxxxxxxxx")

            # Save pooled input data
            raw_output_path = os.path.join(save_dir, f"{condition}_pooled_raw_diffusion_modes.csv")
            pooled_df.to_csv(raw_output_path, index=False)

            # Create pooled output data
            value_columns = ["immobile", "confined", "free"]

            means = pooled_df[value_columns].mean()
            medians = pooled_df[value_columns].median()
            stds = pooled_df[value_columns].std()
            ns = pooled_df[value_columns].count()  # counts non-NaN values
            sems = stds / (ns ** 0.5)

            pooled_results_df = pd.DataFrame({
                "N": ns,
                "Mean": means,
                "Median": medians,
                "Standard Deviation": stds,
                "Standard Error of the Mean": sems
            })
            pooled_results_df.index = pooled_results_df.index.str.strip()
            pooled_results_df.columns = pooled_results_df.columns.str.strip()
            pooled_stats_dict[condition] = pooled_results_df

            stats_output_path = os.path.join(
                save_dir,
                f"{condition}_pooled_results_diffusion_modes.csv"
            )

            pooled_results_df.to_csv(stats_output_path)

        return pooled_df_dict, pooled_stats_dict

    def plot(pooled_df_dict, pooled_stats_dict, save_dir):

        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['pdf.fonttype'] = 42

        conditions = list(pooled_stats_dict.keys())
        n_conditions = len(conditions)

        modes = ["immobile", "confined", "free"]
        colors = ["#7a7a7a", "#d999bc", "#1affc0"]

        # Extract values
        means_matrix = np.array([
            [pooled_stats_dict[cond].loc[mode, "Mean"] for mode in modes]
            for cond in conditions
        ])  # shape: (n_conditions, 3)

        fig, ax = plt.subplots(figsize=(8, 8))

        bottom = np.zeros(n_conditions)  # for stacked bars

        # Stacked bars
        for i, mode in enumerate(modes):
            values = means_matrix[:, i]
            sems = np.array([
                pooled_stats_dict[cond].loc[mode, "Standard Error of the Mean"]
                for cond in conditions
            ])
            ax.bar(
                x=np.arange(n_conditions),
                height=values,
                bottom=bottom,
                color=colors[i],
                edgecolor="black",
                label=mode,
                yerr=sems,
                capsize=5, # small horizontal cap
                error_kw=dict(lw=1.5) # line width of error bars
            )
            bottom += values  # stack next bar

        # Axes
        ax.set_xticks(np.arange(n_conditions))
        ax.set_xticklabels(conditions, rotation=0, ha="center", fontsize=14)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticks(np.arange(0, 101, 10), minor=True)
        ax.set_yticks(np.arange(0, 101, 5), minor=True)
        ax.tick_params(axis='y', which='major', length=10, width=1)  # major ticks
        ax.tick_params(axis='y', which='minor', length=5, width=1)  # minor ticks
        ax.set_ylabel("motion mode / %", fontsize=24)
        # ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=18)
        ax.tick_params(axis="x", labelsize=18)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[::-1]
        labels = labels[::-1]
        legend = ax.legend(
            handles, labels,
            fontsize=18,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            handlelength=1.0,  # width of the markerr
            handleheight=3.0,  # height of the marker
            handletextpad=0.6  # distance text <-> marker
        )
        for text in legend.get_texts():
            text.set_rotation(90)
            text.set_ha('center') # horizontal alignment
            text.set_va('center') # vertical alignment

        plt.tight_layout()

        # plt.show()

        plot_path = os.path.join(save_dir, "diff_modes_plot_pooled.pdf")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    pooled_df_dict, pooled_stats_dict = calculate(diff_modes_paths, condition_save_dirs)
    plot(pooled_df_dict, pooled_stats_dict, save_dir)

def main(config_path):

    start_time = time.time()

    # Load configuration
    config = load_user_input(config_path)  # returns dictionary

    # Perform analyses
    if config["diff_coeff_global"] == True:
        analyse_diff_coeff_global(
            diff_coeff_global_paths = config["diff_coeff_global_paths"],
            condition_save_dirs = config["condition_save_dirs"],
            save_dir = config["save_dir"]
        )
    if config["diff_coeff_free"] == True:
        analyse_diff_coeff_free(
            diff_coeff_free_paths = config["diff_coeff_free_paths"],
            condition_save_dirs = config["condition_save_dirs"],
            save_dir = config["save_dir"]
        )
    if config["diff_modes"] == True:
        analyse_diffusion_modes(
            diff_modes_paths = config["diff_modes_paths"],
            condition_save_dirs = config["condition_save_dirs"],
            save_dir = config["save_dir"]
        )
    # TODO: add other analyses here

    # Print execution time
    print(f"--- analysis took {time.time() - start_time:.2f} seconds ---")


if __name__ == "__main__":

    try:
        cfg_path = sys.argv[1]
        main(cfg_path)
    except FileExistsError:
        print("Usage: python analyse_results.py analyse_results_config.ini")