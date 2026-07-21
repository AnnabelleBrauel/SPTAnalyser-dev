import configparser
import sys
import time
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

class IncorrectConfigException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

def load_user_input(config_path):

    config = configparser.ConfigParser()
    config.read(config_path)

    dirs = []

    try:
        if not config["DIRS"]:
            raise IncorrectConfigException("No files defined in [DIRS].")
        for key in config["DIRS"]:
            dirs.append(config["DIRS"][key])
    except KeyError:
        raise IncorrectConfigException("Section [DIRS] missing in config file.")

    try:
        def get_float(cfg, cfg_section, cfg_key, fallback=np.nan):
            value = cfg.get(cfg_section, cfg_key, fallback=None)
            return float(value) if value not in [None, ""] else fallback

        resting_bin_shift = get_float(config, "PLOT_SETTINGS", "resting_bin_shift")
        bin_size_resting = get_float(config, "PLOT_SETTINGS", "bin_size_resting")
        x_start_time = get_float(config, "PLOT_SETTINGS", "x_start_time")
        x_end_time = get_float(config, "PLOT_SETTINGS", "x_end_time")
        y_start_diffusion = get_float(config, "PLOT_SETTINGS", "y_start_diffusion")
        y_end_diffusion = get_float(config, "PLOT_SETTINGS", "y_end_diffusion")
        y_start_immobile = get_float(config, "PLOT_SETTINGS", "y_start_immobile")
        y_end_immobile = get_float(config, "PLOT_SETTINGS", "y_end_immobile")
    except KeyError as e:
        raise IncorrectConfigException(f"Missing parameter in [PLOT_SETTINGS]: {e}")

    try:
        save_dir = config["SAVE_DIR"]["save_dir"]
    except KeyError:
        raise IncorrectConfigException("Parameter save_dir missing in [SAVE_DIR].")

    return {"dirs": dirs,
            "resting_bin_shift": resting_bin_shift,
            "bin_size_resting": bin_size_resting,
            "x_start_time": x_start_time,
            "x_end_time": x_end_time,
            "y_start_diffusion": y_start_diffusion,
            "y_end_diffusion": y_end_diffusion,
            "y_start_immobile": y_start_immobile,
            "y_end_immobile": y_end_immobile,
            "save_dir": save_dir
            }


def load_data(dirs):
    """Load CSVs and return dict: {condition: dataframe}"""
    data = {}

    for path in dirs:
        parent = os.path.basename(os.path.dirname(path))

        # Extract condition name from parent folder
        cond = parent.replace("time_resolved_analysis_", "")

        df = pd.read_csv(path)
        data[cond] = df

    return data


def plot_free_diffusion(data, resting_bin_shift, bin_size_resting, x_start_time, x_end_time, y_start_diffusion, y_end_diffusion, save_dir):

    cmap = plt.get_cmap("Set2")
    n_colors = len(data)
    palette = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    conditions = list(data.keys())
    offsets = np.linspace(-1.2, 1.2, len(conditions))

    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['pdf.fonttype'] = 42  # essential for importing the plot into another program

    fig, ax = plt.subplots(figsize=(12, 6))

    def plot_segmented_line(ax, x, y, color):
        """
        Plot dashed line segments with gaps around markers.

        shrink : fraction of each segment removed at both ends
        """

        shrink=0.08

        for i in range(len(x) - 1):
            x0, y0 = x[i], y[i]
            x1, y1 = x[i + 1], y[i + 1]

            dx = x1 - x0
            dy = y1 - y0

            xs = x0 + shrink * dx
            ys = y0 + shrink * dy

            xe = x1 - shrink * dx
            ye = y1 - shrink * dy

            ax.plot([xs, xe], [ys, ye],
                    linestyle="--",
                    linewidth=1,
                    color=color)

    for i, (cond, offset) in enumerate(zip(conditions, offsets)):

        df = data[cond]

        x = df["Time_mid"].values

        # Apply overlay shift only to resting condition
        if cond.lower() == "resting":
            x = x + resting_bin_shift * bin_size_resting

        mean = df["D_free_mean"].values
        sem = df["D_free_sem"].values

        # normalization (skip resting)
        #if cond.lower() != "resting":
        #    pre0 = df[df["Time_mid"] < 0]

        #    if len(pre0) == 0:
        #        raise ValueError(f"No pre-0 bin found for {cond}")

        #    ref_idx = pre0["Time_mid"].idxmax()
        #    ref_value = df.loc[ref_idx, "D_free_mean"]

        #    mean = mean / ref_value
        #    sem = sem / ref_value

        # apply dodge AFTER normalization
        x = x + offset

        color = palette[i % len(palette)]

        # resting in gray
        if cond.lower() == "resting":
            ax.errorbar(x, mean, yerr=sem, fmt='o', capsize=3, color="grey", label=cond)
            #ax.plot(x, mean, linestyle='--', linewidth=1, color="grey")
            plot_segmented_line(ax, x, mean, "grey")

        else:
            ax.errorbar(x, mean, yerr=sem, fmt='o', capsize=3, label=cond, color=color)
            #ax.plot(x, mean, linestyle='--', linewidth=1, color=color)
            plot_segmented_line(ax, x, mean, color)

    # Axes
    if x_start_time is not None and not np.isnan(x_start_time) and \
            x_end_time is not None and not np.isnan(x_end_time):
        plt.xlim(x_start_time, x_end_time)
    elif x_start_time is not None and not np.isnan(x_start_time):
        plt.xlim(left=x_start_time)
    elif x_end_time is not None and not np.isnan(x_end_time):
        plt.xlim(right=x_end_time)
    if y_start_diffusion is not None and not np.isnan(y_start_diffusion) and \
            y_end_diffusion is not None and not np.isnan(y_end_diffusion):
        plt.ylim(y_start_diffusion, y_end_diffusion)
    elif y_start_diffusion is not None and not np.isnan(y_start_diffusion):
        plt.ylim(bottom=y_start_diffusion)
    elif y_end_diffusion is not None and not np.isnan(y_end_diffusion):
        plt.ylim(top=y_end_diffusion)

    # Vertical line at 0 min for ligand addition
    plt.axvline(x=0, color='red', linestyle='--', lw=1.5)

    plt.xlabel("time / min", fontsize=14)
    plt.ylabel(r"D$_\mathrm{free}$ / µm$^\mathrm{2}$s$^\mathrm{-1}$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()

    # Save
    out = rf"{save_dir}\D_free_overlay.pdf"
    plt.savefig(out, transparent=True, bbox_inches='tight')
    print("Saved:", out)

    plt.show()
    plt.close()


def main(config_path):

    start_time = time.time()

    # Load configuration
    config = load_user_input(config_path)

    # Load data
    data = load_data(config["dirs"])

    # Plot
    plot_free_diffusion(
        data,
        resting_bin_shift=config["resting_bin_shift"],
        bin_size_resting=config["bin_size_resting"],
        x_start_time=config["x_start_time"],
        x_end_time=config["x_end_time"],
        y_start_diffusion=config["y_start_diffusion"],
        y_end_diffusion=config["y_end_diffusion"],
        save_dir=config["save_dir"]
    )

    # Execution time
    print(f"--- {time.time() - start_time:.2f} seconds ---")

if __name__ == "__main__":

    try:
        cfg_path = sys.argv[1]
        main(cfg_path)
    except FileExistsError:
        print("Usage: python timeResolvedAnalysis.py your_config_file.ini")