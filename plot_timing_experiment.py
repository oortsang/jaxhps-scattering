import logging
import os
import argparse
from typing import List, Tuple
from datetime import datetime

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.io import loadmat
import pandas as pd

from hps.src.logging_utils import FMT, TIMEFMT
from hps.src.plotting import get_discrete_cmap


# Disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot results for wave scattering convergence study."
    )
    parser.add_argument(
        "--plots_dir",
        default="data/timing",
        type=str,
        help="Directory to save the reference solution.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


P = 16


def load_data_from_one_method(
    method_name: str, l_vals: List[int], plots_dir: str
) -> pd.DataFrame:
    """Inside plots_dir, there are files saved as timing_{method_name}_l_{l}.mat.
    Inside each file, key "times" is a 1D array of time samples. We want to save the mean and stddev of the times for each l.
    """
    df_lst: List[pd.DataFrame] = []
    for l in l_vals:
        file_path = os.path.join(plots_dir, f"timing_{method_name}_l_{l}.mat")
        if not os.path.exists(file_path):
            logging.warning("File %s does not exist.", file_path)
            continue
        logging.info("Loading data from %s", file_path)
        data = loadmat(file_path)
        times = data["times"].flatten()
        mean_time = np.mean(times)
        std_time = np.std(times)

        df = pd.DataFrame(
            {
                "l": [l],
                "mean_time": [mean_time],
                "std_time": [std_time],
                "method_name": [method_name],
                "n_disc_pts": [P**2 * 4**l],
            }
        )
        df_lst.append(df)

    df_out = pd.concat(df_lst, ignore_index=True)
    logging.debug("load_data_from_one_method: df_out = %s", df_out)
    return df_out


def main(args: argparse.Namespace) -> None:
    """
    1. loads all of the available data.
    2. Plots the timing results.
    """

    # Each lst element is a tuple. 1st element is the string for finding filename and 2nd element is the label for the plot.
    # The order of the lst is the order of the plot.
    method_names_lst = [
        # ("laptop", "Laptop"),
        ("multicore_cpu", "Multicore CPU"),
        ("gpu_no_recomp", "1 GPU; Naive Implementation"),
        ("gpu_baseline_recomp", "1 H100 GPU; Baseline Recomputation"),
        ("gpu_our_recomp", "1 H100 GPU; Our Recomputation"),
    ]

    # Load the data
    # l_vals = [2, 3, 4, 5, 6, 7, 8, 9]
    l_vals = [4, 5, 6, 7, 8, 9]

    df_lst: List[pd.DataFrame] = []
    for method_name, label in method_names_lst:
        out_df = load_data_from_one_method(method_name, l_vals, args.plots_dir)
        df_lst.append(out_df)

    # Make the plot
    fig, ax = plt.subplots(figsize=(5, 5))
    c = get_discrete_cmap(len(method_names_lst), "parula")

    for i, (method_name, label) in enumerate(method_names_lst):
        df_i = df_lst[i]
        if "cpu" in method_name:
            marker = "x"
        else:
            marker = "o"
        ax.errorbar(
            df_i["n_disc_pts"],
            df_i["mean_time"],
            ls="none",
            marker=marker,
            markersize=5,
            capsize=7,
            yerr=df_i["std_time"],
            label=label,
            color=c[i],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    # Set top and right spines to be invisible
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend location lower right
    ax.legend(loc="lower right")
    ax.set_xlabel("\\# Discretization Points", fontsize=20)
    ax.set_ylabel("Runtime (seconds)", fontsize=20)

    fp_out = os.path.join(args.plots_dir, "timing.svg")
    logging.info("Saving timing plot to %s", fp_out)
    plt.savefig(fp_out, bbox_inches="tight")
    plt.clf()

    logging.info("Finished")


if __name__ == "__main__":
    args = setup_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=FMT,
        datefmt=TIMEFMT,
    )
    main(args)
