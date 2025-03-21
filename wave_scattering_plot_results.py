import logging
import os
import argparse
from typing import List, Tuple
from datetime import datetime


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import pandas as pd
import jax.numpy as jnp

from hps.src.logging_utils import FMT, TIMEFMT
from hps.src.plotting import (
    get_discrete_cmap,
    plot_field_for_wave_scattering_experiment,
    make_scaled_colorbar,
    parula_cmap,
    CMAP_PAD,
    FONTSIZE_2,
    FONTSIZE_3,
    TICKSIZE_2,
    TICKSIZE_3,
    FIGSIZE_2,
    FIGSIZE_3,
)
from hps.src.scattering_potentials import (
    q_luneburg,
    q_vertically_graded,
    q_horizontally_graded,
    q_gaussian_bumps,
    q_GBM_1,
)
from hps.src.wave_scattering_utils import get_uin


# Disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot results for wave scattering convergence study."
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    parser.add_argument(
        "--scattering_potential",
        default="gauss_bumps",
        help="Scattering potential to use.",
        choices=[
            "luneburg",
            "vertically_graded",
            "horizontally_graded",
            "gauss_bumps",
            "GBM_1",
        ],
    )
    parser.add_argument("-k", default="20", type=str, help="Wave number for plotting.")
    return parser.parse_args()


def find_lst_of_dirs(input_dir: str, prefix: str = "approx_soln_") -> List[str]:
    """
    Returns a list of all sub-directories of `input_dir` that contain the prefix `prefix`.
    """

    # List of all directories in input_dir
    all_dirs = os.listdir(input_dir)

    # Filter out directories that do not contain the prefix
    filtered_dirs = [d for d in all_dirs if d.startswith(prefix)]
    return filtered_dirs


def lst_of_dirs_to_df(input_dir: str, lst_of_dirs: List[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ["dir", "l", "p", "time_mean", "time_std", "l_inf_err"]
    """

    data = []
    # df = pd.DataFrame(
    #     [], columns=["dir", "l", "p", "time_mean", "time_std", "l_inf_err"]
    # )
    for d in lst_of_dirs:
        # example d looks like approx_soln_p_12_l_5

        # Get the l and p values from the directory name
        lp_str = d.split("soln")[1]
        lp_vals = lp_str.split("_")
        # logging.debug("lst_of_dirs_to_df: d: %s", d)
        # logging.debug("lst_of_dirs_to_df: lp_vals: %s", lp_vals)
        l = lp_vals[4]
        p = lp_vals[2]
        l = int(l)
        p = int(p)

        # Load the data from the directory
        data_fp = os.path.join(input_dir, d, "data.mat")
        # logging.debug("lst_of_dirs_to_df: data_fp: %s", data_fp)
        data_dict = loadmat(data_fp)
        times = data_dict["times"]
        time_mean = np.mean(times).item()
        time_std = np.std(times).item()
        l_inf_err = data_dict["rel_linf_error"][0].item()
        dd = {
            "dir": d,
            "l": l,
            "p": p,
            "time_mean": time_mean,
            "time_std": time_std,
            "l_inf_err": l_inf_err,
        }
        # for k, v in dd.items():
        #     print("k: %s, type(v): %s" % (k, type(v)))
        data.append(dd)

    df = pd.DataFrame(data)

    return df


def plot_convergence(df: pd.DataFrame, output_fp: str, k_str: str) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(FIGSIZE_2, FIGSIZE_2))
    CAPSIZE = 7
    MARKERSIZE = 5

    # For each p value, plot 1 / l_inf_error on the x axis
    # and time_mean on the y axis. Use time_std to add error bars.
    p_vals = df["p"].unique()
    # Sort them
    p_vals = np.sort(p_vals)

    cmap = get_discrete_cmap(len(p_vals), "parula")
    for i, p in enumerate(p_vals):
        df_p = df[df["p"] == p]
        # Sort df_p by l
        df_p = df_p.sort_values("l")
        logging.debug("plot_convergence: df_p[time_std]: %s", df_p["time_std"])
        ax.errorbar(
            df_p["time_mean"],
            df_p["l_inf_err"],
            xerr=df_p["time_std"],
            fmt="o-",
            label=f"$p = {p}$",
            color=cmap[i],
            capsize=CAPSIZE,
            markersize=MARKERSIZE,
        )
    ax.set_ylabel("Relative $\\ell_\infty$ error", fontsize=FONTSIZE_2)
    ax.set_xlabel("Runtime (s)", fontsize=FONTSIZE_2)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")

    # xticks and yticks should be FONTSIZE_2
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE_2)
    ax.grid()
    fig.tight_layout()
    plt.savefig(output_fp, bbox_inches="tight")


def plot_scattering_potential(
    q: np.array,
    output_fp: str,
    k: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    """
    1. Plot the scattering potential q.
    2. Plot the wavelength of the wave given k.
    3. Save the plot to output_fp.
    """
    fig, ax = plt.subplots(1, 1, figsize=(FIGSIZE_3, FIGSIZE_3))

    # Plot the scattering potential
    im = ax.imshow(q, extent=[xmin, xmax, ymin, ymax], cmap="plasma")

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=CMAP_PAD)
    # plt.colorbar(im, cax=cax)
    make_scaled_colorbar(im, ax, fontsize=TICKSIZE_3)

    # Set ticks to [-1, 0, 1]
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    # ticks should be FONTSIZE_3
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE_3)

    # Plot the wavelength
    # wavelength = (2 * np.pi) / k
    # ax.text(0.6, -0.7, "$\\lambda$", fontsize=FONTSIZE, color="white")

    # # Below lambda, plot a line with caps with length 1/16
    # ax.errorbar(
    #     x=0.6,
    #     y=-0.75,
    #     xerr=[
    #         [0],
    #         [wavelength],
    #     ],
    #     color="white",
    #     capsize=CAPSIZE,
    #     elinewidth=THICKNESS,
    #     capthick=THICKNESS,
    # )

    fig.tight_layout()

    plt.savefig(output_fp, bbox_inches="tight")


def main(args: argparse.Namespace) -> None:

    # Check the scattering potential argument
    if args.scattering_potential == "luneburg":
        args.data_dir = f"data/wave_scattering/luneburg_k_{int(args.k)}"
        q_fn_handle = q_luneburg
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "vertically_graded":
        args.data_dir = f"data/wave_scattering/vertically_graded_k_{int(args.k)}"
        q_fn_handle = q_vertically_graded
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "horizontally_graded":
        args.data_dir = f"data/wave_scattering/horizontally_graded_k_{int(args.k)}"
        q_fn_handle = q_horizontally_graded
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "gauss_bumps":
        args.data_dir = f"data/wave_scattering/gauss_bumps_k_{int(args.k)}"
        q_fn_handle = q_gaussian_bumps
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "GBM_1":
        args.data_dir = f"data/wave_scattering/GBM_1_k_{int(args.k)}"
        q_fn_handle = q_GBM_1
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    else:
        raise ValueError("Invalid scattering potential")

    output_dir = args.data_dir
    assert os.path.isdir(args.data_dir), f"Directory {args.data_dir} does not exist."
    logging.info("Saving plots to %s", output_dir)

    lst_of_dirs = find_lst_of_dirs(args.data_dir)
    logging.info("Found %i directories", len(lst_of_dirs))
    df = lst_of_dirs_to_df(args.data_dir, lst_of_dirs)
    logging.info("df: %s", df)
    logging.info("df.shape: %s", df.shape)

    # Plot the convergence plot
    output_fp = os.path.join(
        output_dir, f"k_{args.k}_{args.scattering_potential}_convergence.svg"
    )
    logging.info("Plotting convergence to %s", output_fp)
    plot_convergence(
        df,
        output_fp,
        args.k,
    )

    # Load data for plotting the fields
    data_fp = os.path.join(args.data_dir, "reference_solution.mat")
    data_dict = loadmat(data_fp)
    uscat_ref = data_dict["uscat"]
    target_pts = data_dict["target_pts"]
    q_ref = data_dict["q"]

    uin_ref = get_uin(float(args.k), target_pts, jnp.array([0.0]))
    utot_ref = uscat_ref + uin_ref[..., 0]
    logging.debug("utot_ref shape: %s", utot_ref.shape)

    # Plot the scattering potential
    output_fp = os.path.join(
        output_dir, f"k_{args.k}_{args.scattering_potential}_scattering_potential.svg"
    )
    logging.info("Plotting scattering potential to %s", output_fp)
    plot_scattering_potential(q_ref, output_fp, float(args.k), xmin, xmax, ymin, ymax)

    # Plot the reference solution real part
    output_fp = os.path.join(
        output_dir,
        f"k_{args.k}_{args.scattering_potential}_utot_ground_truth_real.svg",
    )
    logging.info("Plotting utot real part to %s", output_fp)
    plot_field_for_wave_scattering_experiment(
        field=utot_ref.real,
        target_pts=target_pts,
        use_bwr_cmap=True,
        save_fp=output_fp,
    )


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
