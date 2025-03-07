import logging
import os
import argparse
from typing import List, Tuple
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.io import loadmat
import pandas as pd

from hps.src.logging_utils import FMT, TIMEFMT
from hps.src.plotting import get_discrete_cmap, parula_cmap
from hps.src.quadrature.quad_2D.interpolation import (
    interp_from_nonuniform_hps_to_regular_grid,
)
from hps.src.inverse_scattering_utils import (
    q_point_sources,
    source_locations_to_scattered_field,
    forward_model,
    SAMPLE_TREE,
    L,
    P,
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    OBSERVATION_BOOLS,
    K,
)
from hps.src.wave_scattering_utils import get_uin

N_BUMPS = 4

# Disable matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot results for wave scattering convergence study."
    )
    parser.add_argument(
        "--plots_dir",
        default="data/inverse_wave_scattering",
        type=str,
        help="Directory to save the reference solution.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def plot_uscat(
    uscat_regular: jnp.array, observation_pts: jnp.array, plots_dir: str
) -> None:
    uscat_real_fp = os.path.join(plots_dir, "utot_ground_truth_real.svg")
    logging.info("plot_uscat: Saving uscat plot to %s", uscat_real_fp)
    plt.imshow(
        uscat_regular.real,
        cmap=parula_cmap,
        vmin=-1 * uscat_regular.real.max(),
        vmax=uscat_regular.real.max(),
        extent=(XMIN, XMAX, YMIN, YMAX),
    )
    plt.plot(observation_pts[:, 0], observation_pts[:, 1], "x", color="black")

    # plt.plot(observation_pts[:, 0], observation_pts[:, 1], "x", color="black")
    plt.colorbar()
    plt.savefig(uscat_real_fp, bbox_inches="tight")
    plt.clf()

    uscat_abs_fp = os.path.join(plots_dir, "utot_ground_truth_abs.svg")
    logging.info("plot_uscat: Saving uscat plot to %s", uscat_abs_fp)
    plt.imshow(
        jnp.abs(uscat_regular),
        cmap="hot",
        extent=(XMIN, XMAX, YMIN, YMAX),
    )
    plt.colorbar()
    plt.savefig(uscat_abs_fp, bbox_inches="tight")
    plt.clf()


def plot_iterates(iterates: jnp.array, plots_dir: str, q_evals: jnp.array) -> None:
    """
    Make a plot of the evaluations of the ground-truth q and then draw arrows showing the iterates.
    """

    fig, ax = plt.subplots(figsize=(5, 5))
    FONTSIZE = 11
    CAPSIZE = 2
    THICKNESS = 1.2

    # plot q
    plt.imshow(q_evals, cmap=parula_cmap, extent=(XMIN, XMAX, YMIN, YMAX))
    plt.colorbar()

    # The wavelength is 2pi / K. Plot a white bar with the wavelength in the bottom-right corner.
    wavelength = (2 * np.pi) / K
    ax.text(0.6, -0.7, "$\\lambda$", fontsize=FONTSIZE, color="white")

    # Below lambda, plot a line with caps with length 1/16
    ax.errorbar(
        x=0.6,
        y=-0.75,
        xerr=[
            [0],
            [wavelength],
        ],
        color="white",
        capsize=CAPSIZE,
        elinewidth=THICKNESS,
        capthick=THICKNESS,
    )

    # Identify the first iterates
    iterate_0 = iterates[0, :]
    logging.info("plot_iterates: iterate_0 shape: %s", iterate_0.shape)

    # Draw a white ring around each of the first iterates
    for j in range(iterate_0.shape[0] // 2):
        plt.plot(
            iterate_0[2 * j],
            iterate_0[2 * j + 1],
            "o",
            color="white",
            markersize=4,
            zorder=4,
        )

    # Draw a black ring around each of the last iterates
    iterate_20 = iterates[-1, :]
    for j in range(iterate_20.shape[0] // 2):
        plt.plot(
            iterate_20[2 * j],
            iterate_20[2 * j + 1],
            "o",
            color="black",
            markersize=4,
            zorder=4,
        )

    # First one we want to put $t=0$ slightly left of the iterate
    # GAP_0 = 0.05
    # ax.text(
    #     iterate_0[0] - GAP_0,
    #     iterate_0[1],
    #     f"$t={0}$ ",
    #     fontsize=FONTSIZE,
    #     color="white",
    #     ha="right",
    #     va="bottom",
    # )

    # # Last one we want to put $t=0$ slightly right of the iterate
    # ax.text(
    #     iterate_0[-2] + GAP_0,
    #     iterate_0[-1],
    #     f" $t={0}$",
    #     fontsize=FONTSIZE,
    #     color="white",
    #     ha="left",
    #     va="top",
    # )

    # # Middle two, put $t=0$ at T_0_A and draw arrows
    # GAP = 0.8
    # HEADWIDTH = 0.025
    # HEADLENGTH = 0.025
    # WIDTH = 7e-04
    # T_0_A = [0.0, 0.5]
    # ax.text(
    #     T_0_A[0],
    #     T_0_A[1],
    #     f"$t=0$",
    #     fontsize=FONTSIZE,
    #     color="white",
    #     ha="left",
    #     va="bottom",
    # )
    # # Draw an arrow from T_0_A to iterate_0
    # ax.arrow(
    #     T_0_A[0],
    #     T_0_A[1],
    #     (iterate_0[2] - T_0_A[0]) * GAP,
    #     (iterate_0[3] - T_0_A[1]) * GAP,
    #     width=WIDTH,
    #     head_width=HEADWIDTH,
    #     head_length=HEADLENGTH,
    #     length_includes_head=True,
    #     fc="white",
    #     ec="white",
    #     zorder=4,
    # )
    # ax.arrow(
    #     T_0_A[0],
    #     T_0_A[1],
    #     (iterate_0[4] - T_0_A[0]) * GAP,
    #     (iterate_0[5] - T_0_A[1]) * GAP,
    #     width=WIDTH,
    #     head_width=HEADWIDTH,
    #     head_length=HEADLENGTH,
    #     length_includes_head=True,
    #     fc="white",
    #     ec="white",
    #     zorder=4,
    # )

    # # Draw $t=20$ at T_20_A and draw an arrow to iterate_20
    # iterate_20 = iterates[-1, :]
    # T_20_A = [-0.5, 0.15]
    # ax.text(
    #     T_20_A[0],
    #     T_20_A[1],
    #     f"$t=20$",
    #     fontsize=FONTSIZE,
    #     color="white",
    #     ha="right",
    #     va="bottom",
    # )
    # ax.arrow(
    #     T_20_A[0],
    #     T_20_A[1],
    #     (iterate_20[4] - T_20_A[0]) * GAP,
    #     (iterate_20[5] - T_20_A[1]) * GAP,
    #     width=WIDTH,
    #     head_width=HEADWIDTH,
    #     head_length=HEADLENGTH,
    #     length_includes_head=True,
    #     fc="white",
    #     ec="white",
    #     zorder=4,
    # )

    # # Draw $t=20$ at T_20_B and draw an arrow to iterate_20
    # T_20_B = [-0.5, -0.5]
    # ax.text(
    #     T_20_B[0],
    #     T_20_B[1],
    #     f"$t=20$",
    #     fontsize=FONTSIZE,
    #     color="white",
    #     ha="right",
    #     va="top",
    # )
    # ax.arrow(
    #     T_20_B[0],
    #     T_20_B[1],
    #     (iterate_20[6] - T_20_B[0]) * GAP,
    #     (iterate_20[7] - T_20_B[1]) * GAP,
    #     width=WIDTH,
    #     head_width=HEADWIDTH,
    #     head_length=HEADLENGTH,
    #     length_includes_head=True,
    #     fc="white",
    #     ec="white",
    #     zorder=4,
    # )

    # # Draw $t=20$ at T_20_C and draw two arrows to iterate_20
    # T_20_C = [0.6, 0.1]
    # ax.text(
    #     T_20_C[0],
    #     T_20_C[1],
    #     f"$t=20$",
    #     fontsize=FONTSIZE,
    #     color="white",
    #     ha="left",
    #     va="bottom",
    # )
    # ax.arrow(
    #     T_20_C[0],
    #     T_20_C[1],
    #     (iterate_20[0] - T_20_C[0]) * GAP,
    #     (iterate_20[1] - T_20_C[1]) * GAP,
    #     width=WIDTH,
    #     head_width=HEADWIDTH,
    #     head_length=HEADLENGTH,
    #     length_includes_head=True,
    #     fc="white",
    #     ec="white",
    #     zorder=4,
    # )
    # ax.arrow(
    #     T_20_C[0],
    #     T_20_C[1],
    #     (iterate_20[2] - T_20_C[0]) * GAP,
    #     (iterate_20[3] - T_20_C[1]) * GAP,
    #     width=WIDTH,
    #     head_width=HEADWIDTH,
    #     head_length=HEADLENGTH,
    #     length_includes_head=True,
    #     fc="white",
    #     ec="white",
    #     zorder=4,
    # )
    # Plot the iterates
    n_q = iterates.shape[1] // 2
    for j in range(n_q):
        # plot arrows for iterates
        plt.plot(
            iterates[:, 2 * j], iterates[:, 2 * j + 1], "o-", color="red", markersize=4
        )

    fp = os.path.join(plots_dir, "iterates.svg")
    logging.info("plot_iterates: Saving iterates plot to %s", fp)
    plt.savefig(fp, bbox_inches="tight")
    plt.clf()


def plot_residuals(residuals: jnp.array, plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))

    c = get_discrete_cmap(3, "parula")
    plt.plot(residuals, color=c[0])

    plt.yscale("log")
    plt.xlabel("Iteration $t$", fontsize=20)
    plt.ylabel(
        "$ \\| \\mathcal{F}[\\theta^*] - \\mathcal{F}[\\theta_t] \\|_2^2$", fontsize=20
    )
    # Make the x-ticks integers [0, 5, 10, 15]
    plt.xticks(np.arange(0, 20, 5))
    # Turn off the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(os.path.join(plots_dir, "residuals.svg"), bbox_inches="tight")
    plt.clf()


def main(args: argparse.Namespace) -> None:

    logging.info("Plots will be saved to %s", args.plots_dir)

    # Set up boolean array for observation points

    observation_pts = SAMPLE_TREE.leaf_cheby_points[OBSERVATION_BOOLS].reshape(-1, 2)
    logging.debug("Observation points has shape %s", observation_pts.shape)

    # Set up ground-truth scatterer locations by copying in from terminal on the cluster.
    # I can't figure out how to get the jax PRNG to spit out the same random numbers on
    # cluster and my local machine.
    ground_truth_locations = jnp.array(
        [
            [0.3653929, -0.04679559],
            [-0.26146232, -0.29576259],
            [-0.18262641, 0.07377936],
            [0.3619092, 0.11641294],
        ]
    )
    logging.info("Ground-truth scatterer locations: %s", ground_truth_locations)

    q_evals_hps = q_point_sources(SAMPLE_TREE.leaf_cheby_points, ground_truth_locations)
    n_X = 200
    corners = jnp.array([[XMIN, YMIN], [XMAX, YMIN], [XMAX, YMAX], [XMIN, YMAX]])
    # q_evals_regular, regular_grid = interp_from_hps_to_regular_grid(
    # L, P, corners, XMIN, XMAX, YMIN, YMAX, q_evals_hps, n_X
    # )
    q_evals_regular, regular_grid = interp_from_nonuniform_hps_to_regular_grid(
        root=SAMPLE_TREE.root, p=P, f_evals=q_evals_hps, n_pts=n_X
    )

    # Load the iterates
    iterates_fp = os.path.join(args.plots_dir, "iterates_data.mat")
    dd = loadmat(iterates_fp)
    iterates = dd["iterates"]
    logging.debug("iterates has shape %s", iterates.shape)

    # Plot ground-truth q and iterates
    logging.info("Plotting iterates")
    plot_iterates(iterates, args.plots_dir, q_evals_regular)

    uscat_gt, _ = source_locations_to_scattered_field(ground_truth_locations)
    uscat_regular, regular_grid = interp_from_nonuniform_hps_to_regular_grid(
        root=SAMPLE_TREE.root, p=P, f_evals=uscat_gt, n_pts=n_X
    )
    logging.info("Plotting uscat")
    uin_regular = get_uin(K, regular_grid, jnp.array([0.0])[..., 0])
    utot_regular = uin_regular + uscat_regular
    plot_uscat(utot_regular, observation_pts, args.plots_dir)

    logging.info("Plotting residuals")
    residuals = dd["resid_norms"].flatten()
    plot_residuals(residuals, args.plots_dir)


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
