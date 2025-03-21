import sys
import argparse
import os
import logging

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse.linalg import LinearOperator, lsqr, svds
from scipy.io import savemat, loadmat

# Disable all matplorlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Uncomment for debugging NaNs. Slows code down.
# jax.config.update("jax_debug_nans", True)

from hps.src.logging_utils import FMT, TIMEFMT
from hps.src.plotting import get_discrete_cmap, FIGSIZE_2, FONTSIZE_2, TICKSIZE_2
from hps.src.solver_obj import SolverObj, create_solver_obj_2D
from hps.src.up_down_passes import (
    local_solve_stage,
    build_stage,
    down_pass,
    fused_pde_solve_2D,
    fused_pde_solve_2D_ItI,
)
from hps.accuracy_checks.utils import plot_soln_from_cheby_nodes


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_dir", type=str, default="data/hp_convergence")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


XMIN = -1.0
XMAX = 1.0
YMIN = -1.0
YMAX = 1.0
CORNERS = jnp.array([[XMIN, YMIN], [XMAX, YMIN], [XMAX, YMAX], [XMIN, YMAX]])


def plot_problem_1(
    err_vals: jnp.array, l_vals: jnp.array, p_vals: jnp.array, plot_fp: str
) -> None:

    p_vals = p_vals.flatten()
    l_vals = l_vals.flatten()
    logging.info("plot_problem_1: p_vals: %s", p_vals)
    logging.info("plot_problem_1: l_vals: %s", l_vals)

    fig, ax = plt.subplots(figsize=(FIGSIZE_2, FIGSIZE_2))

    LABELSIZE = 20
    # Compute 1 / h values
    n_patches_per_side = 2**l_vals
    h_vals = (XMAX - XMIN) / n_patches_per_side
    h_inv_vals = 1 / h_vals

    # These are manually tuned to place the black dotted lines at the correct place.
    xmins = jnp.array([2.0, 5.0, 5.0, 3.0])
    xmaxes = jnp.array([18.0, 20.0, 20.0, 10.0])
    c = jnp.array([150.0, 2000.0, 2500.0, 380.0])

    xvals = np.zeros((p_vals.shape[0], 100), dtype=np.float64)
    yvals = np.zeros((p_vals.shape[0], 100), dtype=np.float64)

    # How many L vals to show for each order p.
    n_to_plot = jnp.array([6, 6, 6, 5, 4])

    cmap = get_discrete_cmap(len(p_vals), "parula")

    for i, p in enumerate(p_vals):
        xvals[i] = np.linspace(xmins[i], xmaxes[i], 100)
        yvals[i] = c[i] * (1 / xvals[i]) ** (p - 2)

    # For each value of p, plot the error vs 1 / h
    for i, p in enumerate(p_vals):
        n_i = n_to_plot[i]
        ax.plot(
            h_inv_vals[:n_i], err_vals[:n_i, i], ".-", color=cmap[i], label=f"$p={p}$"
        )

        if p == np.max(p_vals):
            # Only want to lable this once.
            plt.plot(xvals[i], yvals[i], "--", color="black", label="$O(h^{p - 2})$")
        else:
            plt.plot(xvals[i], yvals[i], "--", color="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$1/h$", fontsize=FONTSIZE_2)
    ax.set_ylabel("Relative $\\ell_\\infty$ Error", fontsize=FONTSIZE_2)
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE_2)

    # Turn off splines top and right
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.legend()
    ax.grid()

    plt.savefig(plot_fp, bbox_inches="tight")


def plot_problem_2(
    err_vals: jnp.array, l_vals: jnp.array, p_vals: jnp.array, plot_fp: str
) -> None:
    p_vals = p_vals.flatten()
    l_vals = l_vals.flatten()

    fig, ax = plt.subplots(figsize=(FIGSIZE_2, FIGSIZE_2))

    LABELSIZE = 20
    # Compute 1 / h values
    n_patches_per_side = 2**l_vals
    h_vals = (XMAX - XMIN) / n_patches_per_side
    h_inv_vals = 1 / h_vals

    # These are manually tuned to place the black dotted lines at the correct place.
    xmins = jnp.array([8.0, 5.0, 5.0, 3.0])
    xmaxes = jnp.array([30.0, 20.0, 20.0, 10.0])
    c = jnp.array([150.0, 1000.0, 1000.0, 1000.0])

    xvals = np.zeros((p_vals.shape[0], 100), dtype=np.float64)
    yvals = np.zeros((p_vals.shape[0], 100), dtype=np.float64)

    # How many L vals to show for each order p.
    n_to_plot = jnp.array([6, 6, 5, 4, 4])

    cmap = get_discrete_cmap(len(p_vals), "parula")

    for i, p in enumerate(p_vals):
        xvals[i] = np.linspace(xmins[i], xmaxes[i], 100)
        yvals[i] = c[i] * (1 / xvals[i]) ** (p - 2)

    # For each value of p, plot the error vs 1 / h
    for i, p in enumerate(p_vals):
        n_i = n_to_plot[i]
        ax.plot(
            h_inv_vals[:n_i], err_vals[:n_i, i], ".-", color=cmap[i], label=f"$p={p}$"
        )

        if p == np.max(p_vals):
            # Only want to lable this once.
            plt.plot(xvals[i], yvals[i], "--", color="black", label="$O(h^{p - 2})$")
        else:
            plt.plot(xvals[i], yvals[i], "--", color="black")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$1/h$", fontsize=FONTSIZE_2)
    ax.set_ylabel("Relative $\\ell_\\infty$ Error", fontsize=FONTSIZE_2)
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE_2)

    # Turn off splines top and right
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.legend()
    ax.grid()

    plt.savefig(plot_fp, bbox_inches="tight")


def main(args: argparse.Namespace) -> None:
    """
    1. Loads and plots the DtN data
    2. Loads and plots the ItI data
    """
    dtn_fp = os.path.join(args.plots_dir, "error_vals_DtN.mat")
    dtn_data = loadmat(dtn_fp)
    dtn_plot_fp = os.path.join(args.plots_dir, "hp_convergence_DtN.svg")
    plot_problem_1(
        err_vals=dtn_data["error_vals"],
        l_vals=dtn_data["l_vals"],
        p_vals=dtn_data["p_vals"],
        plot_fp=dtn_plot_fp,
    )

    iti_fp = os.path.join(args.plots_dir, "error_vals_ItI.mat")
    iti_data = loadmat(iti_fp)
    iti_plot_fp = os.path.join(args.plots_dir, "hp_convergence_ItI.svg")
    plot_problem_2(
        err_vals=iti_data["error_vals"],
        l_vals=iti_data["l_vals"],
        p_vals=iti_data["p_vals"],
        plot_fp=iti_plot_fp,
    )


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
