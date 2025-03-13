import os
import logging
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# supress matplotlib debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("jax").setLevel(logging.WARNING)

from hps.src.logging_utils import FMT, TIMEFMT, DATESTR
from hps.src.plotting import (
    get_discrete_cmap,
    plot_func_with_grid,
    FIGSIZE_2,
    FONTSIZE_2,
    TICKSIZE_2,
)
from hps.accuracy_checks.test_cases_3D import (
    d_xx_adaptive_meshing_data_fn,
    d_yy_adaptive_meshing_data_fn,
    d_zz_adaptive_meshing_data_fn,
)

K_TOL = "tol"
K_N_LEAVES = "n_leaves"
K_MAX_DEPTHS = "max_depths"
K_MESH_TIMES = "mesh_times"
K_LOCAL_SOLVE_TIMES = "local_solve_times"
K_BUILD_TIMES = "build_times"
K_DOWN_PASS_TIMES = "down_pass_times"


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="data/adaptive_meshing_3D")
    parser.add_argument("--l2", action="store_true")
    parser.add_argument("--mesh_tol", type=float, default=1e-2)
    parser.add_argument("--mesh_p", type=int, default=16)

    return parser.parse_args()


def source_fn(x: jnp.array) -> jnp.array:
    return (
        d_xx_adaptive_meshing_data_fn(x)
        + d_yy_adaptive_meshing_data_fn(x)
        + d_zz_adaptive_meshing_data_fn(x)
    )


def runtime_bar_chart(
    tol_vals: jnp.array,
    local_solve_times: jnp.array,
    build_times: jnp.array,
    down_pass_times: jnp.array,
    plot_dir: str,
) -> None:
    """
    For each entry in tol_vals, make a bar chart where local_solve_times is in blue on the bottom,
    build_times is in orange in the middle, and down_pass_times is in green on top.

    Convert tol_vals to strings for the x-axis tick labels.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    width = 0.35
    x = [str(tol) for tol in tol_vals]
    ax.bar(x, local_solve_times, width, label="Local Solve Times", color="b")
    ax.bar(
        x,
        build_times,
        width,
        label="Merge Times",
        color="orange",
        bottom=local_solve_times,
    )
    ax.bar(
        x,
        down_pass_times,
        width,
        label="Down Pass Times",
        color="g",
        bottom=local_solve_times + build_times,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tol_vals)
    # set ylim 1, 10^3
    ax.set_ylim(1, 10**2)
    ax.set_yscale("log")
    ax.set_xlabel("Tolerance", fontsize=FONTSIZE_2)
    ax.set_ylabel("Time (s)", fontsize=FONTSIZE_2)
    ax.set_title("Runtime Breakdown", fontsize=FONTSIZE_2)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "runtime_breakdown.svg"))
    plt.close(fig)


def plot_tol_vs_error(
    tol_vals: jnp.array,
    adaptive_errors: jnp.array,
    p_vals: jnp.array,
    plot_fp: str,
    nrm_str: str,
) -> None:
    """
    Plot the tolerance vs the measured error. The horizontal axis is the tolerance, and the vertical
    axis is the relative error.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    colors = get_discrete_cmap(len(p_vals), "plasma")
    for i, p in enumerate(p_vals):
        p_str = f"$p = {int(p)}$"
        ax.plot(tol_vals[i], adaptive_errors[i], ".-", color=colors[i], label=p_str)
        ax.plot(tol_vals[i], tol_vals[i], "--", color="black")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tolerance", fontsize=FONTSIZE_2)
    ax.set_ylabel(f"Relative {nrm_str} Error", fontsize=FONTSIZE_2)

    # Set top and right spines invisible
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_fp)
    plt.close(fig)


def plot_different_meshing_strats(
    adaptive_errors: jnp.array,
    adaptive_runtimes: jnp.array,
    uniform_errors: jnp.array,
    uniform_runtimes: jnp.array,
    p_vals_adaptive: jnp.array,
    p_vals_uniform: jnp.array,
    plot_fp: str,
    nrm_str: str,
) -> None:
    """
    horizontal axis is runtime, vertical axis is error. For each p, plot adaptive and uniform
    refinement on the same plot.

    Args:
        adaptive_errors (jnp.array): Has shape (n_p_vals, n_tol_vals)
        adaptive_runtimes (jnp.array): Has shape (n_p_vals, n_tol_vals)
        uniform_errors (jnp.array): Has shape (n_p_vals, n_tol_vals)
        uniform_runtimes (jnp.array): Has shape (n_p_vals, n_tol_vals)
        p_vals (jnp.array): Has shape (n_p_vals,)
        plot_dir (str): _description_
    """

    fig, ax = plt.subplots(figsize=(FIGSIZE_2, FIGSIZE_2))

    # Get cmap
    cmap = get_discrete_cmap(len(p_vals_adaptive), "parula")
    for i, p in enumerate(p_vals_adaptive):
        p_str = f"$p = {int(p)}$"
        adaptive_errors_i = adaptive_errors[i]
        adaptive_runtimes_i = adaptive_runtimes[i]

        ax.plot(
            adaptive_runtimes_i,
            adaptive_errors_i,
            ".-",
            color=cmap[i],
            label="Adaptive; " + p_str,
        )

    for i, p in enumerate(p_vals_uniform):
        p_str = f"$p = {int(p)}$"
        uniform_errors_i = uniform_errors[i]
        uniform_runtimes_i = uniform_runtimes[i]
        ax.plot(
            uniform_runtimes_i,
            uniform_errors_i,
            "x--",
            color=cmap[i],
            label="Uniform; " + p_str,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylabel(f"Relative {nrm_str} Error", fontsize=FONTSIZE_2)
    ax.set_xlabel("Runtime (s)", fontsize=FONTSIZE_2)

    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE_2)

    # Set top and right spines invisible
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_fp, bbox_inches="tight")


def main(args: argparse.Namespace) -> None:

    # Load the adaptive refinement data.
    adaptive_str = "adaptive_data"
    if args.l2:
        nrm_str = "_l2"
        nrm_str_plt = "$\\ell_2$"
        K_ERRORS = "l2_errors"
    else:
        nrm_str = "_linf"
        K_ERRORS = "linf_errors"
        nrm_str_plt = "$\\ell_\infty$"

    adaptive_errors = []
    adaptive_runtimes = []
    adaptive_tols = []

    p_vals = []

    for p in [8, 12, 16]:
        try:  # Load the data
            data_fp = os.path.join(
                args.plot_dir, f"p_{p}", f"{adaptive_str}{nrm_str}_p_{p}.mat"
            )
            data_dd = loadmat(data_fp)
            adaptive_errors.append(data_dd[K_ERRORS].flatten())
            adaptive_runtimes.append(
                data_dd[K_MESH_TIMES].flatten()
                + data_dd[K_LOCAL_SOLVE_TIMES].flatten()
                + data_dd[K_BUILD_TIMES].flatten()
                + data_dd[K_DOWN_PASS_TIMES].flatten()
            )
            adaptive_tols.append(data_dd[K_TOL].flatten())
            p_vals.append(p)

            # These don't change, just need to load them once
            eval_pts_x = data_dd["eval_pts_x"]
            eval_pts_y = data_dd["eval_pts_y"]
            eval_pts_z = data_dd["eval_pts_z"]

            # Flip the first axis
            eval_pts_x = jnp.flip(eval_pts_x, axis=0)
            eval_pts_y = jnp.flip(eval_pts_y, axis=0)
            eval_pts_z = jnp.flip(eval_pts_z, axis=0)

        except FileNotFoundError:
            logging.warning(f"Could not find {data_fp}")
            continue

    # adaptive_errors = jnp.array(adaptive_errors)
    # adaptive_runtimes = jnp.array(adaptive_runtimes)
    # adaptive_tols = jnp.array(adaptive_tols)
    p_vals_adaptive = jnp.array(p_vals)

    # Load the uniform refinement data
    uniform_errors = []
    uniform_runtimes = []
    p_vals_uniform = []
    for p in [8, 12, 16]:
        p_int = int(p)
        try:
            data_fp = os.path.join(args.plot_dir, f"p_{p}", f"hp_data_p_{p_int}.mat")
            data_dd = loadmat(data_fp)
            uniform_errors.append(data_dd[K_ERRORS].flatten())
            uniform_runtimes.append(
                data_dd[K_LOCAL_SOLVE_TIMES].flatten()
                + data_dd[K_BUILD_TIMES].flatten()
                + data_dd[K_DOWN_PASS_TIMES].flatten()
            )
            p_vals_uniform.append(p)

        except FileNotFoundError:
            logging.warning(f"Could not find {data_fp}")
            continue
    p_vals_uniform = jnp.array(p_vals_uniform)

    # uniform_errors = jnp.array(uniform_errors)
    # uniform_runtimes = jnp.array(uniform_runtimes)

    # Plot tolerance vs measured error
    logging.info("Plotting tolerance vs error")
    plot_fp = os.path.join(args.plot_dir, f"tol_vs_error{nrm_str}.svg")
    plot_tol_vs_error(
        adaptive_errors=adaptive_errors,
        tol_vals=adaptive_tols,
        p_vals=p_vals,
        plot_fp=plot_fp,
        nrm_str=nrm_str_plt,
    )

    # Plot uniform vs adaptive meshing
    logging.info("Plotting uniform vs adaptive meshing")
    plot_fp = os.path.join(args.plot_dir, f"adaptive_vs_uniform{nrm_str}.svg")
    plot_different_meshing_strats(
        adaptive_errors=adaptive_errors,
        adaptive_runtimes=adaptive_runtimes,
        uniform_errors=uniform_errors,
        uniform_runtimes=uniform_runtimes,
        p_vals_adaptive=p_vals_adaptive,
        p_vals_uniform=p_vals_uniform,
        plot_fp=plot_fp,
        nrm_str=nrm_str_plt,
    )

    # Plot the mesh on top of the source function
    logging.info("Plotting mesh with grid")
    mesh_fp = os.path.join(
        args.plot_dir,
        f"p_{args.mesh_p}",
        f"tree_p_{args.mesh_p}_tol_{args.mesh_tol}.mat",
    )
    logging.info("Loading mesh from %s", mesh_fp)
    mesh_dd = loadmat(mesh_fp)

    logging.info("Plotting rho with grid")
    corners_x = mesh_dd["x_leaf_corners"][:, :, [1, 2]]
    logging.debug("corners_x shape: %s", corners_x.shape)
    fp_x = os.path.join(args.plot_dir, "source_with_grid_x.svg")
    plot_func_with_grid(
        eval_pts_x,
        corners_x,
        "$x_2$",
        "$x_3$",
        fp_x,
        source_fn,
        bwr_cmap=False,
        figsize=FIGSIZE_2,
        fontsize=FONTSIZE_2,
        ticksize=TICKSIZE_2,
    )

    corners_y = mesh_dd["y_leaf_corners"][:, :, [0, 2]]
    fp_y = os.path.join(args.plot_dir, "source_with_grid_y.svg")
    plot_func_with_grid(
        eval_pts_y,
        corners_y,
        "$x_1$",
        "$x_3$",
        fp_y,
        source_fn,
        bwr_cmap=False,
        figsize=FIGSIZE_2,
        fontsize=FONTSIZE_2,
        ticksize=TICKSIZE_2,
    )

    corners_z = mesh_dd["z_leaf_corners"][:, :, [0, 1]]
    fp_z = os.path.join(args.plot_dir, "source_with_grid_z.svg")
    plot_func_with_grid(
        eval_pts_z,
        corners_z,
        "$x_1$",
        "$x_2$",
        fp_z,
        source_fn,
        bwr_cmap=False,
        figsize=FIGSIZE_2,
        fontsize=FONTSIZE_2,
        ticksize=TICKSIZE_2,
    )

    logging.info("Finished")


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
