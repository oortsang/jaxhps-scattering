from typing import Callable, List, Tuple, Dict
import os
import pickle
import logging
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# supress matplotlib debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("jax").setLevel(logging.WARNING)

from hps.src.logging_utils import FMT, TIMEFMT, DATESTR
from hps.src.plotting import plot_func_with_grid
from hps.src.quadrature.trees import Node, get_all_leaves
from hps.src.poisson_boltzmann_eqn_helpers import rho, permittivity, vdw_permittivity

K_TOL = "tol"
K_ERRORS = "errors"
K_N_LEAVES = "n_leaves"
K_MAX_DEPTHS = "max_depths"
K_LOCAL_SOLVE_TIMES = "local_solve_times"
K_BUILD_TIMES = "build_times"
K_DOWN_PASS_TIMES = "down_pass_times"
K_MESH_TIMES = "mesh_times"


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="data/poisson_boltzmann_3D")
    parser.add_argument(
        "--mesh_tol",
        type=float,
        default=1e-2,
        help="The tolerance at which we will draw the adaptive mesh",
    )
    parser.add_argument(
        "--mesh_p", type=int, default=12, help="The polynomial degree of the tree"
    )
    return parser.parse_args()


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
    ax.set_ylim(1, 3 * 10**2)
    ax.set_yscale("log")
    ax.set_xlabel("Tolerance", fontsize=LABELSIZE)
    ax.set_ylabel("Time (s)", fontsize=LABELSIZE)
    ax.set_title("Runtime Breakdown", fontsize=LABELSIZE)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "runtime_breakdown.png"))
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    """
    We want plots of:
    1. rho with grid
    2. permittivity with grid
    3. Solution
    4. Also want a table of | p | tol | # leaves | max depth | runtime |
    """
    # Determine whether we are in the vdW setting
    if "vdw" in args.plot_dir:
        perm_fn = vdw_permittivity
    else:
        perm_fn = permittivity

    # Load the data into a dataframe
    df_lst = []
    for p in [8, 10, 12, 16]:
        data_fp = os.path.join(args.plot_dir, f"p_{p}", "data.mat")
        data_dd = loadmat(data_fp)
        tol_vals = data_dd[K_TOL].flatten()
        p_vals = np.ones_like(tol_vals) * p
        max_depth_vals = data_dd[K_MAX_DEPTHS].flatten()
        n_leaves_vals = data_dd[K_N_LEAVES].flatten()
        local_solve_times = data_dd[K_LOCAL_SOLVE_TIMES].flatten()
        build_times = data_dd[K_BUILD_TIMES].flatten()
        down_pass_times = data_dd[K_DOWN_PASS_TIMES].flatten()
        mesh_times = data_dd[K_MESH_TIMES].flatten()
        df = pd.DataFrame(
            {
                "tol": tol_vals,
                "p": p_vals,
                "max_depth": max_depth_vals,
                "n_leaves": n_leaves_vals,
                "local_solve_time": local_solve_times,
                "build_time": build_times,
                "down_pass_time": down_pass_times,
                "mesh_time": mesh_times,
            }
        )
        df_lst.append(df)

    df = pd.concat(df_lst)
    df["runtime"] = (
        df["local_solve_time"]
        + df["build_time"]
        + df["down_pass_time"]
        + df["mesh_time"]
    )
    df = df.sort_values(by=["p", "tol"])
    df = df.drop(columns=["local_solve_time", "build_time", "down_pass_time"])
    df = df[["p", "tol", "n_leaves", "max_depth", "runtime"]]
    print(df.to_latex(index=False))
    mesh_fp = os.path.join(
        args.plot_dir,
        f"p_{args.mesh_p}",
        f"tree_p_{args.mesh_p}_tol_{args.mesh_tol}.mat",
    )
    logging.info("Loading mesh from %s", mesh_fp)
    mesh_dd = loadmat(mesh_fp)

    # Plot rho with grid
    logging.info("Plotting rho with grid")
    eval_pts_x = data_dd["eval_pts_x"]
    corners_x = mesh_dd["x_leaf_corners"][:, :, [1, 2]]
    logging.debug("corners_x shape: %s", corners_x.shape)
    fp_x = os.path.join(args.plot_dir, "rho_with_grid_x.svg")
    plot_func_with_grid(eval_pts_x, None, "$x_2$", "$x_3$", fp_x, rho)

    eval_pts_y = data_dd["eval_pts_y"]
    corners_y = mesh_dd["y_leaf_corners"][:, :, [0, 2]]
    fp_y = os.path.join(args.plot_dir, "rho_with_grid_y.svg")
    plot_func_with_grid(eval_pts_y, None, "$x_1$", "$x_3$", fp_y, rho)

    eval_pts_z = data_dd["eval_pts_z"]
    corners_z = mesh_dd["z_leaf_corners"][:, :, [0, 1]]
    fp_z = os.path.join(args.plot_dir, "rho_with_grid_z.svg")
    plot_func_with_grid(eval_pts_z, None, "$x_1$", "$x_2$", fp_z, rho)

    # Plot permittivity with grid
    logging.info("Plotting permittivity with grid")
    fp_x = os.path.join(args.plot_dir, "perm_with_grid_x.svg")
    plot_func_with_grid(eval_pts_x, corners_x, "$x_2$", "$x_3$", fp_x, perm_fn)

    fp_y = os.path.join(args.plot_dir, "perm_with_grid_y.svg")
    plot_func_with_grid(eval_pts_y, corners_y, "$x_1$", "z", fp_y, perm_fn)

    fp_z = os.path.join(args.plot_dir, "perm_with_grid_z.svg")
    plot_func_with_grid(eval_pts_z, corners_z, "$x_1$", "$x_2$", fp_z, perm_fn)

    # Make a dataframe of the data
    logging.info("Making a dataframe of the data")

    # Plot tolerance vs # leaves
    # logging.info("Plotting tolerance vs # leaves")

    # fp_n_leaves = os.path.join(args.plot_dir, "n_leaves.png")
    # tol = data_dd[K_TOL].flatten()
    # n_leaves = data_dd[K_N_LEAVES].flatten()
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.plot(1 / tol, n_leaves, ".-")
    # ax.set_xscale("log")
    # ax.set_xlabel("1 / Tolerance", fontsize=LABELSIZE)
    # ax.set_ylabel("# Leaves", fontsize=LABELSIZE)
    # ax.set_title("Tolerance vs # Leaves", fontsize=LABELSIZE)
    # ax.grid()
    # fig.tight_layout()
    # fig.savefig(fp_n_leaves)
    # plt.close(fig)

    # # Plot tolerance vs max depth
    # logging.info("Plotting tolerance vs max depth")
    # fp_max_depths = os.path.join(args.plot_dir, "max_depths.png")
    # max_depths = data_dd[K_MAX_DEPTHS].flatten()
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.plot(1 / tol, max_depths, ".-")
    # ax.set_xscale("log")
    # ax.set_xlabel("1 / Tolerance", fontsize=LABELSIZE)
    # ax.set_ylabel("Max Depth", fontsize=LABELSIZE)
    # ax.set_title("Tolerance vs Max Depth", fontsize=LABELSIZE)
    # ax.grid()
    # fig.tight_layout()
    # fig.savefig(fp_max_depths)

    # # Plot the runtime breakdown
    # logging.info("Plotting the runtime breakdown")
    # runtime_bar_chart(
    #     tol,
    #     data_dd[K_LOCAL_SOLVE_TIMES].flatten(),
    #     data_dd[K_BUILD_TIMES].flatten(),
    #     data_dd[K_DOWN_PASS_TIMES].flatten(),
    #     args.plot_dir,
    # )
    logging.info("Finished")


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
