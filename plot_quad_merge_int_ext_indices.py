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
from hps.src.plotting import get_discrete_cmap
from hps.src.quadrature.trees import Node, add_four_children
from hps.src.quadrature.quad_2D.grid_creation import (
    get_all_boundary_gauss_legendre_points,
    get_all_leaf_2d_cheby_points,
)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="data/quad_merge_indices")
    parser.add_argument("-p", type=int, default=8)
    return parser.parse_args()


FONTSIZE_OMEGA = 20


def plot_quad_merge_indices_main_text(q: int, fp_out: str) -> None:
    # Set up a Node with four children on [-1,1]^2
    root = Node(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, zmin=None, zmax=None, depth=0)
    add_four_children(root)

    # Get the boundary points for each child
    boundary_points = jnp.concatenate(
        [
            get_all_boundary_gauss_legendre_points(q=q, root=root.children[i])
            for i in range(4)
        ]
    )

    # Plot the boundary points.
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the boundary points. If the boundary points x val = 0 or y val = 0, plot them in red.
    # Otherwise, plot blue.
    x_zero_bools = boundary_points[:, 0] == 0
    y_zero_bools = boundary_points[:, 1] == 0
    boundary_points_red = boundary_points[jnp.logical_or(x_zero_bools, y_zero_bools)]
    boundary_points_blue = boundary_points[
        jnp.logical_and(jnp.logical_not(x_zero_bools), jnp.logical_not(y_zero_bools))
    ]
    print(
        "plot_quad_merge_indices_main_text: boundary_points_red.shape",
        boundary_points_red.shape,
    )
    print(
        "plot_quad_merge_indices_main_text: boundary_points_blue.shape",
        boundary_points_blue.shape,
    )
    ax.scatter(
        boundary_points_red[:, 0], boundary_points_red[:, 1], marker="x", color="red"
    )
    ax.scatter(
        boundary_points_blue[:, 0], boundary_points_blue[:, 1], marker=".", color="blue"
    )
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    # Turn off all splines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Plot text $\\Omega_a$ at [-0.5, -0.5]
    ax.text(-0.5, -0.5, "$\\Omega_a$", fontsize=FONTSIZE_OMEGA, color="black")
    ax.text(0.5, -0.5, "$\\Omega_b$", fontsize=FONTSIZE_OMEGA, color="black")
    ax.text(0.5, 0.5, "$\\Omega_c$", fontsize=FONTSIZE_OMEGA, color="black")
    ax.text(-0.5, 0.5, "$\\Omega_d$", fontsize=FONTSIZE_OMEGA, color="black")

    # Turn off all ticks
    ax.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False
    )
    # Turn off all tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(fp_out, bbox_inches="tight")


def plot_interior_points_main_text(p: int, fp_out: str) -> None:
    # Set up a Node with four children on [-1,1]^2
    root = Node(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, zmin=None, zmax=None, depth=0)
    add_four_children(root)

    # Get the boundary points for each child
    cheby_points = get_all_leaf_2d_cheby_points(p=p, root=root)
    cheby_points = cheby_points.reshape(-1, 2)

    # Plot the boundary points.
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the interior points
    ax.scatter(cheby_points[:, 0], cheby_points[:, 1], marker=".", color="black")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    # Turn off all splines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Plot text $\\Omega_a$ at [-0.5, -0.5]
    # ax.text(-0.5, -0.5, "$\\Omega_a$", fontsize=FONTSIZE_OMEGA, color="black")
    # ax.text(0.5, -0.5, "$\\Omega_b$", fontsize=FONTSIZE_OMEGA, color="black")
    # ax.text(0.5, 0.5, "$\\Omega_c$", fontsize=FONTSIZE_OMEGA, color="black")
    # ax.text(-0.5, 0.5, "$\\Omega_d$", fontsize=FONTSIZE_OMEGA, color="black")

    # Turn off all ticks
    ax.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False
    )
    # Turn off all tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(fp_out, bbox_inches="tight")


def main(args: argparse.Namespace) -> None:

    # Set up the plotting directory
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    # Plot the version for the main text
    fp_out = os.path.join(args.plot_dir, "quad_merge_indices_main_text.svg")
    plot_quad_merge_indices_main_text(args.p - 2, fp_out)

    # Plot interior points for the main text
    fp_out = os.path.join(args.plot_dir, "interior_points_main_text.svg")
    plot_interior_points_main_text(args.p, fp_out)


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
