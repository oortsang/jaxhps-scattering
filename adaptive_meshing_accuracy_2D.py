import os
from typing import Callable, Tuple, List
import sys
import argparse
import logging
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib

from hps.src.quadrature.quad_2D.adaptive_meshing import (
    generate_adaptive_mesh_l2,
    generate_adaptive_mesh_linf,
    node_corners_to_2d_corners,
    get_squared_l2_norm_single_panel,
)
from hps.src.solution_obj import create_solver_obj_2D, get_bdry_data_evals_lst_2D
from hps.src.up_down_passes import (
    local_solve_stage,
    build_stage,
    down_pass,
)
from hps.src.solution_obj import SolverObj
from hps.src.quadrature.trees import (
    Node,
    get_all_leaves_jitted,
)
from hps.src.quadrature.quad_2D.interpolation import (
    refinement_operator,
    interp_from_nonuniform_hps_to_uniform_grid,
)
from hps.src.plotting import plot_2D_adaptive_refinement
from hps.src.utils import meshgrid_to_lst_of_pts, points_to_2d_lst_of_points
from hps.accuracy_checks.dirichlet_neumann_data import (
    adaptive_meshing_data_fn,
    d_xx_adaptive_meshing_data_fn,
    d_yy_adaptive_meshing_data_fn,
    default_lap_coeffs,
)
from hps.accuracy_checks.h_refinement_functions import get_l_inf_error_2D
from hps.src.logging_utils import FMT, TIMEFMT


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="data/adaptive_meshing_2D")
    parser.add_argument(
        "--tol", type=float, nargs="+", default=[1e-02, 1e-04, 1e-06, 1e-08]
    )
    parser.add_argument("--hp_convergence", action="store_true")

    return parser.parse_args()


NORTH_OUTER = 1.0
SOUTH_OUTER = 0.0
EAST_OUTER = 1.0
WEST_OUTER = 0.0


def plot_diffs(
    t: SolverObj,
    p: int,
    plot_fp: str,
) -> None:
    n_x = 100

    u_reg, pts = interp_from_nonuniform_hps_to_uniform_grid(
        t.root, p, t.interior_solns, n_x
    )

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    TITLESIZE = 20

    im_0 = ax[0].imshow(
        u_reg, cmap="plasma", extent=[WEST_OUTER, EAST_OUTER, SOUTH_OUTER, NORTH_OUTER]
    )

    plt.colorbar(im_0, ax=ax[0])

    ax[0].set_title("Computed $u$", fontsize=TITLESIZE)

    im_1 = ax[1].imshow(
        adaptive_meshing_data_fn(pts).reshape(n_x, n_x),
        cmap="plasma",
        extent=[WEST_OUTER, EAST_OUTER, SOUTH_OUTER, NORTH_OUTER],
    )
    ax[1].set_title("Expected $u$", fontsize=TITLESIZE)

    plt.colorbar(im_1, ax=ax[1])

    # Plot absolute value of diffs
    im_2 = ax[2].imshow(
        np.abs(u_reg - adaptive_meshing_data_fn(pts).reshape(n_x, n_x)),
        cmap="hot",
        extent=[WEST_OUTER, EAST_OUTER, SOUTH_OUTER, NORTH_OUTER],
    )
    plt.colorbar(im_2, ax=ax[2])
    ax[2].set_title("Absolute Error", fontsize=TITLESIZE)
    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()
    plt.close(fig)


def plot_problem(plot_fp: str) -> None:

    n = 300
    x = jnp.linspace(WEST_OUTER, EAST_OUTER, n)
    y = jnp.linspace(SOUTH_OUTER, NORTH_OUTER, n)
    X, Y = jnp.meshgrid(x, jnp.flipud(y))
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    extent = [WEST_OUTER, EAST_OUTER, SOUTH_OUTER, NORTH_OUTER]
    u_evals = adaptive_meshing_data_fn(pts).reshape(n, n)
    lap_u_evals = d_xx_adaptive_meshing_data_fn(pts).reshape(
        n, n
    ) + d_yy_adaptive_meshing_data_fn(pts).reshape(n, n)
    max_lap_u_evals = jnp.max(jnp.abs(lap_u_evals))
    # plot u, lap_u
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    TITLESIZE = 20

    im_0 = ax[0].imshow(u_evals, cmap="plasma", extent=extent)
    plt.colorbar(im_0, ax=ax[0])
    ax[0].set_title("$u$", fontsize=TITLESIZE)
    im_1 = ax[1].imshow(lap_u_evals, cmap="bwr", extent=extent)
    # Set colorbar to have 0 in the middle
    im_1.set_clim(-max_lap_u_evals, max_lap_u_evals)
    plt.colorbar(im_1, ax=ax[1])
    ax[1].set_title("$\\Delta u$", fontsize=TITLESIZE)

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()
    plt.close(fig)


def source(x: jnp.array) -> jnp.array:
    lap_u = d_xx_adaptive_meshing_data_fn(x) + d_yy_adaptive_meshing_data_fn(x)
    return lap_u


def hp_convergence_test(plot_dir: str) -> None:
    # Generate a sequence of uniform meshes and compute the L_inf error
    # for each mesh.
    P = 12
    l_vals = [2, 3, 4]
    errors = jnp.zeros((len(l_vals),), dtype=jnp.float64)
    for i, l in enumerate(l_vals):
        root = Node(
            xmin=WEST_OUTER,
            xmax=EAST_OUTER,
            ymin=SOUTH_OUTER,
            ymax=NORTH_OUTER,
            zmin=None,
            zmax=None,
            depth=0,
        )
        t = create_solver_obj_2D(p=P, q=P - 2, root=root, uniform_levels=l)
        #############################################################
        # Do a PDE Solve
        source_evals = source(t.leaf_cheby_points)
        D_xx_evals = default_lap_coeffs(t.leaf_cheby_points)
        D_yy_evals = default_lap_coeffs(t.leaf_cheby_points)
        local_solve_stage(
            t, source_term=source_evals, D_xx_coeffs=D_xx_evals, D_yy_coeffs=D_yy_evals
        )
        build_stage(t)
        bdry_data = get_bdry_data_evals_lst_2D(t, f=adaptive_meshing_data_fn)
        down_pass(t, bdry_data)
        #############################################################
        # Compute the error
        expected_soln = adaptive_meshing_data_fn(t.leaf_cheby_points)
        computed_soln = t.interior_solns

        error = jnp.max(jnp.abs(expected_soln - computed_soln)) / jnp.max(
            jnp.abs(expected_soln)
        )
        errors = errors.at[i].set(error)

        #############################################################
        # Plot the diffs
        plot_fp = os.path.join(plot_dir, f"hp_convergence_soln_{l}.png")
        plot_diffs(t, P, plot_fp)

    h_vals = jnp.array([2**-l for l in l_vals])
    one_over_h = 1.0 / h_vals

    convergence_fp = os.path.join(plot_dir, "hp_convergence.png")
    logging.info("Errors: %s", errors)
    plt.plot(one_over_h, errors, "-o")
    plt.xlabel("$ 1 / h $")
    plt.ylabel("Relative L_inf error")
    plt.yscale("log")
    plt.title("H-P Convergence")
    plt.grid()
    plt.savefig(convergence_fp, bbox_inches="tight")


def main(args: argparse.Namespace) -> None:
    ############################################################
    # Set up output directory
    os.makedirs(args.plot_dir, exist_ok=True)
    ############################################################
    # Plot problem

    plot_fp = os.path.join(args.plot_dir, "soln_and_source.png")
    logging.info("Plotting problem to %s", plot_fp)
    plot_problem(plot_fp)

    #############################################################
    # Do adaptive refinement by looping over tol values
    n_tol_vals = len(args.tol)
    error_vals = jnp.zeros((n_tol_vals,), dtype=jnp.float64)

    P = 12
    interp = refinement_operator(P)
    for i, tol in enumerate(args.tol):
        root = Node(
            xmin=WEST_OUTER,
            xmax=EAST_OUTER,
            ymin=SOUTH_OUTER,
            ymax=NORTH_OUTER,
            zmin=None,
            zmax=None,
            depth=0,
        )
        generate_adaptive_mesh_linf(
            root=root,
            refinement_op=interp,
            f_fn=source,
            tol=tol,
            p=P,
            q=P - 2,
            level_restriction_bool=True,
        )
        logging.info("Generated adaptive mesh with L_inf error tolerance %s", tol)
        logging.info(
            "Adaptive mesh number of leaves: %s", len(get_all_leaves_jitted(root))
        )
        adaptive_grid_fp = os.path.join(args.plot_dir, f"adaptive_grid_tol_{tol}.png")
        logging.info("Plotting adaptive mesh to %s", adaptive_grid_fp)
        plot_2D_adaptive_refinement(
            source,
            root,
            P,
            title="Adaptive Mesh",
            fp=adaptive_grid_fp,
        )
        t = create_solver_obj_2D(p=P, q=P - 2, root=root)

        if args.hp_convergence:
            logging.info("Running H-P convergence test")
            hp_convergence_test(args.plot_dir)
            return

        #############################################################
        # Do a PDE Solve
        source_evals = source(t.leaf_cheby_points)
        D_xx_evals = default_lap_coeffs(t.leaf_cheby_points)
        D_yy_evals = default_lap_coeffs(t.leaf_cheby_points)
        local_solve_stage(
            t, source_term=source_evals, D_xx_coeffs=D_xx_evals, D_yy_coeffs=D_yy_evals
        )
        build_stage(t)
        bdry_data = get_bdry_data_evals_lst_2D(t, f=adaptive_meshing_data_fn)
        down_pass(t, bdry_data)

        #############################################################
        # Compute the error
        expected_soln = adaptive_meshing_data_fn(t.leaf_cheby_points)
        computed_soln = t.interior_solns

        error = jnp.max(jnp.abs(expected_soln - computed_soln)) / jnp.max(
            jnp.abs(expected_soln)
        )
        logging.info("Relative L_inf error: %s", error)

        error_vals = error_vals.at[i].set(error)

        #############################################################
        # Plot the solution

        plot_fp = os.path.join(args.plot_dir, f"soln_tol_{tol}.png")
        plot_diffs(t, P, plot_fp)

    #############################################################
    # Plot accuracy vs tol
    error_fp = os.path.join(args.plot_dir, "error_vs_tol.png")
    plt.plot(args.tol, error_vals, "-o")

    # Plot x = y line
    plt.plot(args.tol, args.tol, "--", label="x=y")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Specified Tolerance")
    plt.ylabel("Relative L_inf error")
    plt.title("Accuracy vs Tolerance")
    plt.grid()
    plt.legend()
    plt.savefig(error_fp, bbox_inches="tight")
    logging.info("Finished")


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
