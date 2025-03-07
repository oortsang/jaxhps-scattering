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
from timeit import default_timer
from scipy.io import savemat

# Suppress matplotlib debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)

from hps.src.quadrature.quad_3D.adaptive_meshing import (
    generate_adaptive_mesh_level_restriction,
    node_corners_to_3d_corners,
    get_squared_l2_norm_single_voxel,
    find_leaves_containing_pts,
)
from hps.src.solver_obj import (
    create_solver_obj_3D,
    get_bdry_data_evals_lst_3D,
    SolverObj,
)
from hps.src.up_down_passes import (
    local_solve_stage,
    build_stage,
    down_pass,
)
from hps.src.quadrature.trees import (
    Node,
    get_all_leaves,
    get_all_leaves_jitted,
)
from hps.src.quadrature.quad_3D.interpolation import (
    refinement_operator,
    interp_from_nonuniform_hps_to_uniform_grid,
)
from hps.src.plotting import plot_2D_adaptive_refinement, plot_adaptive_grid_histogram
from hps.src.utils import meshgrid_to_lst_of_pts, points_to_2d_lst_of_points
from hps.accuracy_checks.test_cases_3D import (
    adaptive_meshing_data_fn,
    d_xx_adaptive_meshing_data_fn,
    d_yy_adaptive_meshing_data_fn,
    d_zz_adaptive_meshing_data_fn,
    default_lap_coeffs,
)
from hps.accuracy_checks.h_refinement_functions import get_l_inf_error_2D
from hps.src.logging_utils import FMT, TIMEFMT
from hps.src.config import HOST_DEVICE, DEVICE_ARR


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="data/adaptive_meshing_3D")
    parser.add_argument(
        "--tol", type=float, nargs="+", default=[1e-02, 1e-04, 1e-06, 1e-08]
    )
    parser.add_argument("-p", type=int, default=8)
    parser.add_argument("--hp_convergence", action="store_true")
    parser.add_argument("--max_l", type=int, default=4)
    parser.add_argument("--l2", action="store_true")
    parser.add_argument(
        "-n", type=int, default=500, help="Number of pixels to use when plotting"
    )

    return parser.parse_args()


XMIN = 0.0
XMAX = 1.0
YMIN = 0.0
YMAX = 1.0
ZMIN = 0.0
ZMAX = 1.0


def get_relative_l2_error(
    t: SolverObj,
    f_fn: Callable[[jnp.array], jnp.array],
) -> float:
    expected_soln = f_fn(t.leaf_cheby_points)
    computed_soln = t.interior_solns

    diffs = expected_soln - computed_soln

    all_leaves = get_all_leaves(t.root)
    all_corners = jnp.array([node_corners_to_3d_corners(l) for l in all_leaves])
    logging.debug("get_relative_l2_error: all_corners: %s", all_corners.shape)

    v = jax.vmap(get_squared_l2_norm_single_voxel, in_axes=(0, 0, None))

    squared_l2_norms_errors = v(diffs, all_corners, t.p)

    squared_l2_norms_exact = v(expected_soln, all_corners, t.p)

    error = jnp.sqrt(jnp.sum(squared_l2_norms_errors)) / jnp.sqrt(
        jnp.sum(squared_l2_norms_exact)
    )
    return error


def plot_diffs(
    u_reg_x: jnp.array,
    eval_pts_x: jnp.array,
    u_reg_y: jnp.array,
    eval_pts_y: jnp.array,
    u_reg_z: jnp.array,
    eval_pts_z: jnp.array,
    plot_fp: str,
) -> None:

    TITLESIZE = 20

    # Make a figure with 3 panels.
    # First row will be computed u restricted to x = 0, y = 0, z = 0
    # Second row will be expected u restricted to x = 0, y = 0, z = 0
    # Third row will be the absolute error
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))

    #############################################################
    # First column: Plot along x = 0.0

    u_expected_x = adaptive_meshing_data_fn(eval_pts_x)
    extent = [YMIN, YMAX, ZMIN, ZMAX]

    im_0 = ax[0, 0].imshow(u_reg_x, cmap="plasma", extent=extent)
    plt.colorbar(im_0, ax=ax[0, 0])
    ax[0, 0].set_title("Computed $u$", fontsize=TITLESIZE)
    ax[0, 0].set_xlabel("y", fontsize=TITLESIZE)
    ax[0, 0].set_ylabel("z", fontsize=TITLESIZE)

    im_1 = ax[1, 0].imshow(u_expected_x, cmap="plasma", extent=extent)
    plt.colorbar(im_1, ax=ax[1, 0])
    ax[1, 0].set_title("Expected $u$", fontsize=TITLESIZE)
    ax[1, 0].set_xlabel("y", fontsize=TITLESIZE)
    ax[1, 0].set_ylabel("z", fontsize=TITLESIZE)

    im_2 = ax[2, 0].imshow(np.abs(u_reg_x - u_expected_x), cmap="hot", extent=extent)
    plt.colorbar(im_2, ax=ax[2, 0])
    ax[2, 0].set_title("Absolute Error", fontsize=TITLESIZE)
    ax[2, 0].set_xlabel("y", fontsize=TITLESIZE)
    ax[2, 0].set_ylabel("z", fontsize=TITLESIZE)

    #############################################################
    # Second column: Plot along y = 0.0

    u_expected_y = adaptive_meshing_data_fn(eval_pts_y)
    extent = [XMIN, XMAX, ZMIN, ZMAX]

    im_0 = ax[0, 1].imshow(u_reg_y, cmap="plasma", extent=extent)
    plt.colorbar(im_0, ax=ax[0, 1])
    ax[0, 1].set_title("Computed $u$", fontsize=TITLESIZE)
    ax[0, 1].set_xlabel("x", fontsize=TITLESIZE)
    ax[0, 1].set_ylabel("z", fontsize=TITLESIZE)

    im_1 = ax[1, 1].imshow(u_expected_y, cmap="plasma", extent=extent)
    plt.colorbar(im_1, ax=ax[1, 1])
    ax[1, 1].set_title("Expected $u$", fontsize=TITLESIZE)
    ax[1, 1].set_xlabel("x", fontsize=TITLESIZE)
    ax[1, 1].set_ylabel("z", fontsize=TITLESIZE)

    im_2 = ax[2, 1].imshow(np.abs(u_reg_y - u_expected_y), cmap="hot", extent=extent)
    plt.colorbar(im_2, ax=ax[2, 1])
    ax[2, 1].set_title("Absolute Error", fontsize=TITLESIZE)
    ax[2, 1].set_xlabel("x", fontsize=TITLESIZE)
    ax[2, 1].set_ylabel("z", fontsize=TITLESIZE)

    #############################################################
    # Third column: Plot along z = 0.0

    u_expected_z = adaptive_meshing_data_fn(eval_pts_z)
    extent = [XMIN, XMAX, YMIN, YMAX]

    im_0 = ax[0, 2].imshow(u_reg_z, cmap="plasma", extent=extent)
    plt.colorbar(im_0, ax=ax[0, 2])
    ax[0, 2].set_title("Computed $u$", fontsize=TITLESIZE)
    ax[0, 2].set_xlabel("x", fontsize=TITLESIZE)
    ax[0, 2].set_ylabel("y", fontsize=TITLESIZE)

    im_1 = ax[1, 2].imshow(u_expected_z, cmap="plasma", extent=extent)
    plt.colorbar(im_1, ax=ax[1, 2])
    ax[1, 2].set_title("Expected $u$", fontsize=TITLESIZE)
    ax[1, 2].set_xlabel("x", fontsize=TITLESIZE)
    ax[1, 2].set_ylabel("y", fontsize=TITLESIZE)

    im_2 = ax[2, 2].imshow(np.abs(u_reg_z - u_expected_z), cmap="hot", extent=extent)
    plt.colorbar(im_2, ax=ax[2, 2])
    ax[2, 2].set_title("Absolute Error", fontsize=TITLESIZE)
    ax[2, 2].set_xlabel("x", fontsize=TITLESIZE)
    ax[2, 2].set_ylabel("y", fontsize=TITLESIZE)

    fig.tight_layout()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()
    plt.close(fig)


def plot_problem(plot_fp: str) -> None:

    # Set up part of the grid.
    n = 300
    x = jnp.linspace(XMIN, XMAX, n)
    y = jnp.linspace(YMIN, YMAX, n)
    z = jnp.linspace(ZMIN, ZMAX, n)
    TITLESIZE = 20

    # Make a figure with 3 panels. First row will be u restricted to x = 0, y = 0, z = 0
    # Second row will be lap u restricted to x = 0, y = 0, z = 0
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    #############################################################
    # First column: Plot along x = 0.0
    Y, Z = jnp.meshgrid(y, jnp.flipud(z))
    pts = jnp.stack([jnp.zeros_like(Y.flatten()), Y.flatten(), Z.flatten()], axis=-1)

    u_evals = adaptive_meshing_data_fn(pts).reshape(n, n)
    lap_u_evals = (
        d_xx_adaptive_meshing_data_fn(pts).reshape(n, n)
        + d_yy_adaptive_meshing_data_fn(pts).reshape(n, n)
        + d_zz_adaptive_meshing_data_fn(pts).reshape(n, n)
    )
    max_lap_u_evals = jnp.max(jnp.abs(lap_u_evals))
    extent = [YMIN, YMAX, ZMIN, ZMAX]
    # plot u, lap_u

    im_0 = ax[0, 0].imshow(u_evals, cmap="plasma", extent=extent)
    plt.colorbar(im_0, ax=ax[0, 0])
    ax[0, 0].set_title("$u$", fontsize=TITLESIZE)
    ax[0, 0].set_xlabel("y", fontsize=TITLESIZE)
    ax[0, 0].set_ylabel("z", fontsize=TITLESIZE)
    im_1 = ax[1, 0].imshow(lap_u_evals, cmap="bwr", extent=extent)
    # Set colorbar to have 0 in the middle
    im_1.set_clim(-max_lap_u_evals, max_lap_u_evals)
    plt.colorbar(im_1, ax=ax[1, 0])
    ax[1, 0].set_title("$\\Delta u$", fontsize=TITLESIZE)
    ax[1, 0].set_xlabel("y", fontsize=TITLESIZE)
    ax[1, 0].set_ylabel("z", fontsize=TITLESIZE)

    #############################################################
    # Second column: Plot along y = 0.0
    X, Z = jnp.meshgrid(x, jnp.flipud(z))
    pts = jnp.stack([X.flatten(), jnp.zeros_like(X.flatten()), Z.flatten()], axis=-1)

    u_evals = adaptive_meshing_data_fn(pts).reshape(n, n)
    lap_u_evals = (
        d_xx_adaptive_meshing_data_fn(pts).reshape(n, n)
        + d_yy_adaptive_meshing_data_fn(pts).reshape(n, n)
        + d_zz_adaptive_meshing_data_fn(pts).reshape(n, n)
    )
    max_lap_u_evals = jnp.max(jnp.abs(lap_u_evals))
    extent = [XMIN, XMAX, ZMIN, ZMAX]

    im_0 = ax[0, 1].imshow(u_evals, cmap="plasma", extent=extent)
    plt.colorbar(im_0, ax=ax[0, 1])
    ax[0, 1].set_title("$u$", fontsize=TITLESIZE)
    ax[0, 1].set_xlabel("x", fontsize=TITLESIZE)
    ax[0, 1].set_ylabel("z", fontsize=TITLESIZE)
    im_1 = ax[1, 1].imshow(lap_u_evals, cmap="bwr", extent=extent)
    # Set colorbar to have 0 in the middle
    im_1.set_clim(-max_lap_u_evals, max_lap_u_evals)
    plt.colorbar(im_1, ax=ax[1, 1])
    ax[1, 1].set_title("$\\Delta u$", fontsize=TITLESIZE)
    ax[1, 1].set_xlabel("x", fontsize=TITLESIZE)
    ax[1, 1].set_ylabel("z", fontsize=TITLESIZE)

    #############################################################
    # Second column: Plot along z = 0.0
    X, Y = jnp.meshgrid(x, jnp.flipud(y))
    pts = jnp.stack([X.flatten(), Y.flatten(), jnp.zeros_like(X.flatten())], axis=-1)

    u_evals = adaptive_meshing_data_fn(pts).reshape(n, n)
    lap_u_evals = (
        d_xx_adaptive_meshing_data_fn(pts).reshape(n, n)
        + d_yy_adaptive_meshing_data_fn(pts).reshape(n, n)
        + d_zz_adaptive_meshing_data_fn(pts).reshape(n, n)
    )
    max_lap_u_evals = jnp.max(jnp.abs(lap_u_evals))
    extent = [XMIN, XMAX, YMIN, YMAX]

    im_0 = ax[0, 2].imshow(u_evals, cmap="plasma", extent=extent)
    plt.colorbar(im_0, ax=ax[0, 2])
    ax[0, 2].set_title("$u$", fontsize=TITLESIZE)
    ax[0, 2].set_xlabel("x", fontsize=TITLESIZE)
    ax[0, 2].set_ylabel("y", fontsize=TITLESIZE)
    im_1 = ax[1, 2].imshow(lap_u_evals, cmap="bwr", extent=extent)
    # Set colorbar to have 0 in the middle
    im_1.set_clim(-max_lap_u_evals, max_lap_u_evals)
    plt.colorbar(im_1, ax=ax[1, 2])
    ax[1, 2].set_title("$\\Delta u$", fontsize=TITLESIZE)
    ax[1, 2].set_xlabel("x", fontsize=TITLESIZE)
    ax[1, 2].set_ylabel("y", fontsize=TITLESIZE)

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()
    plt.close(fig)


def source(x: jnp.array) -> jnp.array:
    lap_u = (
        d_xx_adaptive_meshing_data_fn(x)
        + d_yy_adaptive_meshing_data_fn(x)
        + d_zz_adaptive_meshing_data_fn(x)
    )
    return lap_u


def hp_convergence_study() -> None:
    #############################################################
    # Do adaptive refinement by looping over tol values
    linf_error_vals = jnp.zeros((args.max_l,), dtype=jnp.float64)
    l2_error_vals = jnp.zeros((args.max_l,), dtype=jnp.float64)
    local_solve_times = jnp.zeros((args.max_l,), dtype=jnp.float64)
    build_times = jnp.zeros((args.max_l,), dtype=jnp.float64)
    down_pass_times = jnp.zeros((args.max_l,), dtype=jnp.float64)
    D_size_vals = jnp.zeros((args.max_l,), dtype=jnp.int32)

    l_vals = jnp.arange(1, args.max_l + 1)

    for i, l in enumerate(l_vals):
        logging.info("Uniform refinement level %s", l)
        root = Node(
            xmin=XMIN,
            xmax=XMAX,
            ymin=YMIN,
            ymax=YMAX,
            zmin=ZMIN,
            zmax=ZMAX,
            depth=0,
        )

        t = create_solver_obj_3D(p=args.p, q=args.p - 2, root=root, uniform_levels=l)

        #############################################################
        # Do a PDE Solve
        source_evals = source(t.leaf_cheby_points)
        D_xx_evals = default_lap_coeffs(t.leaf_cheby_points)
        D_yy_evals = default_lap_coeffs(t.leaf_cheby_points)
        D_zz_evals = default_lap_coeffs(t.leaf_cheby_points)
        t0 = default_timer()
        local_solve_stage(
            t,
            source_term=source_evals,
            D_xx_coeffs=D_xx_evals,
            D_yy_coeffs=D_yy_evals,
            D_zz_coeffs=D_zz_evals,
        )
        t_local_solve = default_timer() - t0
        logging.info("Local solve stage in %f sec", t_local_solve)
        local_solve_times = local_solve_times.at[i].set(t_local_solve)

        t_0 = default_timer()
        D_size = build_stage(t, return_D_size=True)
        t_build = default_timer() - t_0
        logging.info("Build stage in %f sec", t_build)
        logging.info("D size = %s", D_size)
        build_times = build_times.at[i].set(t_build)
        D_size_vals = D_size_vals.at[i].set(D_size)
        bdry_data = get_bdry_data_evals_lst_3D(t, f=adaptive_meshing_data_fn)

        t_0 = default_timer()
        down_pass(t, bdry_data)
        t_down = default_timer() - t_0
        logging.info("Down pass in %f sec", t_down)
        down_pass_times = down_pass_times.at[i].set(t_down)

        #############################################################
        # Compute the error
        expected_soln = adaptive_meshing_data_fn(t.leaf_cheby_points)
        computed_soln = t.interior_solns

        linf_error = jnp.max(jnp.abs(expected_soln - computed_soln)) / jnp.max(
            jnp.abs(expected_soln)
        )
        logging.info("Relative L_inf error: %s", linf_error)
        linf_error_vals = linf_error_vals.at[i].set(linf_error)

        l2_error = get_relative_l2_error(t, adaptive_meshing_data_fn)
        logging.info("Relative L_2 error: %s", l2_error)
        l2_error_vals = l2_error_vals.at[i].set(l2_error)

        #############################################################
        # Interpolate to a uniform grid

        x = jnp.linspace(XMIN, XMAX, args.n)
        y = jnp.linspace(YMIN, YMAX, args.n)
        z = jnp.linspace(ZMIN, ZMAX, args.n)

        u_reg_x, eval_pts_x = interp_from_nonuniform_hps_to_uniform_grid(
            root=t.root,
            p=args.p,
            f_evals=t.interior_solns,
            to_x=jnp.array([0.0]),
            to_y=jnp.flipud(y),
            to_z=z,
        )
        eval_pts_x = eval_pts_x.reshape(args.n, args.n, 3)
        u_reg_x = u_reg_x.reshape(args.n, args.n)
        x_leaf_corners = find_leaves_containing_pts(eval_pts_x, t.root)
        logging.info("x_leaf_corners  shape: %s", x_leaf_corners.shape)

        u_reg_y, eval_pts_y = interp_from_nonuniform_hps_to_uniform_grid(
            root=t.root,
            p=args.p,
            f_evals=t.interior_solns,
            to_x=jnp.flipud(x),
            to_y=jnp.array([0.0]),
            to_z=z,
        )
        u_reg_y = u_reg_y.reshape(args.n, args.n)
        eval_pts_y = eval_pts_y.reshape(args.n, args.n, 3)
        y_leaf_corners = find_leaves_containing_pts(eval_pts_y, t.root)

        u_reg_z, eval_pts_z = interp_from_nonuniform_hps_to_uniform_grid(
            root=t.root,
            p=args.p,
            f_evals=t.interior_solns,
            to_x=x,
            to_y=jnp.flipud(y),
            to_z=jnp.array([0.0]),
        )
        u_reg_z = u_reg_z.reshape(args.n, args.n)
        eval_pts_z = eval_pts_z.reshape(args.n, args.n, 3)
        z_leaf_corners = find_leaves_containing_pts(eval_pts_z, t.root)

        #############################################################
        # Plot the solution

        plot_fp = os.path.join(args.plot_dir, f"hp_soln_l_{l}_p_{args.p}.png")
        plot_diffs(
            u_reg_x, eval_pts_x, u_reg_y, eval_pts_y, u_reg_z, eval_pts_z, plot_fp
        )

    # #############################################################
    # # Plot accuracy vs tol
    # error_fp = os.path.join(args.plot_dir, "hp_error_vs_l.png")
    # plt.plot(l_vals, linf_error_vals, "-o", label="Linf")
    # plt.plot(l_vals, l2_error_vals, "-o", label="L2")

    # # Plot x = y line
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.xlabel("# of Uniform Refinement Levels")
    # plt.ylabel("Relative L_inf error")
    # plt.title("Accuracy Under Uniform Refinement")
    # plt.grid()
    # plt.legend()
    # plt.savefig(error_fp, bbox_inches="tight")

    #############################################################
    # Save data
    save_fp = os.path.join(args.plot_dir, f"hp_data_p_{args.p}.mat")

    out_dd = {
        "l": l_vals,
        "linf_errors": linf_error_vals,
        "l2_errors": l2_error_vals,
        "local_solve_times": local_solve_times,
        "build_times": build_times,
        "down_pass_times": down_pass_times,
        "D_size_vals": D_size_vals,
    }
    savemat(save_fp, out_dd)


def adaptive_small_ex_for_jit() -> None:
    logging.info("Doing a small example for jitted computation")
    root = Node(
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        zmin=ZMIN,
        zmax=ZMAX,
        depth=0,
    )
    interp = refinement_operator(args.p)

    generate_adaptive_mesh_level_restriction(
        root=root,
        refinement_op=interp,
        f_fn=source,
        tol=0.1,  # Can't be too large or we get a depth-0 discretization
        p=args.p,
        q=args.p - 2,
        restrict_bool=True,
        l2_norm=args.l2,
    )

    t = create_solver_obj_3D(p=args.p, q=args.p - 2, root=root)

    #############################################################
    # Do a PDE Solve
    source_evals = source(t.leaf_cheby_points)
    D_xx_evals = default_lap_coeffs(t.leaf_cheby_points)
    D_yy_evals = default_lap_coeffs(t.leaf_cheby_points)
    D_zz_evals = default_lap_coeffs(t.leaf_cheby_points)
    local_solve_stage(
        t,
        source_term=source_evals,
        D_xx_coeffs=D_xx_evals,
        D_yy_coeffs=D_yy_evals,
        D_zz_coeffs=D_zz_evals,
    )
    D_size = build_stage(t, return_D_size=True)

    bdry_data = get_bdry_data_evals_lst_3D(t, f=adaptive_meshing_data_fn)
    down_pass(t, bdry_data)


def uniform_small_ex_for_jit() -> None:
    logging.info("Doing a small example for jitted computation")
    root = Node(
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        zmin=ZMIN,
        zmax=ZMAX,
        depth=0,
    )
    interp = refinement_operator(args.p)

    t = create_solver_obj_3D(p=args.p, q=args.p - 2, root=root, uniform_levels=1)

    #############################################################
    # Do a PDE Solve
    source_evals = source(t.leaf_cheby_points)
    D_xx_evals = default_lap_coeffs(t.leaf_cheby_points)
    D_yy_evals = default_lap_coeffs(t.leaf_cheby_points)
    D_zz_evals = default_lap_coeffs(t.leaf_cheby_points)
    local_solve_stage(
        t,
        source_term=source_evals,
        D_xx_coeffs=D_xx_evals,
        D_yy_coeffs=D_yy_evals,
        D_zz_coeffs=D_zz_evals,
    )
    D_size = build_stage(t, return_D_size=True)

    bdry_data = get_bdry_data_evals_lst_3D(t, f=adaptive_meshing_data_fn)
    down_pass(t, bdry_data)


def adaptive_convergence_study() -> None:
    #############################################################
    # Do adaptive refinement by looping over tol values
    n_tol_vals = len(args.tol)
    linf_error_vals = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    l2_error_vals = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    mesh_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    local_solve_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    build_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    down_pass_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    n_leaves = jnp.zeros((n_tol_vals,))
    max_depths = jnp.zeros((n_tol_vals,))
    D_size_vals = jnp.zeros((n_tol_vals,))

    nrm_str = "l2" if args.l2 else "linf"

    interp = refinement_operator(args.p)
    for i, tol in enumerate(args.tol):
        root = Node(
            xmin=XMIN,
            xmax=XMAX,
            ymin=YMIN,
            ymax=YMAX,
            zmin=ZMIN,
            zmax=ZMAX,
            depth=0,
        )
        t0 = default_timer()
        generate_adaptive_mesh_level_restriction(
            root=root,
            refinement_op=interp,
            f_fn=source,
            tol=tol,
            p=args.p,
            q=args.p - 2,
            restrict_bool=True,
            l2_norm=args.l2,
        )
        mesh_time = default_timer() - t0
        mesh_times = mesh_times.at[i].set(mesh_time)
        ll = get_all_leaves_jitted(root)
        logging.info("Generated adaptive mesh with error tolerance %s", tol)

        # Save the number of leaves and max depth
        depths = [l.depth for l in ll]
        n_leaves = n_leaves.at[i].set(len(ll))
        max_depths = max_depths.at[i].set(max(depths))

        logging.info(
            "Adaptive mesh number of leaves: %s and max depth: %s. Took %s sec",
            len(ll),
            max(depths),
            mesh_time,
        )

        # Plot a histogram of the side lengths of the leaves
        plot_fp = os.path.join(args.plot_dir, f"adaptive_mesh_hist_tol_{tol}.png")
        plot_adaptive_grid_histogram(root, plot_fp, tol, args.p)

        t = create_solver_obj_3D(p=args.p, q=args.p - 2, root=root)

        #############################################################
        # Do a PDE Solve
        source_evals = source(t.leaf_cheby_points)
        D_xx_evals = default_lap_coeffs(t.leaf_cheby_points)
        D_yy_evals = default_lap_coeffs(t.leaf_cheby_points)
        D_zz_evals = default_lap_coeffs(t.leaf_cheby_points)
        t0 = default_timer()
        local_solve_stage(
            t,
            source_term=source_evals,
            D_xx_coeffs=D_xx_evals,
            D_yy_coeffs=D_yy_evals,
            D_zz_coeffs=D_zz_evals,
        )
        t_local_solve = default_timer() - t0
        logging.info("Local solve stage in %f sec", t_local_solve)
        local_solve_times = local_solve_times.at[i].set(t_local_solve)

        t_0 = default_timer()
        D_size = build_stage(t, return_D_size=True)
        t_build = default_timer() - t_0
        logging.info("Build stage in %f sec", t_build)
        logging.info("D size = %s", D_size)
        D_size_vals = D_size_vals.at[i].set(D_size)

        build_times = build_times.at[i].set(t_build)
        bdry_data = get_bdry_data_evals_lst_3D(t, f=adaptive_meshing_data_fn)

        t_0 = default_timer()
        down_pass(t, bdry_data)
        t_down = default_timer() - t_0
        logging.info("Down pass in %f sec", t_down)
        down_pass_times = down_pass_times.at[i].set(t_down)

        #############################################################
        # Compute the error
        expected_soln = adaptive_meshing_data_fn(t.leaf_cheby_points)
        computed_soln = t.interior_solns

        linf_error = jnp.max(jnp.abs(expected_soln - computed_soln)) / jnp.max(
            jnp.abs(expected_soln)
        )
        logging.info("Relative L_inf error: %s", linf_error)
        linf_error_vals = linf_error_vals.at[i].set(linf_error)

        l2_error = get_relative_l2_error(t, adaptive_meshing_data_fn)
        logging.info("Relative L_2 error: %s", l2_error)
        l2_error_vals = l2_error_vals.at[i].set(l2_error)

        #############################################################
        # Interpolate to a uniform grid

        x = jnp.linspace(XMIN, XMAX, args.n)
        y = jnp.linspace(YMIN, YMAX, args.n)
        z = jnp.linspace(ZMIN, ZMAX, args.n)

        u_reg_x, eval_pts_x = interp_from_nonuniform_hps_to_uniform_grid(
            root=t.root,
            p=args.p,
            f_evals=t.interior_solns,
            to_x=jnp.array([0.0]),
            to_y=jnp.flipud(y),
            to_z=z,
        )
        eval_pts_x = eval_pts_x.reshape(args.n, args.n, 3)
        u_reg_x = u_reg_x.reshape(args.n, args.n)
        x_leaf_corners = find_leaves_containing_pts(eval_pts_x, t.root)
        logging.info("x_leaf_corners  shape: %s", x_leaf_corners.shape)

        u_reg_y, eval_pts_y = interp_from_nonuniform_hps_to_uniform_grid(
            root=t.root,
            p=args.p,
            f_evals=t.interior_solns,
            to_x=jnp.flipud(x),
            to_y=jnp.array([0.0]),
            to_z=z,
        )
        u_reg_y = u_reg_y.reshape(args.n, args.n)
        eval_pts_y = eval_pts_y.reshape(args.n, args.n, 3)
        y_leaf_corners = find_leaves_containing_pts(eval_pts_y, t.root)

        u_reg_z, eval_pts_z = interp_from_nonuniform_hps_to_uniform_grid(
            root=t.root,
            p=args.p,
            f_evals=t.interior_solns,
            to_x=x,
            to_y=jnp.flipud(y),
            to_z=jnp.array([0.0]),
        )
        u_reg_z = u_reg_z.reshape(args.n, args.n)
        eval_pts_z = eval_pts_z.reshape(args.n, args.n, 3)
        z_leaf_corners = find_leaves_containing_pts(eval_pts_z, t.root)

        #############################################################
        # Plot the solution

        plot_fp = os.path.join(args.plot_dir, f"soln_tol_{tol}.png")
        plot_diffs(
            u_reg_x, eval_pts_x, u_reg_y, eval_pts_y, u_reg_z, eval_pts_z, plot_fp
        )

        #############################################################
        # Save the resulting tree
        mesh_fp = os.path.join(args.plot_dir, f"tree_p_{args.p}_tol_{tol}.mat")
        out_dd = {
            "x_leaf_corners": x_leaf_corners,
            "y_leaf_corners": y_leaf_corners,
            "z_leaf_corners": z_leaf_corners,
        }
        savemat(mesh_fp, out_dd)

    #############################################################
    # Plot accuracy vs tol
    error_fp = os.path.join(args.plot_dir, "adaptive_error_vs_tol.png")
    plt.plot(args.tol, linf_error_vals, "-o", label="L_inf error")
    plt.plot(args.tol, l2_error_vals, "-o", label="L_2 error")

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

    #############################################################
    # Save data
    save_fp = os.path.join(args.plot_dir, f"adaptive_data_{nrm_str}_p_{args.p}.mat")

    out_dd = {
        "tol": args.tol,
        "linf_errors": linf_error_vals,
        "l2_errors": l2_error_vals,
        "n_leaves": n_leaves,
        "max_depths": max_depths,
        "mesh_times": mesh_times,
        "local_solve_times": local_solve_times,
        "build_times": build_times,
        "down_pass_times": down_pass_times,
        "D_size_vals": D_size_vals,
        "eval_pts_x": eval_pts_x,
        "eval_pts_y": eval_pts_y,
        "eval_pts_z": eval_pts_z,
    }
    savemat(save_fp, out_dd)


def main(args: argparse.Namespace) -> None:
    logging.info("Host device: %s", HOST_DEVICE)
    logging.info("Compute device: %s", DEVICE_ARR[0])
    ############################################################
    # Set up output directory
    args.plot_dir = os.path.join("data", "adaptive_meshing_3D", f"p_{args.p}")
    os.makedirs(args.plot_dir, exist_ok=True)
    ############################################################
    # Plot problem

    # plot_fp = os.path.join(args.plot_dir, "soln_and_source.png")
    # logging.info("Plotting problem to %s", plot_fp)
    # plot_problem(plot_fp)

    if args.hp_convergence:
        args.tol = None
        uniform_small_ex_for_jit()
        hp_convergence_study()
    else:
        # In this branch we want to save the graph structures produced
        # as well as solve statistics.
        adaptive_small_ex_for_jit()
        adaptive_convergence_study()

    logging.info("Finished")


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
