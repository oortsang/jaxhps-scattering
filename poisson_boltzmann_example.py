import os
import argparse
import logging
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import savemat

from hps.src.quadrature.quad_3D.adaptive_meshing import (
    generate_adaptive_mesh_level_restriction,
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
from hps.src.quadrature.trees import Node, get_all_leaves
from hps.src.quadrature.quad_3D.interpolation import (
    refinement_operator,
    interp_from_nonuniform_hps_to_uniform_grid,
)
from hps.src.plotting import plot_adaptive_grid_histogram
from hps.src.poisson_boltzmann_eqn_helpers import (
    permittivity,
    d_permittivity_d_x,
    d_permittivity_d_y,
    d_permittivity_d_z,
    rho,
    vdw_permittivity,
    d_vdw_permittivity_d_x,
    d_vdw_permittivity_d_y,
    d_vdw_permittivity_d_z,
)
from hps.src.logging_utils import FMT, TIMEFMT


# Suppress matplotlib debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--tol",
        type=float,
        nargs="+",
        default=[1e-01, 1e-02, 1e-03, 1e-04, 1e-05],
    )
    parser.add_argument("-p", type=int, default=8)
    parser.add_argument("-n", type=int, default=500)
    parser.add_argument("--vdW", default=False, action="store_true")

    return parser.parse_args()


XMIN = -1.0
XMAX = 1.0
YMIN = -1.0
YMAX = 1.0
ZMIN = -1.0
ZMAX = 1.0


def get_adaptive_mesh(tol: float, p: int, vdw: bool = False) -> Node:
    # Need to mesh the domain on these five functions:
    # 1. permittivity
    # 2. rho
    # 3. d_permittivity_d_x
    # 4. d_permittivity_d_y
    # 5. d_permittivity_d_z
    root = Node(
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        zmin=ZMIN,
        zmax=ZMAX,
        depth=0,
    )
    interp = refinement_operator(p)
    logging.debug(
        "get_adaptive_mesh: Generating adaptive mesh with tol %s and p=%i",
        tol,
        p,
    )

    if vdw:
        perm_fn = vdw_permittivity
        dx_fn = d_vdw_permittivity_d_x
        dy_fn = d_vdw_permittivity_d_y
        dz_fn = d_vdw_permittivity_d_z
        logging.debug("get_adaptive_mesh: Using vdw permittivity")
    else:
        perm_fn = permittivity
        dx_fn = d_permittivity_d_x
        dy_fn = d_permittivity_d_y
        dz_fn = d_permittivity_d_z
        logging.debug("get_adaptive_mesh: Using standard permittivity")

    # Mesh on permittivity
    generate_adaptive_mesh_level_restriction(
        root=root,
        refinement_op=interp,
        f_fn=perm_fn,
        tol=tol,
        p=p,
        q=p - 2,
        restrict_bool=True,
    )
    l = get_all_leaves(root)
    depths = [l.depth for l in l]
    logging.debug(
        "get_adaptive_mesh: Meshed on permittivity. # leaves: %s, max depth: %s",
        len(l),
        max(depths),
    )

    jax.clear_caches()
    # Mesh on d_permittivity_d_x
    generate_adaptive_mesh_level_restriction(
        root=root,
        refinement_op=interp,
        f_fn=dx_fn,
        tol=tol,
        p=p,
        q=p - 2,
        restrict_bool=True,
    )
    l = get_all_leaves(root)
    depths = [l.depth for l in l]
    logging.debug(
        "get_adaptive_mesh: Meshed on d_permittivity_d_x. # leaves: %s, max depth: %s",
        len(l),
        max(depths),
    )
    jax.clear_caches()

    # Mesh on d_permittivity_d_y
    generate_adaptive_mesh_level_restriction(
        root=root,
        refinement_op=interp,
        f_fn=dy_fn,
        tol=tol,
        p=p,
        q=p - 2,
        restrict_bool=True,
    )
    l = get_all_leaves(root)
    depths = [l.depth for l in l]
    logging.debug(
        "get_adaptive_mesh: Meshed on d_permittivity_d_y. # leaves: %s, max depth: %s",
        len(l),
        max(depths),
    )
    jax.clear_caches()

    # Mesh on d_permittivity_d_z
    generate_adaptive_mesh_level_restriction(
        root=root,
        refinement_op=interp,
        f_fn=dz_fn,
        tol=tol,
        p=p,
        q=p - 2,
        restrict_bool=True,
    )
    l = get_all_leaves(root)
    depths = [l.depth for l in l]
    logging.debug(
        "get_adaptive_mesh: Meshed on d_permittivity_d_z. # leaves: %s, max depth: %s",
        len(l),
        max(depths),
    )

    # Mesh on rho
    generate_adaptive_mesh_level_restriction(
        root=root,
        refinement_op=interp,
        f_fn=rho,
        tol=tol,
        p=p,
        q=p - 2,
        restrict_bool=True,
    )
    l = get_all_leaves(root)
    depths = [l.depth for l in l]
    logging.debug(
        "get_adaptive_mesh: Meshed on rho. # leaves: %s, max depth: %s",
        len(l),
        max(depths),
    )

    return root


def plot_solns(
    t: SolverObj,
    p: int,
    plot_fp: str,
) -> None:
    # Set up part of the grid.
    n = 100
    x = jnp.linspace(XMIN, XMAX, n)
    y = jnp.linspace(YMIN, YMAX, n)
    z = jnp.linspace(ZMIN, ZMAX, n)
    TITLESIZE = 20

    # Make a figure with 3 panels.
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    #############################################################
    # Do all of the interpolations first
    u_reg_x, _ = interp_from_nonuniform_hps_to_uniform_grid(
        root=t.root,
        p=p,
        f_evals=t.interior_solns,
        to_x=jnp.array([0.0]),
        to_y=jnp.flipud(y),
        to_z=z,
    )
    u_reg_x = u_reg_x.reshape(n, n)

    u_reg_y, pts = interp_from_nonuniform_hps_to_uniform_grid(
        root=t.root,
        p=p,
        f_evals=t.interior_solns,
        to_x=jnp.flipud(x),
        to_y=jnp.array([0.0]),
        to_z=z,
    )
    u_reg_y = u_reg_y.reshape(n, n)

    u_reg_z, pts = interp_from_nonuniform_hps_to_uniform_grid(
        root=t.root,
        p=p,
        f_evals=t.interior_solns,
        to_x=x,
        to_y=jnp.flipud(y),
        to_z=jnp.array([0.0]),
    )
    u_reg_z = u_reg_z.reshape(n, n)

    # Find max val for colorbar
    max_val = jnp.max(jnp.abs(jnp.array([u_reg_x, u_reg_y, u_reg_z])))

    #############################################################
    # First column: Plot along x = 0.0

    u_reg_x = u_reg_x.reshape(n, n)
    extent = [YMIN, YMAX, ZMIN, ZMAX]

    im_0 = ax[0].imshow(
        u_reg_x, cmap="plasma", extent=extent, vmin=0.0, vmax=max_val
    )
    plt.colorbar(im_0, ax=ax[0])
    ax[0].set_title("$x=0$ slice", fontsize=TITLESIZE)
    ax[0].set_xlabel("y", fontsize=TITLESIZE)
    ax[0].set_ylabel("z", fontsize=TITLESIZE)

    #############################################################
    # Second column: Plot along y = 0.0

    pts = pts.reshape(n, n, 3)
    extent = [XMIN, XMAX, ZMIN, ZMAX]

    im_0 = ax[1].imshow(
        u_reg_y, cmap="plasma", extent=extent, vmin=0.0, vmax=max_val
    )
    plt.colorbar(im_0, ax=ax[1])
    ax[1].set_title("$y=0$ slice", fontsize=TITLESIZE)
    ax[1].set_xlabel("x", fontsize=TITLESIZE)
    ax[1].set_ylabel("z", fontsize=TITLESIZE)

    #############################################################
    # Third column: Plot along z = 0.0

    pts = pts.reshape(n, n, 3)
    extent = [XMIN, XMAX, YMIN, YMAX]

    im_0 = ax[2].imshow(
        u_reg_z, cmap="plasma", extent=extent, vmin=0.0, vmax=max_val
    )
    plt.colorbar(im_0, ax=ax[2])
    ax[2].set_title("$z=0$ slice", fontsize=TITLESIZE)
    ax[2].set_xlabel("x", fontsize=TITLESIZE)
    ax[2].set_ylabel("y", fontsize=TITLESIZE)

    fig.tight_layout()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    ############################################################
    # Set up output directory
    vdw_str = "_vdw" if args.vdW else ""
    args.plot_dir = os.path.join(
        "data", f"poisson_boltzmann_3D{vdw_str}", f"p_{args.p}"
    )
    os.makedirs(args.plot_dir, exist_ok=True)
    logging.info("Saving to directory: %s", args.plot_dir)

    ############################################################
    # Set the perm and grad perm functions
    if args.vdW:
        perm_fn = vdw_permittivity
        perm_dx_fn = d_vdw_permittivity_d_x
        perm_dy_fn = d_vdw_permittivity_d_y
        perm_dz_fn = d_vdw_permittivity_d_z
    else:
        perm_fn = permittivity
        perm_dx_fn = d_permittivity_d_x
        perm_dy_fn = d_permittivity_d_y
        perm_dz_fn = d_permittivity_d_z
    ############################################################
    # Do one example to JIT compile code
    logging.info("Doing a small example to JIT compile code parts")
    root = get_adaptive_mesh(0.5, args.p, vdw=args.vdW)
    t = create_solver_obj_3D(p=args.p, q=args.p - 2, root=root)
    source_evals = -1 * rho(t.leaf_cheby_points)
    perm_evals = perm_fn(t.leaf_cheby_points)
    D_x_coeffs = perm_dx_fn(t.leaf_cheby_points)
    D_y_coeffs = perm_dy_fn(t.leaf_cheby_points)
    D_z_coeffs = perm_dz_fn(t.leaf_cheby_points)
    local_solve_stage(
        t,
        source_term=source_evals,
        D_xx_coeffs=perm_evals,
        D_yy_coeffs=perm_evals,
        D_zz_coeffs=perm_evals,
        D_x_coeffs=D_x_coeffs,
        D_y_coeffs=D_y_coeffs,
        D_z_coeffs=D_z_coeffs,
    )
    build_stage(t)
    bdry_data = get_bdry_data_evals_lst_3D(
        t, f=lambda x: jnp.zeros_like(x[..., 0])
    )
    down_pass(t, bdry_data)

    #############################################################
    # Do adaptive refinement by looping over tol values
    n_tol_vals = len(args.tol)
    error_vals = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    local_solve_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64) * jnp.nan
    build_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64) * jnp.nan
    down_pass_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64) * jnp.nan
    mesh_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64) * jnp.nan
    n_leaves = jnp.zeros((n_tol_vals,))
    max_depths = jnp.zeros((n_tol_vals,))
    u_reg_x_vals = jnp.zeros((n_tol_vals, args.n, args.n), dtype=jnp.float64)
    u_reg_y_vals = jnp.zeros((n_tol_vals, args.n, args.n), dtype=jnp.float64)
    u_reg_z_vals = jnp.zeros((n_tol_vals, args.n, args.n), dtype=jnp.float64)

    for i, tol in enumerate(args.tol):
        t0 = default_timer()
        root = get_adaptive_mesh(tol, args.p, vdw=args.vdW)
        mesh_time = default_timer() - t0
        mesh_times = mesh_times.at[i].set(mesh_time)
        logging.info(
            "Generated adaptive mesh with L_inf error tolerance %s", tol
        )

        ll = get_all_leaves(root)
        depths = [l.depth for l in ll]

        logging.info(
            "Adaptive mesh number of leaves: %i and max depth: %i, meshing time: %s",
            len(ll),
            max(depths),
            mesh_time,
        )

        # Save number of leaves and max depth
        n_leaves = n_leaves.at[i].set(len(ll))
        max_depths = max_depths.at[i].set(max(depths))

        t = create_solver_obj_3D(p=args.p, q=args.p - 2, root=root)

        # Plot histogram of the side lengths
        plot_fp = os.path.join(args.plot_dir, f"histogram_tol_{tol}.png")
        plot_adaptive_grid_histogram(root, plot_fp, tol, args.p)
        #############################################################
        # Do a PDE Solve
        try:
            source_evals = -1 * rho(t.leaf_cheby_points)
            perm_evals = perm_fn(t.leaf_cheby_points)
            D_x_coeffs = perm_dx_fn(t.leaf_cheby_points)
            D_y_coeffs = perm_dy_fn(t.leaf_cheby_points)
            D_z_coeffs = perm_dz_fn(t.leaf_cheby_points)

            t0 = default_timer()
            local_solve_stage(
                t,
                source_term=source_evals,
                D_xx_coeffs=perm_evals,
                D_yy_coeffs=perm_evals,
                D_zz_coeffs=perm_evals,
                D_x_coeffs=D_x_coeffs,
                D_y_coeffs=D_y_coeffs,
                D_z_coeffs=D_z_coeffs,
            )
            t_local_solve = default_timer() - t0
            logging.info("Local solve stage in %f sec", t_local_solve)
            local_solve_times = local_solve_times.at[i].set(t_local_solve)

            t_0 = default_timer()
            build_stage(t)
            t_build = default_timer() - t_0
            logging.info("Build stage in %f sec", t_build)
            build_times = build_times.at[i].set(t_build)

            bdry_data = get_bdry_data_evals_lst_3D(
                t, f=lambda x: jnp.zeros_like(x[..., 0])
            )
            t_0 = default_timer()
            down_pass(t, bdry_data)
            t_down = default_timer() - t_0
            logging.info("Down pass in %f sec", t_down)
            down_pass_times = down_pass_times.at[i].set(t_down)

            #############################################################
            # Interploate the soln to the x,y,z = 0.0 planes
            x = jnp.linspace(XMIN, XMAX, args.n)
            y = jnp.linspace(YMIN, YMAX, args.n)
            z = jnp.linspace(ZMIN, ZMAX, args.n)

            #############################################################
            # Do all of the interpolations first
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

            # Save these objects
            u_reg_x_vals = u_reg_x_vals.at[i, :, :].set(u_reg_x)
            u_reg_y_vals = u_reg_y_vals.at[i, :, :].set(u_reg_y)
            u_reg_z_vals = u_reg_z_vals.at[i, :, :].set(u_reg_z)

            #############################################################
            # Plot the solution

            plot_fp = os.path.join(args.plot_dir, f"soln_tol_{tol}.png")
            plot_solns(t, args.p, plot_fp)
        except:  # noqa: E722
            # If we run out of memory, still want to save the grid sizes
            logging.info("Could not compute for tol=%s", tol)

        #############################################################
        # Pickle the resulting tree
        mesh_fp = os.path.join(args.plot_dir, f"tree_p_{args.p}_tol_{tol}.mat")
        out_dd = {
            "x_leaf_corners": x_leaf_corners,
            "y_leaf_corners": y_leaf_corners,
            "z_leaf_corners": z_leaf_corners,
        }
        savemat(mesh_fp, out_dd)

    #############################################################
    # Save data
    save_fp = os.path.join(args.plot_dir, "data.mat")

    out_dd = {
        "tol": args.tol,
        "errors": error_vals,
        "n_leaves": n_leaves,
        "max_depths": max_depths,
        "mesh_times": mesh_times,
        "local_solve_times": local_solve_times,
        "build_times": build_times,
        "down_pass_times": down_pass_times,
        "u_reg_x_vals": u_reg_x_vals,
        "u_reg_y_vals": u_reg_y_vals,
        "u_reg_z_vals": u_reg_z_vals,
        "eval_pts_x": eval_pts_x,
        "eval_pts_y": eval_pts_y,
        "eval_pts_z": eval_pts_z,
    }
    savemat(save_fp, out_dd)

    logging.info("Finished")


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
