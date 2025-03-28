import os
import argparse
import logging
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.io import savemat


from hahps import (
    DiscretizationNode3D,
    Domain,
    PDEProblem,
    build_solver,
    solve,
    HOST_DEVICE,
    DEVICE_ARR,
    get_all_leaves,
)
from wavefront_data import (
    wavefront_soln,
    d_xx_wavefront_soln,
    d_yy_wavefront_soln,
    d_zz_wavefront_soln,
    default_lap_coeffs,
)

# Suppress matplotlib debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--tol", type=float, nargs="+", default=[1e-02, 1e-04, 1e-06, 1e-08]
    )
    parser.add_argument("-p", type=int, default=8)
    parser.add_argument("--hp_convergence", action="store_true")
    parser.add_argument("--max_l", type=int, default=4)
    parser.add_argument(
        "-n",
        type=int,
        default=500,
        help="Number of pixels to use when plotting",
    )

    return parser.parse_args()


XMIN = 0.0
XMAX = 1.0
YMIN = 0.0
YMAX = 1.0
ZMIN = 0.0
ZMAX = 1.0


def plot_diffs(
    u_computed: jnp.array,
    root: DiscretizationNode3D,
    eval_pts: jnp.array,
    plot_fp: str,
) -> None:
    TITLESIZE = 20

    # Make a figure with 3 panels. First col is computed u, second col is expected u, third col is the absolute error
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    #############################################################
    # First column: Computed u

    u_expected = wavefront_soln(eval_pts)
    extent = [YMIN, YMAX, ZMIN, ZMAX]

    u_computed = u_computed.squeeze(2)
    u_expected = u_expected.squeeze(2)

    logging.debug(
        "plot_diffs: u_computed shape = %s, u_expected shape = %s",
        u_computed.shape,
        u_expected.shape,
    )

    im_0 = ax[0].imshow(u_computed, cmap="plasma", extent=extent)
    plt.colorbar(im_0, ax=ax[0])
    ax[0].set_title("Computed $u$", fontsize=TITLESIZE)
    ax[0].set_xlabel("$x_1$", fontsize=TITLESIZE)
    ax[0].set_ylabel("$x_2$", fontsize=TITLESIZE)

    #############################################################
    # Find all nodes that intersect z=0 and plot them.
    leaves = get_all_leaves(root)
    leaves_intersect_zero = [l for l in leaves if l.zmin <= 0 and l.zmax >= 0]
    logging.debug(
        "plot_diffs: Found %s leaves intersecting z=0",
        len(leaves_intersect_zero),
    )

    for l in leaves_intersect_zero:
        x = [l.xmin, l.xmax, l.xmax, l.xmin, l.xmin]
        y = [l.ymin, l.ymin, l.ymax, l.ymax, l.ymin]
        ax[0].plot(x, y, "-", color="gray", linewidth=1)

    # im_1 = ax[1].imshow(u_expected, cmap="plasma", extent=extent)
    # plt.colorbar(im_1, ax=ax[1])
    # ax[1].set_title("Expected $u$", fontsize=TITLESIZE)
    # ax[1].set_xlabel("x", fontsize=TITLESIZE)
    # ax[1].set_ylabel("y", fontsize=TITLESIZE)

    im_2 = ax[1].imshow(
        np.abs(u_computed - u_expected), cmap="hot", extent=extent
    )
    plt.colorbar(im_2, ax=ax[1])
    ax[1].set_title("Absolute Error", fontsize=TITLESIZE)
    ax[1].set_xlabel("$x_1$", fontsize=TITLESIZE)
    ax[1].set_ylabel("$x_2$", fontsize=TITLESIZE)

    fig.tight_layout()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()
    plt.close(fig)


def source(x: jnp.array) -> jnp.array:
    lap_u = (
        d_xx_wavefront_soln(x)
        + d_yy_wavefront_soln(x)
        + d_zz_wavefront_soln(x)
    )
    return lap_u


def hp_convergence_study() -> None:
    #############################################################
    # Do adaptive refinement by looping over tol values
    linf_error_vals = jnp.zeros((args.max_l,), dtype=jnp.float64)
    build_times = jnp.zeros((args.max_l,), dtype=jnp.float64)
    down_pass_times = jnp.zeros((args.max_l,), dtype=jnp.float64)

    l_vals = jnp.arange(1, args.max_l + 1)

    for i, l in enumerate(l_vals):
        logging.info("Uniform refinement level %s", l)
        root = DiscretizationNode3D(
            xmin=XMIN,
            xmax=XMAX,
            ymin=YMIN,
            ymax=YMAX,
            zmin=ZMIN,
            zmax=ZMAX,
        )
        domain = Domain(p=args.p, q=args.p - 2, root=root, L=l)

        #############################################################
        # Do a PDE Solve
        source_evals = source(domain.interior_points)
        D_xx_evals = default_lap_coeffs(domain.interior_points)
        D_yy_evals = default_lap_coeffs(domain.interior_points)
        D_zz_evals = default_lap_coeffs(domain.interior_points)

        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=D_xx_evals,
            D_yy_coefficients=D_yy_evals,
            D_zz_coefficients=D_zz_evals,
            source=source_evals,
        )
        t0 = default_timer()

        # Build the solver
        build_solver(pde_problem)

        t_build = default_timer() - t0
        logging.info("Local solves + Merge stage in %f sec", t_build)
        build_times = build_times.at[i].set(t_build)
        bdry_data = domain.get_adaptive_boundary_data_lst(f=wavefront_soln)

        t_0 = default_timer()
        computed_soln = solve(pde_problem, bdry_data)
        t_down = default_timer() - t_0
        logging.info("Down pass in %f sec", t_down)
        down_pass_times = down_pass_times.at[i].set(t_down)

        #############################################################
        # Compute the error
        expected_soln = wavefront_soln(domain.interior_points)

        linf_error = jnp.max(jnp.abs(expected_soln - computed_soln)) / jnp.max(
            jnp.abs(expected_soln)
        )
        logging.info("Relative L_inf error: %s", linf_error)
        linf_error_vals = linf_error_vals.at[i].set(linf_error)

        #############################################################
        # Interpolate to a uniform grid

        x = jnp.linspace(XMIN, XMAX, args.n)
        y = jnp.linspace(YMIN, YMAX, args.n)

        u_reg = domain.interp_from_interior_points(
            computed_soln,
            eval_points_x=x,
            eval_points_y=y,
            eval_points_z=jnp.array([0.0]),
        )

        # Construct the eval pts
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        pts = jnp.stack([X, Y, jnp.zeros_like(X)], axis=-1)  # Shape (n,n,3)

        #############################################################
        # Plot the solution

        plot_fp = os.path.join(args.plot_dir, f"hp_soln_l_{l}_p_{args.p}.pdf")
        plot_diffs(
            u_computed=u_reg,
            eval_pts=pts,
            plot_fp=plot_fp,
        )


def adaptive_small_ex_for_jit() -> None:
    logging.info("Doing a small example to JIT compile the code")
    root = DiscretizationNode3D(
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        zmin=ZMIN,
        zmax=ZMAX,
    )

    domain = Domain.from_adaptive_discretization(
        p=args.p, q=args.p - 2, root=root, f=source, tol=0.1
    )

    #############################################################
    # Do a PDE Solve
    source_evals = source(domain.interior_points)
    D_xx_evals = default_lap_coeffs(domain.interior_points)
    D_yy_evals = default_lap_coeffs(domain.interior_points)
    D_zz_evals = default_lap_coeffs(domain.interior_points)

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=D_xx_evals,
        D_yy_coefficients=D_yy_evals,
        D_zz_coefficients=D_zz_evals,
        source=source_evals,
    )
    build_solver(pde_problem)

    bdry_data = domain.get_adaptive_boundary_data_lst(f=wavefront_soln)
    _ = solve(pde_problem, bdry_data)


def uniform_small_ex_for_jit() -> None:
    logging.info("Doing a small to JIT compile code")
    root = DiscretizationNode3D(
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        zmin=ZMIN,
        zmax=ZMAX,
    )

    domain = Domain(p=args.p, q=args.p - 2, root=root, L=1)

    #############################################################
    # Do a PDE Solve
    source_evals = source(domain.interior_points)
    D_xx_evals = default_lap_coeffs(domain.interior_points)
    D_yy_evals = default_lap_coeffs(domain.interior_points)
    D_zz_evals = default_lap_coeffs(domain.interior_points)

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=D_xx_evals,
        D_yy_coefficients=D_yy_evals,
        D_zz_coefficients=D_zz_evals,
        source=source_evals,
    )
    build_solver(pde_problem)

    bdry_data = domain.get_adaptive_boundary_data_lst(f=wavefront_soln)
    _ = solve(pde_problem, bdry_data)


def adaptive_convergence_study() -> None:
    #############################################################
    # Do adaptive refinement by looping over tol values
    n_tol_vals = len(args.tol)
    linf_error_vals = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    mesh_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    build_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    down_pass_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    n_leaves = jnp.zeros((n_tol_vals,))
    max_depths = jnp.zeros((n_tol_vals,))

    nrm_str = "linf"

    for i, tol in enumerate(args.tol):
        root = DiscretizationNode3D(
            xmin=XMIN,
            xmax=XMAX,
            ymin=YMIN,
            ymax=YMAX,
            zmin=ZMIN,
            zmax=ZMAX,
        )
        t0 = default_timer()
        domain = Domain.from_adaptive_discretization(
            p=args.p,
            q=args.p - 2,
            root=root,
            f=source,
            tol=tol,
        )

        mesh_time = default_timer() - t0
        mesh_times = mesh_times.at[i].set(mesh_time)
        ll = get_all_leaves(root)
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
        plot_fp = os.path.join(
            args.plot_dir, f"adaptive_mesh_hist_tol_{tol}.pdf"
        )

        #############################################################
        # Do a PDE Solve
        source_evals = source(domain.interior_points)
        D_xx_evals = default_lap_coeffs(domain.interior_points)
        D_yy_evals = default_lap_coeffs(domain.interior_points)
        D_zz_evals = default_lap_coeffs(domain.interior_points)

        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=D_xx_evals,
            D_yy_coefficients=D_yy_evals,
            D_zz_coefficients=D_zz_evals,
            source=source_evals,
        )

        t0 = default_timer()

        # Build the solver
        build_solver(pde_problem)

        t_build = default_timer() - t0
        logging.info("Local solve + merge stage in %f sec", t_build)

        build_times = build_times.at[i].set(t_build)
        bdry_data = domain.get_adaptive_boundary_data_lst(f=wavefront_soln)

        t_0 = default_timer()
        computed_soln = solve(pde_problem=pde_problem, boundary_data=bdry_data)
        t_down = default_timer() - t_0
        logging.info("Down pass in %f sec", t_down)
        down_pass_times = down_pass_times.at[i].set(t_down)

        logging.debug(
            "adaptive_convergence_study: computed_soln shape: %s",
            computed_soln.shape,
        )

        #############################################################
        # Compute the error
        expected_soln = wavefront_soln(domain.interior_points)

        linf_error = jnp.max(jnp.abs(expected_soln - computed_soln)) / jnp.max(
            jnp.abs(expected_soln)
        )
        logging.info("Relative L_inf error: %s", linf_error)
        linf_error_vals = linf_error_vals.at[i].set(linf_error)

        #############################################################
        # Interpolate to a uniform grid

        x = jnp.linspace(XMIN, XMAX, args.n)
        y = jnp.flipud(jnp.linspace(YMIN, YMAX, args.n))

        u_reg, pts = domain.interp_from_interior_points(
            samples=computed_soln,
            eval_points_x=x,
            eval_points_y=y,
            eval_points_z=jnp.array([0.0]),
        )
        # # Construct the eval pts
        # X, Y = jnp.meshgrid(x, y, indexing="ij")
        # pts = jnp.stack([X, Y, jnp.zeros_like(X)], axis=-1)  # Shape (n,n,3)

        logging.debug(
            "adaptive_convergence_study: u_reg shape: %s", u_reg.shape
        )
        logging.debug("adaptive_convergence_study: pts shape: %s", pts.shape)

        #############################################################
        # Plot the solution

        plot_fp = os.path.join(args.plot_dir, f"soln_tol_{tol}.pdf")
        plot_diffs(u_reg, domain.root, pts, plot_fp)

    #############################################################
    # Plot accuracy vs tol
    error_fp = os.path.join(args.plot_dir, "adaptive_error_vs_tol.pdf")
    plt.plot(args.tol, linf_error_vals, "-o", label="L_inf error")

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
    save_fp = os.path.join(
        args.plot_dir, f"adaptive_data_{nrm_str}_p_{args.p}.mat"
    )

    out_dd = {
        "tol": args.tol,
        "linf_errors": linf_error_vals,
        "n_leaves": n_leaves,
        "max_depths": max_depths,
        "mesh_times": mesh_times,
        "build_times": build_times,
        "down_pass_times": down_pass_times,
    }
    savemat(save_fp, out_dd)


def main(args: argparse.Namespace) -> None:
    logging.info("Host device: %s", HOST_DEVICE)
    logging.info("Compute device: %s", DEVICE_ARR[0])
    ############################################################
    # Set up output directory
    args.plot_dir = os.path.join(
        "data", "examples", "adaptive_meshing_3D", f"p_{args.p}"
    )
    os.makedirs(args.plot_dir, exist_ok=True)
    logging.info("Saving outputs to %s", args.plot_dir)

    ############################################################
    # Run meshing
    if args.hp_convergence:
        args.tol = None
        uniform_small_ex_for_jit()
        hp_convergence_study()
    else:
        adaptive_small_ex_for_jit()
        adaptive_convergence_study()

    logging.info("Finished")


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s:ha-hps: %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    main(args)
