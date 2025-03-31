import os
import argparse
import logging
import jax.numpy as jnp
from timeit import default_timer
from scipy.io import savemat

from hahps import (
    Domain,
    DiscretizationNode3D,
    build_solver,
    solve,
    PDEProblem,
    get_all_leaves,
)
from poisson_boltzmann_utils import (
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
from plotting_utils import plot_func_with_grid

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


def get_adaptive_domain(tol: float, p: int, vdw: bool = False) -> Domain:
    # Need to mesh the domain on these five functions:
    # 1. permittivity
    # 2. rho
    # 3. d_permittivity_d_x
    # 4. d_permittivity_d_y
    # 5. d_permittivity_d_z
    root = DiscretizationNode3D(
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        zmin=ZMIN,
        zmax=ZMAX,
        depth=0,
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

    f_lst = [
        perm_fn,
        dx_fn,
        dy_fn,
        dz_fn,
        rho,
    ]

    domain = Domain.from_adaptive_discretization(
        p=p, q=p - 2, root=root, f=f_lst, tol=tol
    )
    l = get_all_leaves(domain.root)
    depths = [l.depth for l in l]
    logging.debug(
        "get_adaptive_mesh: Meshed on rho. # leaves: %s, max depth: %s",
        len(l),
        max(depths),
    )

    return domain


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
    domain = get_adaptive_domain(0.5, args.p, vdw=args.vdW)
    source_evals = -1 * rho(domain.interior_points)
    perm_evals = perm_fn(domain.interior_points)
    D_x_coeffs = perm_dx_fn(domain.interior_points)
    D_y_coeffs = perm_dy_fn(domain.interior_points)
    D_z_coeffs = perm_dz_fn(domain.interior_points)

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=perm_evals,
        D_yy_coefficients=perm_evals,
        D_zz_coefficients=perm_evals,
        D_x_coefficients=D_x_coeffs,
        D_y_coefficients=D_y_coeffs,
        D_z_coefficients=D_z_coeffs,
        source_term=source_evals,
    )

    # Solve the problem
    build_solver(pde_problem)

    # Zero boundary condition
    bdry_data = domain.get_adaptive_boundary_data_lst(
        f=lambda x: jnp.zeros_like(x[..., 0])
    )

    solve(pde_problem, bdry_data)

    #############################################################
    # Do adaptive refinement by looping over tol values
    n_tol_vals = len(args.tol)
    error_vals = jnp.zeros((n_tol_vals,), dtype=jnp.float64)
    build_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64) * jnp.nan
    down_pass_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64) * jnp.nan
    mesh_times = jnp.zeros((n_tol_vals,), dtype=jnp.float64) * jnp.nan
    n_leaves = jnp.zeros((n_tol_vals,))
    max_depths = jnp.zeros((n_tol_vals,))

    for i, tol in enumerate(args.tol):
        t0 = default_timer()
        domain = get_adaptive_domain(tol, args.p, vdw=args.vdW)
        mesh_time = default_timer() - t0
        mesh_times = mesh_times.at[i].set(mesh_time)
        logging.info(
            "Generated adaptive mesh with L_inf error tolerance %s", tol
        )

        ll = get_all_leaves(domain)
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

        #############################################################
        # Do a PDE Solve
        try:
            source_evals = -1 * rho(domain.interior_points)
            perm_evals = perm_fn(domain.interior_points)
            D_x_coeffs = perm_dx_fn(domain.interior_points)
            D_y_coeffs = perm_dy_fn(domain.interior_points)
            D_z_coeffs = perm_dz_fn(domain.interior_points)

            pde_problem = PDEProblem(
                domain=domain,
                D_xx_coefficients=perm_evals,
                D_yy_coefficients=perm_evals,
                D_zz_coefficients=perm_evals,
                D_x_coefficients=D_x_coeffs,
                D_y_coefficients=D_y_coeffs,
                D_z_coefficients=D_z_coeffs,
                source_term=source_evals,
            )

            t0 = default_timer()

            build_solver(pde_problem)
            t_build = default_timer() - t0
            logging.info("Local solve + merge stage in %f sec", t_build)
            build_times = build_times.at[i].set(t_build)

            # Zero boundary condition
            bdry_data = domain.get_adaptive_boundary_data_lst(
                f=lambda x: jnp.zeros_like(x[..., 0])
            )
            t_0 = default_timer()
            solve(pde_problem, bdry_data)
            t_down = default_timer() - t_0
            logging.info("Down pass in %f sec", t_down)
            down_pass_times = down_pass_times.at[i].set(t_down)

        except Exception as e:
            logging.error("Error solving PDE: %s", e)

        #############################################################
        # Want to plot the permittivity along the z=0 plane
        x = jnp.linspace(XMIN, XMAX, args.n)
        y = jnp.flipud(jnp.linspace(YMIN, YMAX, args.n))
        z = jnp.array([0.0])

        perm_evals_hps = perm_fn(domain.interior_points)
        perm_evals_regular, pts = domain.interp_from_interior_points(
            perm_evals_hps, x, y, z
        )

        perm_evals_regular = perm_evals_regular.squeeze()
        pts = pts[:, :, 0]

        leaves = get_all_leaves(domain.root)
        leaves_intersect_zero = [
            l for l in leaves if l.zmin <= 0 and l.zmax >= 0
        ]

        plot_func_with_grid(perm_evals_regular, pts, leaves_intersect_zero)

    #############################################################
    # Save data
    save_fp = os.path.join(args.plot_dir, "data.mat")

    out_dd = {
        "tol": args.tol,
        "errors": error_vals,
        "n_leaves": n_leaves,
        "max_depths": max_depths,
        "mesh_times": mesh_times,
        "build_times": build_times,
        "down_pass_times": down_pass_times,
    }
    savemat(save_fp, out_dd)

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
