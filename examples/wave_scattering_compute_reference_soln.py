import argparse
import os
import logging

import jax.numpy as jnp
import jax
from scipy.io import savemat

from wave_scattering_utils import (
    solve_scattering_problem,
    load_SD_matrices,
    get_uin,
)
from scattering_potentials import (
    q_gaussian_bumps,
    q_GBM_1,
    q_horizontally_graded,
    q_luneburg,
    q_vertically_graded,
)
from plotting_utils import plot_field_for_wave_scattering_experiment


# Silence matplotlib debug messages
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True

# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_default_device", jax.devices("gpu")[0])


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reference solution for wave scattering problem."
    )

    parser.add_argument(
        "-l",
        type=int,
        default=5,
        help="Number of levels in the quadtree.",
    )
    parser.add_argument(
        "-p",
        type=int,
        default=16,
        help="Chebyshev polynomial order.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=500,
        help="Number of points per dimension for the output regular grid.",
    )
    parser.add_argument("-k", type=float, default=100.0, help="Wavenumber.")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--SD_matrix_prefix",
        default="data/examples/SD_matrices",
    )
    # Specify either "luneburg", "vertically_graded", or "gauss_bumps"
    parser.add_argument(
        "--scattering_potential",
        default="gauss_bumps",
        help="Scattering potential to use.",
        choices=[
            "luneburg",
            "vertically_graded",
            "horizontally_graded",
            "gauss_bumps",
            "GBM_1",
        ],
    )
    parser.add_argument("--plot_utot", default=False, action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Does the following:

    1. Solves the wave scattering problem on a quadtree with specified parameters:
        - l: number of levels in the quadtree
        - p: Chebyshev polynomial order
        - k: Incident wave frequency
    2. Interpolates the solution onto a regular grid with specified number of points:
        - n: number of points per dimension for the output regular grid
    3. Saves the interpolated solution to a file.
        - The file is saved in the specified output directory.
    4. Plots the interpolated solution.
    """
    # Check the scattering potential argument
    if args.scattering_potential == "luneburg":
        args.output_dir = (
            f"data/examples/wave_scattering/luneburg_k_{int(args.k)}"
        )
        q_fn_handle = q_luneburg
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "vertically_graded":
        args.output_dir = (
            f"data/examples/wave_scattering/vertically_graded_k_{int(args.k)}"
        )
        q_fn_handle = q_vertically_graded
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "horizontally_graded":
        args.output_dir = f"data/examples/wave_scattering/horizontally_graded_k_{int(args.k)}"
        q_fn_handle = q_horizontally_graded
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "gauss_bumps":
        args.output_dir = (
            f"data/examples/wave_scattering/gauss_bumps_k_{int(args.k)}"
        )
        q_fn_handle = q_gaussian_bumps
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "GBM_1":
        args.output_dir = (
            f"data/examples/wave_scattering/GBM_1_k_{int(args.k)}"
        )
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
        q_fn_handle = q_GBM_1
    else:
        raise ValueError("Invalid scattering potential")

    output_dir = args.output_dir
    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Outputs will be written to %s", output_dir)

    # Steps 1 and 2 are combined in the solve_scattering_problem function.
    q = args.p - 2
    nside = 2**args.l
    k_str = str(int(args.k))
    S_D_matrices_fp = os.path.join(
        args.SD_matrix_prefix, f"SD_k{k_str}_n{q}_nside{nside}_dom1.mat"
    )

    domain_bounds = jnp.array([xmin, xmax, ymin, ymax])
    source_dirs = jnp.array([0.0])

    wave_freq = args.k

    logging.debug("Loading S and D from disk...")
    S, D = load_SD_matrices(S_D_matrices_fp)

    uscat, target_pts, solve_time = solve_scattering_problem(
        l=args.l,
        p=args.p,
        n=args.n,
        k=wave_freq,
        q_fn=q_fn_handle,
        domain_bounds=domain_bounds,
        source_dirs=source_dirs,
        S=S,
        D=D,
    )
    # The returned uscat array has a n_src dimension that we want to remove.
    uscat = uscat[..., 0]
    logging.info("Computed reference solution in %f seconds", solve_time)

    # Step 3: Save the interpolated solution to a file.
    fp_out = os.path.join(output_dir, "reference_solution.mat")
    savemat(
        fp_out,
        {
            "uscat": uscat,
            "target_pts": target_pts,
            "q": q_fn_handle(target_pts),
        },
    )

    # Step 4: Plot
    logging.info("Plotting results...")
    # plot q
    q_vals = q_fn_handle(target_pts)
    plot_field_for_wave_scattering_experiment(
        q_vals,
        target_pts,
        # title="q",
        save_fp=os.path.join(output_dir, "q.pdf"),
    )

    # plot real part of uscat
    plot_field_for_wave_scattering_experiment(
        jnp.real(uscat),
        target_pts,
        use_bwr_cmap=True,
        # title="uscat real",
        save_fp=os.path.join(output_dir, "uscat_real.pdf"),
    )

    # plot the imaginary part of uscat
    plot_field_for_wave_scattering_experiment(
        jnp.imag(uscat),
        target_pts,
        use_bwr_cmap=True,
        # title="uscat imag",
        save_fp=os.path.join(output_dir, "uscat_imag.pdf"),
    )

    # plto the absolute value of uscat
    plot_field_for_wave_scattering_experiment(
        jnp.abs(uscat),
        target_pts,
        use_bwr_cmap=False,
        cmap_str="hot",
        # title="uscat abs",
        save_fp=os.path.join(output_dir, "uscat_abs.pdf"),
    )

    # Plot utot if specified
    if args.plot_utot:
        logging.info("plotting utot...")
        uin = get_uin(args.k, target_pts, source_dirs)
        uin = jax.device_put(uin, jax.devices("cpu")[0])
        utot = uin[..., 0] + uscat

        # plot real part of utot
        plot_field_for_wave_scattering_experiment(
            jnp.real(utot),
            target_pts,
            use_bwr_cmap=True,
            # title="utot real",
            save_fp=os.path.join(output_dir, "utot_real.pdf"),
        )
        # plot abs of utot
        plot_field_for_wave_scattering_experiment(
            jnp.abs(utot),
            target_pts,
            use_bwr_cmap=False,
            cmap_str="hot",
            # title="utot abs",
            save_fp=os.path.join(output_dir, "utot_abs.pdf"),
        )

    logging.info("Finished")


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s:jaxhps: %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    main(args)
