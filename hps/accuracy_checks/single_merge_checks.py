import logging
from typing import Callable
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from hps.accuracy_checks.dirichlet_neumann_data import (
    TEST_CASE_POISSON_POLY,
    TEST_CASE_POISSON_NONPOLY,
    TEST_CASE_NONCONSTANT_COEFF_POLY,
    TEST_CASE_NONPOLY_ITI,
    TEST_CASE_HELMHOLTZ_1,
    ETA,
)
from hps.accuracy_checks.utils import (
    setup_tree_quad_merge,
)
from hps.src.up_down_passes import build_stage, down_pass, local_solve_stage
from hps.src.solver_obj import (
    get_bdry_data_evals_lst_2D,
)


def check_l_inf_error_convergence_quad_merge(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
    check_dtn: bool = False,
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    visualize: bool = False,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^2
    with increasing levels of Chebyshev discretization.

    It computes the solution by solving on 4 separate nodes, and merging DtN maps.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """
    # south = -jnp.pi / 2
    # north = jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2

    n_X = 100
    x = jnp.linspace(west, east, n_X)
    X, Y = jnp.meshgrid(x, x)

    error_vals = jnp.ones_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        t = setup_tree_quad_merge(p)
        # print(
        #     "check_l_inf_error_convergence_EW_merge: t.start_tree_NS", t.start_tree_NS
        # )

        # Assemble differential operator coeffs for the Laplacian
        d_xx_coeffs = d_xx_coeff_fn(t.leaf_cheby_points)
        d_yy_coeffs = d_yy_coeff_fn(t.leaf_cheby_points)
        source = source_fn(t.leaf_cheby_points)
        # logging.debug(
        #     "check_l_inf_error_convergence_EW_merge: t.leaf_cheby_points = %s",
        #     t.leaf_cheby_points.shape,
        # )
        # logging.debug(
        #     "check_l_inf_error_convergence_EW_merge: d_xx_coeffs = %s",
        #     d_xx_coeffs.shape,
        # )
        # logging.debug(
        #     "check_l_inf_error_convergence_EW_merge: d_yy_coeffs = %s",
        #     d_yy_coeffs.shape,
        # )
        # Local solve stage

        local_solve_stage(
            t, source_term=source, D_xx_coeffs=d_xx_coeffs, D_yy_coeffs=d_yy_coeffs
        )

        build_stage(t)

        root_bdry_points = t.root_boundary_points
        n_per_side = root_bdry_points.shape[0] // 4
        boundary_data_lst = get_bdry_data_evals_lst_2D(t, dirichlet_data_fn)
        # boundary_data = dirichlet_data_fn(root_bdry_points)

        if check_dtn:
            # Evaluate the accuracy of the DtN map
            expected_soln = jnp.concatenate(
                [
                    -1 * dudy_fn(root_bdry_points[:n_per_side]),
                    dudx_fn(root_bdry_points[n_per_side : 2 * n_per_side]),
                    dudy_fn(root_bdry_points[2 * n_per_side : 3 * n_per_side]),
                    -1 * dudx_fn(root_bdry_points[3 * n_per_side :]),
                ]
            )
            map_DtN = t.interior_node_DtN_maps[-1]
            # print(
            #     "check_l_inf_error_convergence_EW_merge: map_DtN shape", map_DtN.shape
            # )
            # print(
            #     "check_l_inf_error_convergence_EW_merge: boundary_data shape",
            #     boundary_data.shape,
            # )
            computed_soln_values = map_DtN @ jnp.concatenate(boundary_data_lst)

        else:

            down_pass(t, boundary_data_lst)

            all_cheby_points = jnp.reshape(t.leaf_cheby_points, (-1, 2))
            computed_soln_values = jnp.reshape(t.interior_solns, (-1))
            expected_soln = dirichlet_data_fn(all_cheby_points)

        l_inf_error = jnp.max(jnp.abs(computed_soln_values - expected_soln))
        error_vals = error_vals.at[i].set(l_inf_error)

    logging.info(
        "check_l_inf_error_convergence_quad_merge: error_vals = %s", error_vals
    )
    # Plot the convergence on a semi-log plot
    if compute_convergence_rate:
        log_error_vals = jnp.log(error_vals)
        p_values = jnp.array(p_values, dtype=jnp.float64)
        m, b = jnp.polyfit(p_values, log_error_vals, 1)

        xvals = jnp.linspace(p_values[0], p_values[-1], 100)
        yvals = np.exp(m * xvals + b)

    fig, ax = plt.subplots()
    if check_dtn:
        t = "Error convergence for DtN map"
    else:
        t = "Error convergence for Poisson BVP"
    ax.set_title(t)
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(p_values, error_vals, "o-")
    ax.set_yscale("log")
    ax.grid()

    if compute_convergence_rate:
        ax.plot(xvals, yvals, "--", label=f"Slope = {m:.2f}")
        ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def check_l_inf_error_convergence_quad_merge_ItI(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eta: float,
    check_ItI: bool = False,
) -> None:
    """
    This function solves a Poisson-like boundary value problem on the domain [-pi/2, pi/2]^2
    with increasing levels of Chebyshev discretization.

    It computes the solution by solving on 4 separate nodes, and merging DtN maps.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """

    error_vals = jnp.ones_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        t = setup_tree_quad_merge(p, eta=eta, use_ItI=True)
        # print(
        #     "check_l_inf_error_convergence_EW_merge: t.start_tree_NS", t.start_tree_NS
        # )

        # Assemble differential operator coeffs for the Laplacian
        d_xx_coeffs = d_xx_coeff_fn(t.leaf_cheby_points)
        d_yy_coeffs = d_yy_coeff_fn(t.leaf_cheby_points)
        source = source_fn(t.leaf_cheby_points)
        # logging.debug(
        #     "check_l_inf_error_convergence_EW_merge: t.leaf_cheby_points = %s",
        #     t.leaf_cheby_points.shape,
        # )
        # logging.debug(
        #     "check_l_inf_error_convergence_EW_merge: d_xx_coeffs = %s",
        #     d_xx_coeffs.shape,
        # )
        # logging.debug(
        #     "check_l_inf_error_convergence_EW_merge: d_yy_coeffs = %s",
        #     d_yy_coeffs.shape,
        # )
        # Local solve stage

        local_solve_stage(
            t, source_term=source, D_xx_coeffs=d_xx_coeffs, D_yy_coeffs=d_yy_coeffs
        )

        build_stage(t)

        root_bdry_points = t.root_boundary_points
        n_per_side = root_bdry_points.shape[0] // 4
        boundary_f = dirichlet_data_fn(root_bdry_points)

        boundary_normals = jnp.concatenate(
            [
                -1 * dudy_fn(root_bdry_points[:n_per_side]),
                dudx_fn(root_bdry_points[n_per_side : 2 * n_per_side]),
                dudy_fn(root_bdry_points[2 * n_per_side : 3 * n_per_side]),
                -1 * dudx_fn(root_bdry_points[3 * n_per_side :]),
            ]
        )
        incoming_imp_data = boundary_normals + 1j * eta * boundary_f

        # break incoming_imp_data into a list of four arrays
        incoming_imp_data = [
            incoming_imp_data[:n_per_side],
            incoming_imp_data[n_per_side : 2 * n_per_side],
            incoming_imp_data[2 * n_per_side : 3 * n_per_side],
            incoming_imp_data[3 * n_per_side :],
        ]

        down_pass(t, incoming_imp_data)

        if check_ItI:
            expected_out_imp_data = boundary_normals - 1j * eta * boundary_f
            map_ItI = t.interior_node_R_maps[-1]
            computed_out_imp_data = map_ItI @ jnp.concatenate(incoming_imp_data)
            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: expected_out_imp_data shape = %s",
                expected_out_imp_data.shape,
            )
            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: computed_out_imp_data shape = %s",
                computed_out_imp_data.shape,
            )

            expected_soln = expected_out_imp_data
            computed_soln_values = computed_out_imp_data

            # plt.plot(expected_soln.real, "o-", label="Expected.real")
            # plt.plot(expected_soln.imag, "o-", label="Expected.imag")
            # plt.plot(computed_soln_values.real, "x-", label="Computed.real")
            # plt.plot(computed_soln_values.imag, "x-", label="Computed.imag")
            # plt.legend()
            # plt.show()
            # plt.clf()

        else:

            down_pass(t, incoming_imp_data)

            all_cheby_points = jnp.reshape(t.leaf_cheby_points, (-1, 2))
            computed_soln_values = jnp.reshape(t.interior_solns, (-1))
            expected_soln = dirichlet_data_fn(all_cheby_points)

            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: expected_soln = %s",
                expected_soln.shape,
            )
            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: computed_soln_values = %s",
                computed_soln_values.shape,
            )

        l_inf_error = jnp.max(jnp.abs(computed_soln_values - expected_soln))
        error_vals = error_vals.at[i].set(l_inf_error)

    logging.info(
        "check_l_inf_error_convergence_quad_merge: error_vals = %s", error_vals
    )
    # Plot the convergence on a semi-log plot
    if compute_convergence_rate:
        log_error_vals = jnp.log(error_vals)
        p_values = jnp.array(p_values, dtype=jnp.float64)
        m, b = jnp.polyfit(p_values, log_error_vals, 1)

        xvals = jnp.linspace(p_values[0], p_values[-1], 100)
        yvals = np.exp(m * xvals + b)

    fig, ax = plt.subplots()
    if check_ItI:
        t = "Error convergence for ItI map"
    else:
        t = "Error convergence for Poisson BVP"
    ax.set_title(t)
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(p_values, error_vals, "o-")
    ax.set_yscale("log")
    ax.grid()

    if compute_convergence_rate:
        ax.plot(xvals, yvals, "--", label=f"Slope = {m:.2f}")
        ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def check_l_inf_error_convergence_quad_merge_ItI_Helmholtz_like(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    part_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    I_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
    homog_dudx_fn: Callable[[jnp.ndarray], jnp.ndarray],
    homog_dudy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    part_dudx_fn: Callable[[jnp.array], jnp.array],
    part_dudy_fn: Callable[[jnp.array], jnp.array],
    eta: float,
    check_ItI: bool = False,
    check_imp: bool = False,
) -> None:
    """
    This function solves a Poisson-like boundary value problem on the domain
    [-pi/2, pi/2]^2 with increasing order of Chebyshev solution on each panel
    It computes the solution by solving on 4 separate leaves, and merging
    DtN maps.

    It then computes the L_inf error of a few objects:
     - particular solution outgoing impedance (Optional, if check_imp is set True)
     - resulting ItI map (Optional, if check_ItI is set True)
     - solution on interior nodes

    Args:
        p_values (jnp.ndarray): Integers specifying increasing Chebyshev orders.
        dirichlet_data_fn (Callable[[jnp.ndarray], jnp.ndarray]): Boundary data
        part_data_fn (Callable[[jnp.ndarray], jnp.ndarray]): Solution to the
            inhomogeneous problem.
        d_xx_coeff_fn (Callable[[jnp.ndarray], jnp.ndarray]): _description_
        d_yy_coeff_fn (Callable[[jnp.ndarray], jnp.ndarray]): _description_
        source_fn (Callable[[jnp.ndarray], jnp.ndarray]): _description_
        compute_convergence_rate (bool): _description_
        plot_fp (str): _description_
        homog_dudx_fn (Callable[[jnp.ndarray], jnp.ndarray]): _description_
        homog_dudy_fn (Callable[[jnp.ndarray], jnp.ndarray]): _description_
        part_dudx_fn (Callable[[jnp.array], jnp.array]): _description_
        part_dudy_fn (Callable[[jnp.array], jnp.array]): _description_
        eta (float): _description_
        check_ItI (bool, optional): _description_. Defaults to False.
    """
    # south = -jnp.pi / 2
    # north = jnp.pi / 2
    # east = jnp.pi / 2
    # west = -jnp.pi / 2

    error_vals = jnp.ones_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        t = setup_tree_quad_merge(p, eta=eta, use_ItI=True)

        # Assemble differential operator coeffs for the Laplacian
        d_xx_coeffs = d_xx_coeff_fn(t.leaf_cheby_points)
        d_yy_coeffs = d_yy_coeff_fn(t.leaf_cheby_points)
        I_coeffs = I_coeff_fn(t.leaf_cheby_points)
        source = source_fn(t.leaf_cheby_points)

        # Local solve stage
        local_solve_stage(
            t,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            I_coeffs=I_coeffs,
        )

        build_stage(t)

        root_bdry_points = t.root_boundary_points
        n_per_side = root_bdry_points.shape[0] // 4
        # boundary_f = dirichlet_data_fn(root_bdry_points) + part_data_fn(
        #     root_bdry_points
        # )

        boundary_homog_normals = jnp.concatenate(
            [
                -1 * homog_dudy_fn(root_bdry_points[:n_per_side]),
                homog_dudx_fn(root_bdry_points[n_per_side : 2 * n_per_side]),
                homog_dudy_fn(root_bdry_points[2 * n_per_side : 3 * n_per_side]),
                -1 * homog_dudx_fn(root_bdry_points[3 * n_per_side :]),
            ]
        )
        part_boundary_n = jnp.concatenate(
            [
                -1 * part_dudy_fn(root_bdry_points[:n_per_side]),
                part_dudx_fn(root_bdry_points[n_per_side : 2 * n_per_side]),
                part_dudy_fn(root_bdry_points[2 * n_per_side : 3 * n_per_side]),
                -1 * part_dudx_fn(root_bdry_points[3 * n_per_side :]),
            ]
        )
        incoming_imp_data = boundary_homog_normals + 1j * eta * dirichlet_data_fn(
            root_bdry_points
        )

        # break incoming_imp_data into a list of four arrays
        incoming_imp_data = [
            incoming_imp_data[:n_per_side],
            incoming_imp_data[n_per_side : 2 * n_per_side],
            incoming_imp_data[2 * n_per_side : 3 * n_per_side],
            incoming_imp_data[3 * n_per_side :],
        ]

        if check_ItI:
            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI_Helmholtz_like: Checking ItI map"
            )
            # Check that the ItI map can map from incoming to outgoing homogeneous impedance data.
            expected_out_imp_data = (
                boundary_homog_normals - 1j * eta * dirichlet_data_fn(root_bdry_points)
            )
            incoming_imp_data = jnp.concatenate(incoming_imp_data)
            map_ItI = t.interior_node_R_maps[-1]
            computed_out_imp_data = map_ItI @ incoming_imp_data
            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: expected_out_imp_data shape = %s",
                expected_out_imp_data.shape,
            )
            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: computed_out_imp_data shape = %s",
                computed_out_imp_data.shape,
            )

            expected_soln = expected_out_imp_data
            computed_soln_values = computed_out_imp_data

            # plt.plot(expected_soln.real, "o-", label="Expected.real")
            # plt.plot(expected_soln.imag, "o-", label="Expected.imag")
            # plt.plot(computed_soln_values.real, "x-", label="Computed.real")
            # plt.plot(computed_soln_values.imag, "x-", label="Computed.imag")
            # plt.legend()
            # plt.show()
            # plt.clf()
        elif check_imp:
            # Check the outgoing partial impedance data assembled during
            # the merge

            part_boundary_f = part_data_fn(t.root_boundary_points)

            part_boundary_n = jnp.concatenate(
                [
                    -1 * part_dudy_fn(root_bdry_points[:n_per_side]),
                    part_dudx_fn(root_bdry_points[n_per_side : 2 * n_per_side]),
                    part_dudy_fn(root_bdry_points[2 * n_per_side : 3 * n_per_side]),
                    -1 * part_dudx_fn(root_bdry_points[3 * n_per_side :]),
                ]
            )
            # plt.plot(part_boundary_f, "x-", label="V evals")
            # plt.plot(part_boundary_n, "o-", label="V normal evals")
            # plt.legend()
            # plt.grid()
            # plt.show()
            # plt.clf()

            expected_out_part_imp_data = part_boundary_n - 1j * eta * part_boundary_f
            computed_out_part_imp_data = t.h_last.flatten()
            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: expected_out_part_imp_data shape = %s",
                expected_out_part_imp_data.shape,
            )
            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: computed_out_part_imp_data shape = %s",
                computed_out_part_imp_data.shape,
            )

            expected_soln = expected_out_part_imp_data
            computed_soln_values = computed_out_part_imp_data

            # plt.plot(expected_soln.real, "o-", label="Expected.real")
            # plt.plot(expected_soln.imag, "o-", label="Expected.imag")
            # plt.plot(computed_soln_values.real, "x-", label="Computed.real")
            # plt.plot(computed_soln_values.imag, "x-", label="Computed.imag")
            # plt.legend()
            # plt.show()
            # plt.clf()

        else:

            down_pass(t, incoming_imp_data)

            all_cheby_points = jnp.reshape(t.leaf_cheby_points, (-1, 2))
            computed_soln_values = jnp.reshape(t.interior_solns, (-1))
            expected_soln = dirichlet_data_fn(all_cheby_points) + part_data_fn(
                all_cheby_points
            )

            # plot_soln_from_cheby_nodes(
            #     t.leaf_cheby_points.reshape(-1, 2),
            #     corners,
            #     expected_soln.real,
            #     computed_soln_values.real,
            # )

            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: expected_soln = %s",
                expected_soln.shape,
            )
            logging.debug(
                "check_l_inf_error_convergence_quad_merge_ItI: computed_soln_values = %s",
                computed_soln_values.shape,
            )

        l_inf_error = jnp.max(jnp.abs(computed_soln_values - expected_soln))
        error_vals = error_vals.at[i].set(l_inf_error)
        logging.debug(
            "check_l_inf_error_convergence_quad_merge_ItI_Helmholtz_like: l_inf_error: %f",
            l_inf_error,
        )

    logging.info(
        "check_l_inf_error_convergence_quad_merge: error_vals = %s", error_vals
    )
    # Plot the convergence on a semi-log plot
    if compute_convergence_rate:
        log_error_vals = jnp.log(error_vals)
        p_values = jnp.array(p_values, dtype=jnp.float64)
        m, b = jnp.polyfit(p_values, log_error_vals, 1)

        xvals = jnp.linspace(p_values[0], p_values[-1], 100)
        yvals = np.exp(m * xvals + b)

    fig, ax = plt.subplots()
    if check_ItI:
        t = "Error convergence for ItI map"
    else:
        t = "Error convergence for Helmholtz-Like BVP"
    ax.set_title(t)
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(p_values, error_vals, "o-")
    ax.set_yscale("log")
    ax.grid()

    if compute_convergence_rate:
        ax.plot(xvals, yvals, "--", label=f"Slope = {m:.2f}")
        ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def single_merge_check_0(plot_fp: str) -> None:
    """
    Poisson BVP: f(x, y) = x^2 - y^2.
    """
    logging.info("Running single_merge_check_0")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_quad_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_POLY["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
    )


def single_merge_check_1(plot_fp: str) -> None:
    """
    Poisson BVP: f(x, y) = x^2 - y^2. Tests DTN map.
    """
    logging.info("Running single_merge_check_1")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_quad_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_POLY["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        check_dtn=True,
        dudx_fn=TEST_CASE_POISSON_POLY["du_dx_fn"],
        dudy_fn=TEST_CASE_POISSON_POLY["du_dy_fn"],
    )


def single_merge_check_2(plot_fp: str) -> None:
    """
    Poisson BVP: f(x, y) = e^x sin(y).
    """
    logging.info("Running single_merge_check_2")
    p_values = jnp.array([4, 6, 8, 10, 12, 14, 16])
    check_l_inf_error_convergence_quad_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_NONPOLY["source_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
    )


def single_merge_check_3(plot_fp: str) -> None:
    """
    Poisson BVP: f(x, y) = e^x sin(y). Tests DtN map.
    """
    logging.info("Running single_merge_check_3")
    p_values = jnp.array([4, 6, 8, 10, 12, 14, 16])
    check_l_inf_error_convergence_quad_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_NONPOLY["source_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        check_dtn=True,
        dudx_fn=TEST_CASE_POISSON_NONPOLY["du_dx_fn"],
        dudy_fn=TEST_CASE_POISSON_NONPOLY["du_dy_fn"],
    )


def single_merge_check_4(plot_fp: str) -> None:
    """
    This is test case nonconstant coefficients with polynomial data.
    """
    logging.info("Running single_merge_check_4")

    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_quad_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_NONCONSTANT_COEFF_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_NONCONSTANT_COEFF_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_NONCONSTANT_COEFF_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_NONCONSTANT_COEFF_POLY["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
    )


def single_merge_check_5(plot_fp: str) -> None:
    """Checks ItI map with homogeneous problem with polynomial data."""
    logging.info("Running single_merge_check_5")
    p_values = jnp.array([4, 8, 12, 16])

    check_l_inf_error_convergence_quad_merge_ItI(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_POLY["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        dudx_fn=TEST_CASE_POISSON_POLY["du_dx_fn"],
        dudy_fn=TEST_CASE_POISSON_POLY["du_dy_fn"],
        eta=4.0,
        check_ItI=True,
    )


def single_merge_check_6(plot_fp: str) -> None:
    """Checks solution with homogeneous problem with polynomial data."""
    logging.info("Running single_merge_check_6")
    p_values = jnp.array([4, 8, 12, 16])

    check_l_inf_error_convergence_quad_merge_ItI(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_POLY["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        dudx_fn=TEST_CASE_POISSON_POLY["du_dx_fn"],
        dudy_fn=TEST_CASE_POISSON_POLY["du_dy_fn"],
        eta=4.0,
        check_ItI=False,
    )


def single_merge_check_7(plot_fp: str) -> None:
    """Checks solution with non-homogeneous problem with non-homogeneous data."""
    logging.info("Running single_merge_check_7")
    p_values = jnp.array([4, 8, 12, 16, 20])

    check_l_inf_error_convergence_quad_merge_ItI(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_NONPOLY_ITI["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_NONPOLY_ITI["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_NONPOLY_ITI["d_yy_coeff_fn"],
        source_fn=TEST_CASE_NONPOLY_ITI["source_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        dudx_fn=TEST_CASE_NONPOLY_ITI["du_dx_fn"],
        dudy_fn=TEST_CASE_NONPOLY_ITI["du_dy_fn"],
        eta=4.0,
        check_ItI=False,
    )


def single_merge_check_8(plot_fp: str) -> None:
    """Checks computed outgoing impedance data in a Helmholtz problem with nonzero source and non-polynomial data.
    This check is deprecated because we no longer save the outgoing impedance data.
    """

    logging.info("Running single_merge_check_8")

    # p_values = jnp.array([4, 8, 12, 16, 20, 24, 28])

    # check_l_inf_error_convergence_quad_merge_ItI_Helmholtz_like(
    #     p_values=p_values,
    #     dirichlet_data_fn=TEST_CASE_HELMHOLTZ_1["dirichlet_data_fn"],
    #     part_data_fn=TEST_CASE_HELMHOLTZ_1["part_soln_fn"],
    #     d_xx_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_xx_coeff_fn"],
    #     d_yy_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_yy_coeff_fn"],
    #     I_coeff_fn=TEST_CASE_HELMHOLTZ_1["I_coeff_fn"],
    #     source_fn=TEST_CASE_HELMHOLTZ_1["source_fn"],
    #     compute_convergence_rate=True,
    #     plot_fp=plot_fp,
    #     homog_dudx_fn=TEST_CASE_HELMHOLTZ_1["du_dx_fn"],
    #     homog_dudy_fn=TEST_CASE_HELMHOLTZ_1["du_dy_fn"],
    #     part_dudx_fn=TEST_CASE_HELMHOLTZ_1["part_du_dx_fn"],
    #     part_dudy_fn=TEST_CASE_HELMHOLTZ_1["part_du_dy_fn"],
    #     eta=ETA,
    #     check_ItI=False,
    #     check_imp=True,
    # )


def single_merge_check_9(plot_fp: str) -> None:
    """Checks the ItI map in a Helmholtz problem with nonzero source and non-polynomial data."""
    logging.info("Running single_merge_check_9")
    p_values = jnp.array([4, 8, 12, 16, 20, 24])

    check_l_inf_error_convergence_quad_merge_ItI_Helmholtz_like(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_HELMHOLTZ_1["dirichlet_data_fn"],
        part_data_fn=TEST_CASE_HELMHOLTZ_1["part_soln_fn"],
        d_xx_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_yy_coeff_fn"],
        I_coeff_fn=TEST_CASE_HELMHOLTZ_1["I_coeff_fn"],
        source_fn=TEST_CASE_HELMHOLTZ_1["source_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        homog_dudx_fn=TEST_CASE_HELMHOLTZ_1["du_dx_fn"],
        homog_dudy_fn=TEST_CASE_HELMHOLTZ_1["du_dy_fn"],
        part_dudx_fn=TEST_CASE_HELMHOLTZ_1["part_du_dx_fn"],
        part_dudy_fn=TEST_CASE_HELMHOLTZ_1["part_du_dy_fn"],
        eta=ETA,
        check_ItI=True,
        check_imp=False,
    )


def single_merge_check_10(plot_fp: str) -> None:
    """Checks computed solution in a Helmholtz problem with nonzero source and non-polynomial data."""
    logging.info("Running single_merge_check_10")
    p_values = jnp.array([4, 8, 12, 16, 20, 24])

    def z(x):
        return jnp.zeros_like(x[..., 0])

    check_l_inf_error_convergence_quad_merge_ItI_Helmholtz_like(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_HELMHOLTZ_1["dirichlet_data_fn"],
        # dirichlet_data_fn=z,
        part_data_fn=TEST_CASE_HELMHOLTZ_1["part_soln_fn"],
        d_xx_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_yy_coeff_fn"],
        I_coeff_fn=TEST_CASE_HELMHOLTZ_1["I_coeff_fn"],
        source_fn=TEST_CASE_HELMHOLTZ_1["source_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        homog_dudx_fn=TEST_CASE_HELMHOLTZ_1["du_dx_fn"],
        homog_dudy_fn=TEST_CASE_HELMHOLTZ_1["du_dy_fn"],
        # homog_dudx_fn=z,
        # homog_dudy_fn=z,
        part_dudx_fn=TEST_CASE_HELMHOLTZ_1["part_du_dx_fn"],
        part_dudy_fn=TEST_CASE_HELMHOLTZ_1["part_du_dy_fn"],
        eta=ETA,
        check_ItI=False,
        check_imp=False,
    )
