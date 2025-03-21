import logging
from typing import Callable
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from hps.accuracy_checks.dirichlet_neumann_data import (
    TEST_CASE_POISSON_POLY,
    TEST_CASE_POISSON_NONPOLY,
    TEST_CASE_POLY_PART_HOMOG,
    TEST_CASE_HELMHOLTZ_1,
    TEST_CASE_HELMHOLTZ_0,
)
from hps.accuracy_checks.utils import plot_soln_from_cheby_nodes

from hps.src.solver_obj import (
    create_solver_obj_2D,
    get_bdry_data_evals_lst_2D,
)
from hps.src.up_down_passes import build_stage, down_pass, local_solve_stage
from hps.src.quadrature.trees import Node
from hps.src.methods.fused_methods import (
    _fused_local_solve_and_build_2D_ItI,
    _down_pass_from_fused_ItI,
)


def check_l_inf_error_convergence_fixed_p_increasing_l(
    p: int,
    l_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
    check_dtn: bool = False,
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    particular_soln_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^2
    with increasing levels of Chebyshev discretization.

    It computes the solution by breaking the domain into an increasingly large number of patches,
    on each patch we discretize with a Chebyshev grid of size pxp.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """

    # Set up the domain
    south = -jnp.pi / 2
    north = jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2

    n_X = 100
    x = jnp.linspace(west, east, n_X)
    X, Y = jnp.meshgrid(x, x)

    error_vals = jnp.zeros_like(l_values, dtype=jnp.float64)

    for i, l in enumerate(l_values):
        l = int(l)
        # print("check_l_inf_error_convergence_fixed_p_increasing_l: l = ", l)
        q = p - 2

        # Set up the Tree
        root = Node(
            xmin=west,
            xmax=east,
            ymin=south,
            ymax=north,
            depth=0,
            zmin=None,
            zmax=None,
        )
        tree = create_solver_obj_2D(p=p, q=q, root=root, uniform_levels=l)

        # Create coeffs for Laplacian
        d_xx_coeffs = d_xx_coeff_fn(tree.leaf_cheby_points)
        d_yy_coeffs = d_yy_coeff_fn(tree.leaf_cheby_points)

        # Check this is an elliptic problem:
        assert jnp.all(d_xx_coeffs * d_yy_coeffs > 0)

        source = source_fn(tree.leaf_cheby_points)
        local_solve_stage(
            tree,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )

        # Do the upward pass
        build_stage(tree)

        # Define the boundary data
        # boundary_points = tree.root_boundary_points
        boundary_data_lst = get_bdry_data_evals_lst_2D(tree, dirichlet_data_fn)

        if check_dtn:
            # Evaluate the accuracy of the DtN map
            # Compute n_per_side which is number of Gauss points per side
            # of the root node.
            n_per_side = tree.root_boundary_points.shape[0] // 4
            expected_soln_vals = jnp.concatenate(
                [
                    # S boundary
                    -1 * dudy_fn(tree.root_boundary_points[:n_per_side]),
                    dudx_fn(
                        tree.root_boundary_points[n_per_side : 2 * n_per_side]
                    ),
                    dudy_fn(
                        tree.root_boundary_points[
                            2 * n_per_side : 3 * n_per_side
                        ]
                    ),
                    # W boundary
                    -1 * dudx_fn(tree.root_boundary_points[3 * n_per_side :]),
                ]
            )
            all_soln_vals = tree.root.DtN @ jnp.concatenate(boundary_data_lst)

            # plt.plot(expected_soln_vals, "-x", label="Expected")
            # plt.plot(all_soln_vals, "-o", label="Computed")
            # plt.legend()
            # plt.show()

        else:
            # Evaluate the solution at all of the Chebyshev points
            # Do the downward pass
            down_pass(tree, boundary_data_lst)

            all_cheby_pts = jnp.reshape(tree.leaf_cheby_points, (-1, 2))
            all_soln_vals = jnp.reshape(tree.interior_solns, (-1))

            expected_soln_vals = dirichlet_data_fn(all_cheby_pts).flatten()

            if particular_soln_fn is not None:
                expected_soln_vals = (
                    expected_soln_vals
                    + particular_soln_fn(all_cheby_pts).flatten()
                )

            # plot_soln_from_cheby_nodes(
            #     cheby_nodes=all_cheby_pts,
            #     corners=corners,
            #     expected_soln=expected_soln_vals,
            #     computed_soln=all_soln_vals,
            # )
        error_vals = error_vals.at[i].set(
            jnp.max(jnp.abs(all_soln_vals - expected_soln_vals))
        )
    logging.info(
        "check_l_inf_error_convergence_fixed_p_increasing_l: error_vals = %s",
        error_vals,
    )
    # # Plot the convergence
    if compute_convergence_rate:
        log_error_vals = jnp.log(error_vals)
        log_l_values = jnp.log(l_values)
        m, b = jnp.polyfit(log_l_values, log_error_vals, 1)

        xvals = jnp.linspace(l_values[0], l_values[-1], 100)
        yvals = np.exp(m * np.log(xvals) + b)

    fig, ax = plt.subplots()
    if check_dtn:
        t = "Error convergence for DtN map"
    else:
        t = "Error convergence for Poisson BVP"
    ax.set_title(t)
    ax.set_xlabel("l = # levels in binary tree")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(l_values, error_vals, "o-")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid()

    if compute_convergence_rate:
        ax.plot(xvals, yvals, "--", label=f"Slope = {m:.2f}")
        ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def check_l_inf_error_convergence_fixed_l_increasing_p(
    p_values: jnp.ndarray,
    l: int,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^2
    with increasing levels of Chebyshev discretization.

    It computes the solution by breaking the domain into 2**l  patches,
    on each patch we discretize with a Chebyshev grid of size pxp, for varying values of p.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """

    # Set up the domain
    south = -jnp.pi / 2
    north = jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2
    corners = [(west, south), (east, south), (east, north), (west, north)]

    n_X = 100
    x = jnp.linspace(west, east, n_X)
    X, Y = jnp.meshgrid(x, x)

    error_vals = jnp.zeros_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        l = int(l)
        p = int(p)
        # print("check_l_inf_error_convergence_fixed_l_increasing_p: p = ", p)
        q = p - 2

        # Set up the Tree
        root = Node(xmin=west, xmax=east, ymin=south, ymax=north, depth=0)
        tree = create_solver_obj_2D(p, q, root, uniform_levels=l)

        # Create coeffs for Laplacian
        d_xx_coeffs = d_xx_coeff_fn(tree.leaf_cheby_points)
        d_yy_coeffs = d_yy_coeff_fn(tree.leaf_cheby_points)

        # Check this problem is elliptic
        assert jnp.all(d_xx_coeffs * d_yy_coeffs > 0)
        source = source_fn(tree.leaf_cheby_points)
        # Do the local solve stage
        local_solve_stage(
            tree,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )

        # Do the upward pass
        build_stage(tree)

        # Define the boundary data
        boundary_points = tree.root_boundary_points
        boundary_data = dirichlet_data_fn(boundary_points)

        # Do the downward pass
        down_pass(tree, boundary_data)

        all_cheby_pts = jnp.reshape(tree.leaf_cheby_points, (-1, 2))
        # all_soln_vals = jnp.reshape(tree.interior_solns, (-1))
        all_soln_vals = tree.interior_solns.flatten()

        expected_soln_vals = dirichlet_data_fn(
            tree.leaf_cheby_points
        ).flatten()

        error_vals = error_vals.at[i].set(
            jnp.max(jnp.abs(all_soln_vals - expected_soln_vals))
        )

        plot_soln_from_cheby_nodes(
            all_cheby_pts, corners, expected_soln_vals, all_soln_vals
        )

    logging.info(
        "check_l_inf_error_convergence_fixed_l_increasing_p: error_vals = %s",
        error_vals,
    )
    # Plot the convergence on a semilog plot
    if compute_convergence_rate:
        p_values = jnp.array(p_values, dtype=jnp.float64)
        log_error_vals = jnp.log(error_vals)
        m, b = jnp.polyfit(p_values, log_error_vals, 1)

        xvals = jnp.linspace(p_values[0], p_values[-1], 100)
        yvals = np.exp(m * xvals + b)

    fig, ax = plt.subplots()
    ax.set_title(
        f"Error convergence for Poisson BVP split into {2**l} patches"
    )
    ax.set_xlabel("p = Order of Chebyshev grid for each patch")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(p_values, error_vals, "o-")
    ax.set_yscale("log")
    ax.grid()

    if compute_convergence_rate:
        ax.plot(xvals, yvals, "--", label=f"Slope = {m:.2f}")
        ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def check_l_inf_error_convergence_fixed_l_increasing_p_ItI(
    p_values: jnp.ndarray,
    l: int,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    check_ItI: bool,
    compute_convergence_rate: bool,
    plot_fp: str,
    eta: float,
    I_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    part_soln_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
) -> None:
    # Set up the domain
    south = -jnp.pi / 2
    north = jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2

    n_X = 100
    x = jnp.linspace(west, east, n_X)
    X, Y = jnp.meshgrid(x, x)

    error_vals = jnp.zeros_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        l = int(l)
        p = int(p)
        # print("check_l_inf_error_convergence_fixed_l_increasing_p: p = ", p)
        q = p - 2

        # Set up the Tree
        root = Node(xmin=west, xmax=east, ymin=south, ymax=north, depth=0)
        tree = create_solver_obj_2D(
            p, q, root, uniform_levels=l, use_ItI=True, eta=eta
        )

        # Create coeffs for Laplacian
        d_xx_coeffs = d_xx_coeff_fn(tree.leaf_cheby_points)
        d_yy_coeffs = d_yy_coeff_fn(tree.leaf_cheby_points)
        if I_coeff_fn is not None:
            I_coeffs = I_coeff_fn(tree.leaf_cheby_points)
        else:
            I_coeffs = None

        # Check this is an elliptic problem:
        assert jnp.all(d_xx_coeffs * d_yy_coeffs > 0)

        source = source_fn(tree.leaf_cheby_points)

        S_arr_lst, R_arr_lst, f_arr_lst = _fused_local_solve_and_build_2D_ItI(
            D_xx=tree.D_xx,
            D_xy=tree.D_xy,
            D_yy=tree.D_yy,
            D_x=tree.D_x,
            D_y=tree.D_y,
            I_P_0=tree.I_P_0,
            Q_I=tree.Q_I,
            F=tree.F,
            G=tree.G,
            p=tree.p,
            l=tree.l,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            I_coeffs=I_coeffs,
        )

        root_bdry_points = tree.root_boundary_points
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

        if check_ItI:
            expected_out_imp_data = boundary_normals - 1j * eta * boundary_f
            map_ItI = R_arr_lst[-1]

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
        else:
            # Do the down pass
            computed_soln_values = _down_pass_from_fused_ItI(
                bdry_data=incoming_imp_data,
                S_arr_lst=S_arr_lst,
                f_lst=f_arr_lst,
                D_xx=tree.D_xx,
                D_xy=tree.D_xy,
                D_yy=tree.D_yy,
                D_x=tree.D_x,
                D_y=tree.D_y,
                I_P_0=tree.I_P_0,
                Q_I=tree.Q_I,
                F=tree.F,
                G=tree.G,
                p=tree.p,
                l=tree.l,
                source_term=source,
                D_xx_coeffs=d_xx_coeffs,
                D_yy_coeffs=d_yy_coeffs,
                I_coeffs=I_coeffs,
            ).flatten()
            expected_soln = dirichlet_data_fn(tree.leaf_cheby_points).flatten()
            if part_soln_fn is not None:
                expected_soln = (
                    expected_soln
                    + part_soln_fn(tree.leaf_cheby_points).flatten()
                )

            # Plot the expected and computed solutions
            # plot_soln_from_cheby_nodes(
            #     tree.leaf_cheby_points.reshape(-1, 2),
            #     corners,
            #     expected_soln.real,
            #     computed_soln_values.real,
            # )
        l_inf_error = jnp.max(jnp.abs(computed_soln_values - expected_soln))
        error_vals = error_vals.at[i].set(l_inf_error)
    logging.info(
        "check_l_inf_error_convergence_fixed_l_increasing_p_ItI: error_vals = %s",
        error_vals,
    )


def multi_merge_check_0(plot_fp: str) -> None:
    """
    Poisson BVP on [-pi/2, pi/2]^2 with Dirichlet data given by a polynomial.

    Measures error against # of patch levels for fixed p = 8
    """
    logging.info("Running multi_merge_check_0")
    l_values = jnp.array([1, 2, 3, 4])
    p = 6

    check_l_inf_error_convergence_fixed_p_increasing_l(
        p,
        l_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_POLY["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
    )


def multi_merge_check_1(plot_fp: str) -> None:
    """
    Poisson BVP on [-pi/2, pi/2]^2 with Dirichlet data given by a non-polynomial function.

    Measures error against # of patch levels for fixed p = 16
    """
    logging.info("Running multi_merge_check_1")
    l_values = jnp.array([1, 2, 3, 4])
    p = 6

    check_l_inf_error_convergence_fixed_p_increasing_l(
        p,
        l_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_NONPOLY["source_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
    )


def multi_merge_check_2(plot_fp: str) -> None:
    """
    Measures convergence of DtN map on [-pi/2, pi/2]^2 with Dirichlet data given by a polynomial funciton.

    Measures error against l for fixed p = 6, and checks the accuracy of the DtN map.
    """
    logging.info("Running multi_merge_check_2")
    l_values = jnp.array([1, 2, 3, 4])
    p = 6

    check_l_inf_error_convergence_fixed_p_increasing_l(
        p,
        l_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_POLY["source_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        check_dtn=True,
        dudx_fn=TEST_CASE_POISSON_POLY["du_dx_fn"],
        dudy_fn=TEST_CASE_POISSON_POLY["du_dy_fn"],
    )


def multi_merge_check_3(plot_fp: str) -> None:
    """
    Measures convergence of DtN map on [-pi/2, pi/2]^2 with Dirichlet data given by a non-polynomial function.

    Measures error against l for fixed p = 6, and checks the accuracy of the DtN map.
    """
    logging.info("Running multi_merge_check_3")
    l_values = jnp.array([1, 2, 3, 4])
    p = 6

    check_l_inf_error_convergence_fixed_p_increasing_l(
        p,
        l_values,
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


def multi_merge_check_4(plot_fp: str) -> None:
    """
    Polynomial data with nonconstant coefficiencts and nonzero source function.
    """
    logging.info("Running multi_merge_check_4")
    l_values = jnp.array([1, 2, 3, 4])
    p = 6
    check_l_inf_error_convergence_fixed_p_increasing_l(
        p,
        l_values,
        dirichlet_data_fn=TEST_CASE_POLY_PART_HOMOG["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POLY_PART_HOMOG["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POLY_PART_HOMOG["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POLY_PART_HOMOG["source_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        particular_soln_fn=TEST_CASE_POLY_PART_HOMOG["particular_soln_fn"],
    )


def multi_merge_check_5(plot_fp: str) -> None:
    """
    ItI with non-polynomial data.
    """
    logging.info("Running multi_merge_check_5")
    p_values = jnp.array([4, 8, 12, 16])
    l = 3
    check_l_inf_error_convergence_fixed_l_increasing_p_ItI(
        p_values=p_values,
        l=l,
        dirichlet_data_fn=TEST_CASE_HELMHOLTZ_1["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_yy_coeff_fn"],
        source_fn=TEST_CASE_HELMHOLTZ_1["source_fn"],
        I_coeff_fn=TEST_CASE_HELMHOLTZ_1["I_coeff_fn"],
        dudx_fn=TEST_CASE_HELMHOLTZ_1["du_dx_fn"],
        dudy_fn=TEST_CASE_HELMHOLTZ_1["du_dy_fn"],
        check_ItI=True,
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        eta=TEST_CASE_HELMHOLTZ_1["eta"],
    )


def multi_merge_check_6(plot_fp: str) -> None:
    """
    ItI with non-polynomial data. Without a particular solution.
    """
    logging.info("Running multi_merge_check_6")
    p_values = jnp.array([4, 8, 12, 16])
    l = 3
    check_l_inf_error_convergence_fixed_l_increasing_p_ItI(
        p_values=p_values,
        l=l,
        dirichlet_data_fn=TEST_CASE_HELMHOLTZ_0["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_HELMHOLTZ_0["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_HELMHOLTZ_0["d_yy_coeff_fn"],
        source_fn=TEST_CASE_HELMHOLTZ_0["source_fn"],
        I_coeff_fn=TEST_CASE_HELMHOLTZ_0["I_coeff_fn"],
        dudx_fn=TEST_CASE_HELMHOLTZ_0["du_dx_fn"],
        dudy_fn=TEST_CASE_HELMHOLTZ_0["du_dy_fn"],
        check_ItI=False,
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        part_soln_fn=TEST_CASE_HELMHOLTZ_0["part_soln_fn"],
        eta=TEST_CASE_HELMHOLTZ_0["eta"],
    )


def multi_merge_check_7(plot_fp: str) -> None:
    """
    ItI with non-polynomial data. With a particular solution
    """
    logging.info("Running multi_merge_check_7")
    p_values = jnp.array([4, 8, 12, 16])
    l = 3
    check_l_inf_error_convergence_fixed_l_increasing_p_ItI(
        p_values=p_values,
        l=l,
        dirichlet_data_fn=TEST_CASE_HELMHOLTZ_1["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_yy_coeff_fn"],
        source_fn=TEST_CASE_HELMHOLTZ_1["source_fn"],
        I_coeff_fn=TEST_CASE_HELMHOLTZ_1["I_coeff_fn"],
        dudx_fn=TEST_CASE_HELMHOLTZ_1["du_dx_fn"],
        dudy_fn=TEST_CASE_HELMHOLTZ_1["du_dy_fn"],
        check_ItI=False,
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        part_soln_fn=TEST_CASE_HELMHOLTZ_1["part_soln_fn"],
        eta=TEST_CASE_HELMHOLTZ_1["eta"],
    )
