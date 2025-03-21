import logging
from typing import Callable
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from hps.src.quadrature.quad_3D.differentiation import precompute_diff_operators
from hps.src.quadrature.quad_3D.interpolation import (
    precompute_P_matrix,
)
from hps.src.quadrature.trees import Node
from hps.src.methods.local_solve_stage import _local_solve_stage_3D
from hps.src.solver_obj import create_solver_obj_3D

from hps.accuracy_checks.test_cases_3D import (
    K_DIRICHLET,
    K_SOURCE,
    K_XX_COEFF,
    K_YY_COEFF,
    K_ZZ_COEFF,
    K_DU_DX,
    K_DU_DY,
    K_DU_DZ,
    K_PART_SOLN,
    K_PART_SOLN_DUDX,
    K_PART_SOLN_DUDY,
    K_PART_SOLN_DUDZ,
    TEST_CASE_POISSON_POLY,
    TEST_CASE_POISSON_NONPOLY,
    TEST_CASE_HOMOG_PART_POLY,
)


def check_l_inf_error_convergence_3D(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_zz_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
    check_dtn: bool = False,
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudz_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^3 with
    increasing levels of Chebyshev discretization.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """

    error_vals = jnp.zeros_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        # print("check_l_inf_error_convergence: p = ", p)
        q = p - 2

        root = Node(
            xmin=-jnp.pi / 2,
            xmax=jnp.pi / 2,
            ymin=-jnp.pi / 2,
            ymax=jnp.pi / 2,
            zmin=-jnp.pi / 2,
            zmax=jnp.pi / 2,
            depth=0,
        )

        t = create_solver_obj_3D(p=p, q=q, root=root)

        cheby_pts = t.leaf_cheby_points

        d_xx_coeffs = d_xx_coeff_fn(cheby_pts)
        d_yy_coeffs = d_yy_coeff_fn(cheby_pts)
        d_zz_coeffs = d_zz_coeff_fn(cheby_pts)
        source = source_fn(cheby_pts)

        # d_xx_coeffs = jnp.expand_dims(d_xx_coeffs, axis=0)
        # d_yy_coeffs = jnp.expand_dims(d_yy_coeffs, axis=0)
        # d_zz_coeffs = jnp.expand_dims(d_zz_coeffs, axis=0)
        # source = jnp.expand_dims(source, axis=0)

        # Put in 8 copy of the data along the 0th axis
        # to simulate the other part of the merge oct so the
        # reshape operation works
        d_xx_coeffs = jnp.repeat(d_xx_coeffs, 8, axis=0)
        d_yy_coeffs = jnp.repeat(d_yy_coeffs, 8, axis=0)
        d_zz_coeffs = jnp.repeat(d_zz_coeffs, 8, axis=0)
        source = jnp.repeat(source, 8, axis=0)

        D_x, D_y, D_z, D_xx, D_yy, D_zz, D_xy, D_xz, D_yz = precompute_diff_operators(
            p, 1.0
        )
        P = precompute_P_matrix(p, q)

        sidelens = jnp.array([root.xmax - root.xmin for _ in range(8)])

        DtN, v, v_prime, Y = _local_solve_stage_3D(
            D_xx=D_xx,
            D_xy=D_xy,
            D_yy=D_yy,
            D_xz=D_xz,
            D_yz=D_yz,
            D_zz=D_zz,
            D_x=D_x,
            D_y=D_y,
            D_z=D_z,
            P=P,
            p=p,
            q=q,
            source_term=source,
            sidelens=sidelens,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            D_zz_coeffs=d_zz_coeffs,
        )
        # only need the first leaf's solution
        Y = Y[0]
        v = v[0]
        DtN = DtN.squeeze()[0]
        v_prime = v_prime.squeeze()[0]

        bdry_pts = t.root_boundary_points

        boundary_data = dirichlet_data_fn(bdry_pts)
        if check_dtn:
            expected_soln = jnp.concatenate(
                [
                    -1 * dudx_fn(bdry_pts[: q**2]),  # face 1
                    dudx_fn(bdry_pts[q**2 : 2 * q**2]),  # face 2
                    -1 * dudy_fn(bdry_pts[2 * q**2 : 3 * q**2]),  # face 3
                    dudy_fn(bdry_pts[3 * q**2 : 4 * q**2]),  # face 4
                    -1 * dudz_fn(bdry_pts[4 * q**2 : 5 * q**2]),  # face 5
                    dudz_fn(bdry_pts[5 * q**2 :]),  # face 6
                ]
            )
            computed_soln_0 = DtN @ boundary_data
            computed_soln = computed_soln_0 + v_prime
            # logging.debug(
            #     "check_l_inf_error_convergence: computed_soln_0 shape = %s",
            #     computed_soln_0.shape,
            # )
            # logging.debug(
            #     "check_l_inf_error_convergence: computed_soln shape = %s",
            #     computed_soln.shape,
            # )
            # logging.debug(
            #     "check_l_inf_error_convergence: expected_soln shape = %s",
            #     expected_soln.shape,
            # )
            # plt.plot(expected_soln.flatten(), "x-", label="Expected soln")
            # plt.plot(computed_soln.flatten(), "o-", label="Computed soln")
            # plt.legend()
            # plt.show()
        else:
            expected_soln = dirichlet_data_fn(cheby_pts)
            computed_soln = Y @ boundary_data + v

        # val = computed_soln[0] / expected_soln[0]
        # print("check_l_inf_error_convergence: val = ", val)
        # print("check_l_inf_error_convergence: 1 / val = ", 1 / val)
        # print("check_l_inf_error_convergence: sidelen = ", sidelens[0])

        # plt.plot(expected_soln.flatten(), "x-", label="Expected soln")
        # plt.plot(computed_soln.flatten(), "o-", label="Computed soln")
        # plt.legend()
        # plt.show()
        # plt.clf()

        l_inf_error = jnp.max(jnp.abs(computed_soln - expected_soln))
        logging.debug(
            "check_l_inf_error_convergence: p=%i, l_inf_error = %s", p, l_inf_error
        )
        error_vals = error_vals.at[i].set(l_inf_error)
    logging.info("check_l_inf_error_convergence: error_vals = %s", error_vals)
    if compute_convergence_rate:
        # Compute convergence on a semilog plot
        p_values = p_values.astype(np.float64)
        log_error_vals = jnp.log(error_vals)
        m, b = jnp.polyfit(p_values, log_error_vals, 1)

        xvals = jnp.linspace(p_values[0], p_values[-1], 100)
        yvals = np.exp(m * xvals + b)

    fig, ax = plt.subplots()
    ax.set_title("Error convergence for Poisson BVP; Single Patch")
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


def check_l_inf_error_convergence_particular_homog_solns_3D(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_zz_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    homogeneous_soln_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_soln_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_dx_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_dy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_dz_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^3 with
    increasing levels of Chebyshev discretization.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """

    error_vals_homog = jnp.zeros_like(p_values, dtype=jnp.float64)
    error_vals_particular = jnp.zeros_like(p_values, dtype=jnp.float64)
    error_vals_particular_flux = jnp.zeros_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        # print("check_l_inf_error_convergence: p = ", p)
        q = p - 2

        # Set up the domain
        root = Node(
            xmin=-jnp.pi / 2,
            xmax=jnp.pi / 2,
            ymin=-jnp.pi / 2,
            ymax=jnp.pi / 2,
            zmin=-jnp.pi / 2,
            zmax=jnp.pi / 2,
            depth=0,
        )

        # Precompute stuff
        t = create_solver_obj_3D(p=p, q=q, root=root)

        cheby_pts = t.leaf_cheby_points
        bdry_pts = t.root_boundary_points

        d_xx_coeffs = d_xx_coeff_fn(cheby_pts)
        d_yy_coeffs = d_yy_coeff_fn(cheby_pts)
        d_zz_coeffs = d_zz_coeff_fn(cheby_pts)
        source = source_fn(cheby_pts)

        # Put in a second, third, and fourth copy of the data along the 0th axis
        # to simulate the other parts of the merge quad so the
        # reshape operation works
        d_xx_coeffs = jnp.repeat(d_xx_coeffs, 8, axis=0)
        d_yy_coeffs = jnp.repeat(d_yy_coeffs, 8, axis=0)
        d_zz_coeffs = jnp.repeat(d_zz_coeffs, 8, axis=0)
        source = jnp.repeat(source, 8, axis=0)

        D_x, D_y, D_z, D_xx, D_yy, D_zz, D_xy, D_xz, D_yz = precompute_diff_operators(
            p, 1.0
        )
        P = precompute_P_matrix(p, q)

        sidelens = jnp.array([root.xmax - root.xmin for _ in range(8)])

        DtN, v, v_prime, Y = _local_solve_stage_3D(
            D_xx=D_xx,
            D_xy=D_xy,
            D_yy=D_yy,
            D_xz=D_xz,
            D_yz=D_yz,
            D_zz=D_zz,
            D_x=D_x,
            D_y=D_y,
            D_z=D_z,
            P=P,
            p=p,
            q=q,
            source_term=source,
            sidelens=sidelens,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            D_zz_coeffs=d_zz_coeffs,
        )
        # only need the first leaf's solution
        Y = Y[1]
        v = v[1]
        DtN = DtN.squeeze()[1]
        v_prime = v_prime.squeeze()[1]

        boundary_data = dirichlet_data_fn(bdry_pts)

        # Check homogeneous soln
        expected_h_soln = homogeneous_soln_fn(cheby_pts)
        computed_h_soln = Y @ boundary_data
        l_inf_error = jnp.max(jnp.abs(expected_h_soln - computed_h_soln))
        error_vals_homog = error_vals_homog.at[i].set(l_inf_error)

        # Check particular soln
        expected_particular_soln = particular_soln_fn(cheby_pts)
        l_inf_error = jnp.max(jnp.abs(expected_particular_soln - v))
        error_vals_particular = error_vals_particular.at[i].set(l_inf_error)

        # Check particular flux
        expected_particular_flux = jnp.concatenate(
            [
                -1 * particular_dx_fn(bdry_pts[: q**2]),  # face 1
                particular_dx_fn(bdry_pts[q**2 : 2 * q**2]),  # face 2
                -1 * particular_dy_fn(bdry_pts[2 * q**2 : 3 * q**2]),  # face 3
                particular_dy_fn(bdry_pts[3 * q**2 : 4 * q**2]),  # face 4
                -1 * particular_dz_fn(bdry_pts[4 * q**2 : 5 * q**2]),  # face 5
                particular_dz_fn(bdry_pts[5 * q**2 :]),  # face 6
            ]
        )

        l_inf_error = jnp.max(jnp.abs(expected_particular_flux - v_prime))
        error_vals_particular_flux = error_vals_particular_flux.at[i].set(l_inf_error)
        logging.debug(
            "check_l_inf_error_convergence_particular_homog_solns: p=%i, error homogeneous soln=%f, error particular soln=%f, error part soln flux=%f",
            p,
            error_vals_homog[i],
            error_vals_particular[i],
            error_vals_particular_flux[i],
        )

    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns: error_vals_homog = %s",
        error_vals_homog,
    )
    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns: error_vals_particular = %s",
        error_vals_particular,
    )
    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns: error_vals_particular_flux = %s",
        error_vals_particular_flux,
    )
    if compute_convergence_rate:
        # Compute convergence on a semilog plot
        p_values = p_values.astype(np.float64)
        xvals = jnp.linspace(p_values[0], p_values[-1], 100)

        log_h_error_vals = jnp.log(error_vals_homog)
        m_h, b_h = jnp.polyfit(p_values, log_h_error_vals, 1)
        yvals_h = np.exp(m_h * xvals + b_h)

        log_p_error_vals = jnp.log(error_vals_particular)
        m_p, b_p = jnp.polyfit(p_values, log_p_error_vals, 1)
        yvals_p = np.exp(m_p * xvals + b_p)

        log_pf_error_vals = jnp.log(error_vals_particular_flux)
        m_pf, b_pf = jnp.polyfit(p_values, log_pf_error_vals, 1)
        yvals_pf = np.exp(m_pf * xvals + b_pf)

    fig, ax = plt.subplots()
    ax.set_title("Error convergence for Poisson BVP; Single Patch")
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(p_values, error_vals_homog, "o-", label="Homogeneous soln", color="red")
    if compute_convergence_rate:
        ax.plot(xvals, yvals_h, "--", label=f"Slope = {m_h:.2f}", color="red")
    ax.plot(
        p_values, error_vals_particular, "o-", label="Particular soln", color="blue"
    )
    if compute_convergence_rate:
        ax.plot(xvals, yvals_p, "--", label=f"Slope = {m_p:.2f}", color="blue")
    ax.plot(
        p_values,
        error_vals_particular_flux,
        "o-",
        label="Particular flux",
        color="green",
    )
    if compute_convergence_rate:
        ax.plot(xvals, yvals_pf, "--", label=f"Slope = {m_pf:.2f}", color="green")
    ax.set_yscale("log")
    ax.grid()

    ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def single_leaf_check_3D_0(plot_fp: str) -> None:
    """
    Poisson boundary problem with dirichlet data f(x, y) = x^2 + y^2 - 2z^2
    """
    logging.info("Running single_leaf_check_3D_0")
    p_values = jnp.array([4, 8, 12])
    check_l_inf_error_convergence_3D(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_POLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_POLY[K_SOURCE],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
    )


def single_leaf_check_3D_1(plot_fp: str) -> None:
    """
    Poisson boundary problem with dirichlet data f(x, y) = x^2 + y^2 - 2z^2.
    Checks the DtN map.
    """
    logging.info("Running single_leaf_check_3D_1")
    p_values = jnp.array([4, 8, 12])
    check_l_inf_error_convergence_3D(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_POLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_POLY[K_SOURCE],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        check_dtn=True,
        dudx_fn=TEST_CASE_POISSON_POLY[K_DU_DX],
        dudy_fn=TEST_CASE_POISSON_POLY[K_DU_DY],
        dudz_fn=TEST_CASE_POISSON_POLY[K_DU_DZ],
    )


def single_leaf_check_3D_2(plot_fp: str) -> None:
    """
    Poisson boundary problem with dirichlet data f(x, y) = e^x sin(sqrt(2)y) e^z
    """
    logging.info("Running single_leaf_check_3D_2")
    p_values = jnp.array([4, 6, 8, 10, 12])
    check_l_inf_error_convergence_3D(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_NONPOLY[K_SOURCE],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
    )


def single_leaf_check_3D_3(plot_fp: str) -> None:
    """
    Poisson boundary problem with dirichlet data f(x, y) = e^x sin(sqrt(2)y) e^z.
    Checks the DtN map.
    """
    logging.info("Running single_leaf_check_3D_3")
    p_values = jnp.array([4, 6, 8, 10, 12])
    check_l_inf_error_convergence_3D(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_NONPOLY[K_SOURCE],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        check_dtn=True,
        dudx_fn=TEST_CASE_POISSON_NONPOLY[K_DU_DX],
        dudy_fn=TEST_CASE_POISSON_NONPOLY[K_DU_DY],
        dudz_fn=TEST_CASE_POISSON_NONPOLY[K_DU_DZ],
    )


def single_leaf_check_3D_4(plot_fp: str) -> None:
    """
    Tests the convergence of the particular solution, homogeneous solution, and particular solution, with polynomial source and dirichlet data.
    """
    logging.info("Running single_leaf_check_3D_4")
    p_values = jnp.array([4, 8, 12])
    check_l_inf_error_convergence_particular_homog_solns_3D(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_HOMOG_PART_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_HOMOG_PART_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_HOMOG_PART_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_HOMOG_PART_POLY[K_ZZ_COEFF],
        homogeneous_soln_fn=TEST_CASE_HOMOG_PART_POLY[K_DIRICHLET],
        particular_soln_fn=TEST_CASE_HOMOG_PART_POLY[K_PART_SOLN],
        particular_dx_fn=TEST_CASE_HOMOG_PART_POLY[K_PART_SOLN_DUDX],
        particular_dy_fn=TEST_CASE_HOMOG_PART_POLY[K_PART_SOLN_DUDY],
        particular_dz_fn=TEST_CASE_HOMOG_PART_POLY[K_PART_SOLN_DUDZ],
        source_fn=TEST_CASE_HOMOG_PART_POLY[K_SOURCE],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
    )
