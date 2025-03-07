import logging
from typing import Tuple, Callable
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

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
from hps.accuracy_checks.utils import (
    _distance_around_boundary,
    setup_tree_quad_merge,
    plot_soln_from_cheby_nodes,
)
from hps.src.up_down_passes import build_stage, down_pass, local_solve_stage
from hps.src.solver_obj import (
    SolverObj,
    create_solver_obj_3D,
    get_bdry_data_evals_lst_3D,
)
from hps.src.quadrature.trees import Node, get_all_leaves, get_nodes_at_level


def check_l_inf_error_convergence_oct_merge(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_zz_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    soln_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
    check_dtn: bool = False,
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudz_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^2
    with increasing levels of Chebyshev discretization.

    It computes the solution by solving on 4 separate nodes, and merging DtN maps.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """
    corners = jnp.array(
        [[-jnp.pi / 2, -jnp.pi / 2, -jnp.pi / 2], [jnp.pi / 2, jnp.pi / 2, jnp.pi / 2]]
    )

    error_vals = jnp.ones_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        q = p - 2
        l = 1
        root = Node(
            xmin=-jnp.pi / 2,
            xmax=jnp.pi / 2,
            ymin=-jnp.pi / 2,
            ymax=jnp.pi / 2,
            zmin=-jnp.pi / 2,
            zmax=jnp.pi / 2,
            depth=0,
        )

        t = create_solver_obj_3D(p=p, q=q, root=root, uniform_levels=l)

        cheby_pts = t.leaf_cheby_points

        # Assemble differential operator coeffs for the Laplacian
        d_xx_coeffs = d_xx_coeff_fn(cheby_pts)
        d_yy_coeffs = d_yy_coeff_fn(cheby_pts)
        d_zz_coeffs = d_zz_coeff_fn(cheby_pts)
        source = source_fn(cheby_pts)

        # Local solve stage

        local_solve_stage(
            t,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            D_zz_coeffs=d_zz_coeffs,
        )

        build_stage(t)

        root_bdry_points = t.root_boundary_points
        bdry_data_lst = get_bdry_data_evals_lst_3D(t, dirichlet_data_fn)

        if check_dtn:
            # Evaluate the accuracy of the DtN map
            expected_soln = jnp.concatenate(
                [
                    -1 * dudx_fn(root_bdry_points[: t.root.n_0]),  # face 0
                    dudx_fn(
                        root_bdry_points[t.root.n_0 : t.root.n_0 + t.root.n_1]
                    ),  # face 1
                    -1
                    * dudy_fn(
                        root_bdry_points[
                            t.root.n_0
                            + t.root.n_1 : t.root.n_0
                            + t.root.n_1
                            + t.root.n_2
                        ]
                    ),  # face 2
                    dudy_fn(
                        root_bdry_points[
                            t.root.n_0
                            + t.root.n_1
                            + t.root.n_2 : t.root.n_0
                            + t.root.n_1
                            + t.root.n_2
                            + t.root.n_3
                        ]
                    ),  # face 3
                    -1
                    * dudz_fn(
                        root_bdry_points[
                            t.root.n_0
                            + t.root.n_1
                            + t.root.n_2
                            + t.root.n_3 : t.root.n_0
                            + t.root.n_1
                            + t.root.n_2
                            + t.root.n_3
                            + t.root.n_4
                        ]
                    ),  # face 4
                    dudz_fn(root_bdry_points[-t.root.n_5 :]),  # face 5
                ]
            )
            map_DtN = t.root.DtN
            if map_DtN is None:
                map_DtN = t.interior_node_DtN_maps[-1]

            computed_soln_values = map_DtN @ jnp.concatenate(bdry_data_lst)

            # Plot the expected and computed solutions on face i=0
            # expected_i = expected_soln[: t.root.n_0]
            # computed_i = computed_soln_values[: t.root.n_0]
            # pts_i = t.root_boundary_points[: t.root.n_0, :2]
            # # SW, SE, NE, NW
            # corners_i = jnp.array(
            #     [
            #         [t.root.xmin, t.root.ymin],
            #         [t.root.xmax, t.root.ymin],
            #         [t.root.xmax, t.root.ymax],
            #         [t.root.xmin, t.root.ymax],
            #     ]
            # )
            # plot_soln_from_cheby_nodes(
            #     cheby_nodes=pts_i,
            #     corners=corners_i,
            #     computed_soln=computed_i,
            #     expected_soln=expected_i,
            # )

            # plt.plot(expected_soln, "o-", label="Expected")
            # plt.plot(computed_soln_values, "x-", label="Computed")
            # plt.legend()
            # plt.show()

        else:

            down_pass(t, boundary_data_lst=bdry_data_lst)

            all_cheby_points = jnp.reshape(t.leaf_cheby_points, (-1, 3))
            computed_soln_values = jnp.reshape(t.interior_solns, (-1))

            #######################
            # If we want to check solutions on particular voxels, we can do so here.
            # all_cheby_points = t.leaf_cheby_points[1]
            # computed_soln_values = t.interior_solns[1]
            # logging.debug(
            #     "check_l_inf_error_convergence_oct_merge: all_cheby_points = %s",
            #     all_cheby_points.shape,
            # )
            # logging.debug(
            #     "check_l_inf_error_convergence_oct_merge: computed_soln_values = %s",
            #     computed_soln_values.shape,
            # )

            expected_soln = soln_fn(all_cheby_points)

        l_inf_error = jnp.max(jnp.abs(computed_soln_values - expected_soln))
        error_vals = error_vals.at[i].set(l_inf_error)
        logging.debug(
            "check_l_inf_error_convergence_oct_merge: p=%i, l_inf_error = %s",
            p,
            l_inf_error,
        )

    logging.info("check_l_inf_error_convergence_oct_merge: error_vals = %s", error_vals)
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


def single_merge_check_3D_0(plot_fp: str) -> None:
    """Poisson BVP with constant coefficients and polynomial Dirichlet data."""
    logging.info("Running single_merge_check_3D_0")
    p_values = jnp.array([4, 6, 8, 10])
    check_l_inf_error_convergence_oct_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_POLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_POLY[K_SOURCE],
        soln_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        check_dtn=False,
        dudx_fn=TEST_CASE_POISSON_POLY[K_DU_DX],
        dudy_fn=TEST_CASE_POISSON_POLY[K_DU_DY],
        dudz_fn=TEST_CASE_POISSON_POLY[K_DU_DZ],
    )


def single_merge_check_3D_1(plot_fp: str) -> None:
    """Poisson BVP with constant coefficients and polynomial Dirichlet data. Checks DtN map."""
    logging.info("Running single_merge_check_3D_1")
    p_values = jnp.array([4, 6, 8])
    check_l_inf_error_convergence_oct_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_POLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_POLY[K_SOURCE],
        soln_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        check_dtn=True,
        dudx_fn=TEST_CASE_POISSON_POLY[K_DU_DX],
        dudy_fn=TEST_CASE_POISSON_POLY[K_DU_DY],
        dudz_fn=TEST_CASE_POISSON_POLY[K_DU_DZ],
    )


def single_merge_check_3D_2(plot_fp: str) -> None:
    """Poisson BVP with constant coefficients and non-polynomial Dirichlet data."""
    logging.info("Running single_merge_check_3D_2")
    p_values = jnp.array([4, 6, 8, 10, 12])
    check_l_inf_error_convergence_oct_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_NONPOLY[K_SOURCE],
        soln_fn=TEST_CASE_POISSON_NONPOLY[K_DIRICHLET],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        check_dtn=False,
        dudx_fn=TEST_CASE_POISSON_NONPOLY[K_DU_DX],
        dudy_fn=TEST_CASE_POISSON_NONPOLY[K_DU_DY],
        dudz_fn=TEST_CASE_POISSON_NONPOLY[K_DU_DZ],
    )


def single_merge_check_3D_3(plot_fp: str) -> None:
    """Poisson BVP with constant coefficients and non-polynomial Dirichlet data. Checks DtN map."""
    logging.info("Running single_merge_check_3D_3")
    p_values = jnp.array([4, 6, 8, 10, 12])
    check_l_inf_error_convergence_oct_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_NONPOLY[K_ZZ_COEFF],
        soln_fn=TEST_CASE_POISSON_NONPOLY[K_DIRICHLET],
        source_fn=TEST_CASE_POISSON_NONPOLY[K_SOURCE],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        check_dtn=True,
        dudx_fn=TEST_CASE_POISSON_NONPOLY[K_DU_DX],
        dudy_fn=TEST_CASE_POISSON_NONPOLY[K_DU_DY],
        dudz_fn=TEST_CASE_POISSON_NONPOLY[K_DU_DZ],
    )


def single_merge_check_3D_4(plot_fp: str) -> None:
    """Laplace problem with polynomial source and Dirichlet data."""
    logging.info("Running single_merge_check_3D_4")
    soln_fn = lambda x: TEST_CASE_HOMOG_PART_POLY[K_DIRICHLET](
        x
    ) + TEST_CASE_HOMOG_PART_POLY[K_PART_SOLN](x)
    p_values = jnp.array([4, 6, 8])
    check_l_inf_error_convergence_oct_merge(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_HOMOG_PART_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_HOMOG_PART_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_HOMOG_PART_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_HOMOG_PART_POLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_HOMOG_PART_POLY[K_SOURCE],
        soln_fn=soln_fn,
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        check_dtn=False,
    )
