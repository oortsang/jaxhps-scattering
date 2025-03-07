"""This file has a collection of checks that are meant to make sure the 
non-uniform merging and down pass are working correctly.
The checks are designed to run against problems with polynomial data and solutions,
so the convergence will be obvious once the choice of polynomial basis is correct.
"""

import logging
from typing import Tuple, Callable
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from hps.accuracy_checks.test_cases_3D import (
    TEST_CASE_POISSON_POLY,
    K_DU_DY,
    K_DU_DX,
    K_DU_DZ,
    K_SOURCE,
    K_YY_COEFF,
    K_XX_COEFF,
    K_ZZ_COEFF,
    K_DIRICHLET,
    adaptive_meshing_data_fn,
    d_x_adaptive_meshing_data_fn,
    d_y_adaptive_meshing_data_fn,
    d_z_adaptive_meshing_data_fn,
    d_xx_adaptive_meshing_data_fn,
    d_yy_adaptive_meshing_data_fn,
    d_zz_adaptive_meshing_data_fn,
)
from hps.src.up_down_passes import build_stage, down_pass, local_solve_stage
from hps.src.solver_obj import (
    SolverObj,
    create_solver_obj_3D,
    get_bdry_data_evals_lst_3D,
)
from hps.src.quadrature.trees import Node, add_eight_children
from hps.src.quadrature.quad_3D.interpolation import refinement_operator
from hps.src.quadrature.quad_3D.adaptive_meshing import (
    generate_adaptive_mesh_level_restriction,
)
from hps.accuracy_checks.utils import plot_soln_from_cheby_nodes


def adaptive_meshing_data_fn_source(x: jnp.array) -> jnp.array:
    lap_u = (
        d_xx_adaptive_meshing_data_fn(x)
        + d_yy_adaptive_meshing_data_fn(x)
        + d_zz_adaptive_meshing_data_fn(x)
    )
    return lap_u


def setup_tree_wavefront_example(p: int, tol: float) -> SolverObj:
    root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0)
    interp = refinement_operator(p)
    generate_adaptive_mesh_level_restriction(
        root=root,
        refinement_op=interp,
        f_fn=adaptive_meshing_data_fn_source,
        tol=tol,
        p=p,
        q=p - 2,
    )
    t = create_solver_obj_3D(p=p, q=p - 2, root=root)
    return t


def setup_tree_nonuniform_3D(p: int, max_depth: int, child_idx: int):
    # Set up the domain
    south = -jnp.pi / 2
    north = jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2
    root = Node(
        xmin=west, xmax=east, ymin=south, ymax=north, depth=0, zmin=south, zmax=north
    )
    # corners = [(west, south), (east, south), (east, north), (west, north)]
    q = p - 2

    add_to = root

    # Set up the non-uniform tree by recursively adding children to the
    # same child idx. The resulting tree will have a depth of max_depth and
    # will always satisfy the level restriction condition.
    for _ in range(max_depth):
        add_eight_children(add_to=add_to, root=root, q=q)
        add_to = add_to.children[child_idx]

    t = create_solver_obj_3D(p=p, q=q, root=root)
    return t


def check_l_inf_error_convergence_nonuniform_merge_3D(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_zz_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    plot_fp: str,
    max_depth: int,
    check_dtn: bool = False,
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudz_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
) -> None:

    child_idxes = [0, 1, 2, 3, 4, 5, 6, 7]

    n_p_vals = p_values.shape[0]

    error_vals = jnp.full((8, n_p_vals), jnp.nan, dtype=jnp.float64)

    # Loop over the child_idxes to check the convergence of the error
    for child_idx in child_idxes:

        for i, p in enumerate(p_values):
            p = int(p)
            t = setup_tree_nonuniform_3D(p=p, max_depth=max_depth, child_idx=child_idx)

            # Assemble differential operator coeffs for the Laplacian
            d_xx_coeffs = d_xx_coeff_fn(t.leaf_cheby_points)
            d_yy_coeffs = d_yy_coeff_fn(t.leaf_cheby_points)
            d_zz_coeffs = d_zz_coeff_fn(t.leaf_cheby_points)
            source = source_fn(t.leaf_cheby_points)

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
            boundary_data_lst = get_bdry_data_evals_lst_3D(t, dirichlet_data_fn)

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

                computed_soln_values = map_DtN @ jnp.concatenate(boundary_data_lst)

                #########################################
                # Use this code to check faces individually. Otherwise, comment out.
                # computed_soln_values = computed_soln_values[: t.root.n_0]
                # expected_soln = expected_soln[: t.root.n_0]

                ########################################
                # Use this code to plot the solution on face 0. Otherwise, comment out.
                # if p > 4:
                #     #     # Plot the expected and computed solutions on face i=0
                #     expected_i = expected_soln[: t.root.n_0]
                #     computed_i = computed_soln_values[: t.root.n_0]
                #     pts_i = t.root_boundary_points[: t.root.n_0, 1:]
                #     # SW, SE, NE, NW
                #     corners_i = jnp.array(
                #         [
                #             [t.root.xmin, t.root.ymin],
                #             [t.root.xmax, t.root.ymin],
                #             [t.root.xmax, t.root.ymax],
                #             [t.root.xmin, t.root.ymax],
                #         ]
                #     )

                #     plot_soln_from_cheby_nodes(
                #         cheby_nodes=pts_i,
                #         computed_soln=computed_i,
                #         expected_soln=expected_i,
                #         corners=None,
                #     )

                # plt.plot(expected_soln, "-o", label="Expected")
                # plt.plot(computed_soln_values, "-x", label="Computed")
                # plt.legend()
                # plt.grid()
                # plt.show()

            else:

                down_pass(t, boundary_data_lst)

                all_cheby_points = jnp.reshape(t.leaf_cheby_points, (-1, 3))
                computed_soln_values = jnp.reshape(t.interior_solns, (-1))

                #########################################
                # Use this code to check faces individually. Otherwise, comment out.
                # all_cheby_points = t.leaf_cheby_points[0]
                # computed_soln_values = t.interior_solns[0]
                # expected_soln = expected_soln[: t.root.n_0]

                expected_soln = dirichlet_data_fn(all_cheby_points)

            l_inf_error = jnp.max(jnp.abs(computed_soln_values - expected_soln))
            error_vals = error_vals.at[child_idx, i].set(l_inf_error)

        logging.info(
            "check_l_inf_error_convergence_nonuniform_merge: child_idx: %i, error_vals = %s",
            child_idx,
            error_vals[child_idx],
        )
        # break
    # # Plot the convergence on a semi-log plot
    # if compute_convergence_rate:
    #     log_error_vals = jnp.log(error_vals)
    #     p_values = jnp.array(p_values, dtype=jnp.float64)
    #     m, b = jnp.polyfit(p_values, log_error_vals, 1)

    #     xvals = jnp.linspace(p_values[0], p_values[-1], 100)
    #     yvals = np.exp(m * xvals + b)

    fig, ax = plt.subplots()
    if check_dtn:
        t = "Error convergence for DtN map"
    else:
        t = "Error convergence for Poisson BVP"
    ax.set_title(t)
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    for child_idx in child_idxes:
        ax.plot(p_values, error_vals[child_idx], "o-", label=f"Child idx = {child_idx}")
    # ax.plot(p_values, error_vals, "o-")
    ax.set_yscale("log")
    ax.grid()

    # if compute_convergence_rate:
    #     ax.plot(xvals, yvals, "--", label=f"Slope = {m:.2f}")
    #     ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def check_l_inf_error_convergence_wavefront_example_3D(
    tol_values: jnp.ndarray,
    p: int,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_zz_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    plot_fp: str,
    check_dtn: bool = False,
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudz_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
) -> None:
    """Use the grid from the wavefront example but with whatever data we want."""

    n_tol_vals = tol_values.shape[0]

    error_vals = jnp.full((n_tol_vals,), jnp.nan, dtype=jnp.float64)

    for i, tol in enumerate(tol_values):
        t = setup_tree_wavefront_example(p=p, tol=tol)

        # Assemble differential operator coeffs for the Laplacian
        d_xx_coeffs = d_xx_coeff_fn(t.leaf_cheby_points)
        d_yy_coeffs = d_yy_coeff_fn(t.leaf_cheby_points)
        d_zz_coeffs = d_zz_coeff_fn(t.leaf_cheby_points)
        source_evals = source_fn(t.leaf_cheby_points)

        # Local solve stage
        local_solve_stage(
            t,
            source_term=source_evals,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            D_zz_coeffs=d_zz_coeffs,
        )

        build_stage(t)

        root_bdry_points = t.root_boundary_points
        boundary_data_lst = get_bdry_data_evals_lst_3D(t, dirichlet_data_fn)

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

            computed_soln_values = map_DtN @ jnp.concatenate(boundary_data_lst)

            # Plot them.
            # fig, ax = plt.subplots()
            # ax.plot(expected_soln, "-o", label="Expected")
            # ax.plot(computed_soln_values, "-x", label="Computed")
            # ax.legend()
            # ax.grid()
            # plt.show()

        else:

            down_pass(t, boundary_data_lst)

            all_cheby_points = jnp.reshape(t.leaf_cheby_points, (-1, 3))
            computed_soln_values = jnp.reshape(t.interior_solns, (-1))

            expected_soln = dirichlet_data_fn(all_cheby_points)

        l_inf_error = jnp.max(jnp.abs(computed_soln_values - expected_soln))
        error_vals = error_vals.at[i].set(l_inf_error)

    logging.info(
        "check_l_inf_error_convergence_nonuniform_merge:  error_vals = %s",
        error_vals,
    )

    fig, ax = plt.subplots()
    if check_dtn:
        t = "Error convergence for DtN map"
    else:
        t = "Error convergence for Poisson BVP"
    ax.set_title(t)
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(tol_values, error_vals, "o-")
    # ax.plot(p_values, error_vals, "o-")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid()

    # if compute_convergence_rate:
    #     ax.plot(xvals, yvals, "--", label=f"Slope = {m:.2f}")
    #     ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def nonuniform_grid_check_3D_0(plot_fp: str) -> None:
    """
    Poisson BVP: f(x, y) = x^2 - y^2. Discretization is a depth 2 non-uniform grid.

    Checks the computed solution.
    """
    logging.info("Running nonuniform_grid_check_3D_0")
    p_values = jnp.array([4, 6, 8, 10])
    check_l_inf_error_convergence_nonuniform_merge_3D(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_POLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_POLY[K_SOURCE],
        plot_fp=plot_fp,
        max_depth=2,
    )


def nonuniform_grid_check_3D_1(plot_fp: str) -> None:
    """
    Poisson BVP: f(x, y) = x^2 - y^2. Discretization is a depth 2 non-uniform grid.

    Checks the DtN map.
    """
    logging.info("Running nonuniform_grid_check_3D_1")
    p_values = jnp.array([4, 6, 8, 10])

    check_l_inf_error_convergence_nonuniform_merge_3D(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_POLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_POLY[K_SOURCE],
        dudx_fn=TEST_CASE_POISSON_POLY[K_DU_DX],
        dudy_fn=TEST_CASE_POISSON_POLY[K_DU_DY],
        dudz_fn=TEST_CASE_POISSON_POLY[K_DU_DZ],
        plot_fp=plot_fp,
        check_dtn=True,
        max_depth=2,
    )


def nonuniform_grid_check_3D_2(plot_fp: str) -> None:
    """
    Uses the nonuniform grid generated by the 3D wave front example

    Data is a polynomial Poisson problem.

    Checks the computed solution.
    """
    logging.info("Running nonuniform_grid_check_3D_2")
    tol_values = jnp.array([1e-01, 1e-2, 1e-3])
    check_l_inf_error_convergence_wavefront_example_3D(
        tol_values=tol_values,
        p=8,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_POLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_POLY[K_SOURCE],
        dudx_fn=TEST_CASE_POISSON_POLY[K_DU_DX],
        dudy_fn=TEST_CASE_POISSON_POLY[K_DU_DY],
        dudz_fn=TEST_CASE_POISSON_POLY[K_DU_DZ],
        plot_fp=plot_fp,
    )


def nonuniform_grid_check_3D_3(plot_fp: str) -> None:
    """
    Uses the nonuniform grid generated by the 3D wave front example

    Data is a polynomial Poisson problem.

    Checks the DtN map.
    """
    logging.info("Running nonuniform_grid_check_3D_3")
    tol_values = jnp.array([1e-01, 1e-2, 1e-3])
    check_l_inf_error_convergence_wavefront_example_3D(
        tol_values=tol_values,
        p=8,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY[K_DIRICHLET],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY[K_XX_COEFF],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY[K_YY_COEFF],
        d_zz_coeff_fn=TEST_CASE_POISSON_POLY[K_ZZ_COEFF],
        source_fn=TEST_CASE_POISSON_POLY[K_SOURCE],
        dudx_fn=TEST_CASE_POISSON_POLY[K_DU_DX],
        dudy_fn=TEST_CASE_POISSON_POLY[K_DU_DY],
        dudz_fn=TEST_CASE_POISSON_POLY[K_DU_DZ],
        plot_fp=plot_fp,
        check_dtn=True,
    )
