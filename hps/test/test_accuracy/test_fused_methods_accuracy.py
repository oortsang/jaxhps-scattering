# from __future__ import annotations
import logging
from typing import Dict
import jax.numpy as jnp
import pytest

from hps.src.solver_obj import (
    SolverObj,
    create_solver_obj_2D,
)
from hps.src.quadrature.trees import Node
from hps.src.up_down_passes import (
    fused_pde_solve_2D,
    fused_pde_solve_2D_ItI,
    baseline_pde_solve_2D,
)
from hps.test.test_accuracy.cases import (
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    ETA,
    TEST_CASE_POLY_PART_HOMOG,
    TEST_CASE_POLY_ZERO_SOURCE,
    K_DIRICHLET,
    K_XX_COEFF,
    K_YY_COEFF,
    K_SOURCE,
    K_DIRICHLET_DUDX,
    K_DIRICHLET_DUDY,
    K_PART_SOLN,
    K_HOMOG_SOLN,
)


ATOL = 1e-12
RTOL = 0.0

ROOT_DTN = Node(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
ROOT_ITI = Node(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)


# Using p=7 to trigger a code path which uses the recomputation strategies earlier than necessary.
SOLVER_DTN = create_solver_obj_2D(
    p=7, q=5, root=ROOT_DTN, uniform_levels=3, fill_tree=False
)

SOLVER_ITI = create_solver_obj_2D(
    p=7, q=5, root=ROOT_ITI, uniform_levels=3, use_ItI=True, eta=ETA, fill_tree=False
)


def check_fused_accuracy_2D_DtN_uniform(solver: SolverObj, test_case: Dict) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](solver.leaf_cheby_points)
    d_yy_coeffs = test_case[K_YY_COEFF](solver.leaf_cheby_points)
    source = test_case[K_SOURCE](solver.leaf_cheby_points).squeeze(2)
    boundary_g = test_case[K_DIRICHLET](solver.root_boundary_points)
    logging.debug(
        "check_fused_accuracy_2D_DtN_uniform: boundary_g shape: %s", boundary_g.shape
    )
    fused_pde_solve_2D(
        t=solver,
        boundary_data=boundary_g,
        source_term=source,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
    )
    computed_soln = solver.interior_solns

    homog_soln = test_case[K_HOMOG_SOLN](solver.leaf_cheby_points)
    part_soln = test_case[K_PART_SOLN](solver.leaf_cheby_points).reshape(
        homog_soln.shape
    )
    logging.debug(
        "check_fused_accuracy_2D_DtN_uniform: part_soln = %s", part_soln.shape
    )
    logging.debug(
        "check_fused_accuracy_2D_DtN_uniform: homog_soln = %s", homog_soln.shape
    )
    logging.debug(
        "check_fused_accuracy_2D_DtN_uniform: computed_soln = %s", computed_soln.shape
    )
    expected_soln = homog_soln + part_soln
    assert jnp.allclose(computed_soln, expected_soln, atol=ATOL, rtol=RTOL)


def check_baseline_accuracy_2D_DtN_uniform(solver: SolverObj, test_case: Dict) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](solver.leaf_cheby_points)
    d_yy_coeffs = test_case[K_YY_COEFF](solver.leaf_cheby_points)
    source = test_case[K_SOURCE](solver.leaf_cheby_points).squeeze(2)
    boundary_g = test_case[K_DIRICHLET](solver.root_boundary_points)
    baseline_pde_solve_2D(
        t=solver,
        boundary_data=boundary_g,
        source_term=source,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
    )
    computed_soln = solver.interior_solns

    homog_soln = test_case[K_HOMOG_SOLN](solver.leaf_cheby_points)
    part_soln = test_case[K_PART_SOLN](solver.leaf_cheby_points).reshape(
        homog_soln.shape
    )
    logging.debug(
        "check_baseline_accuracy_2D_DtN_uniform: part_soln = %s", part_soln.shape
    )
    logging.debug(
        "check_baseline_accuracy_2D_DtN_uniform: homog_soln = %s", homog_soln.shape
    )
    expected_soln = homog_soln + part_soln
    assert jnp.allclose(computed_soln, expected_soln, atol=ATOL, rtol=RTOL)


def check_fused_accuracy_2D_ItI_uniform(solver: SolverObj, test_case: Dict) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](solver.leaf_cheby_points)
    d_yy_coeffs = test_case[K_YY_COEFF](solver.leaf_cheby_points)
    source = test_case[K_SOURCE](solver.leaf_cheby_points).squeeze(2)
    # Compute incoming impedance data
    n_per_side = solver.root_boundary_points.shape[0] // 4
    boundary_g_normals = jnp.concatenate(
        [
            -1 * test_case[K_DIRICHLET_DUDY](solver.root_boundary_points[:n_per_side]),
            test_case[K_DIRICHLET_DUDX](
                solver.root_boundary_points[n_per_side : 2 * n_per_side]
            ),
            test_case[K_DIRICHLET_DUDY](
                solver.root_boundary_points[2 * n_per_side : 3 * n_per_side]
            ),
            -1
            * test_case[K_DIRICHLET_DUDX](
                solver.root_boundary_points[3 * n_per_side :]
            ),
        ]
    )
    boundary_g_evals = test_case[K_DIRICHLET](solver.root_boundary_points)
    incoming_imp_data = boundary_g_normals + 1j * solver.eta * boundary_g_evals
    fused_pde_solve_2D_ItI(
        t=solver,
        boundary_data=incoming_imp_data,
        source_term=source,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
    )
    computed_soln = solver.interior_solns

    homog_soln = test_case[K_HOMOG_SOLN](solver.leaf_cheby_points)
    part_soln = test_case[K_PART_SOLN](solver.leaf_cheby_points).reshape(
        homog_soln.shape
    )
    logging.debug(
        "check_fused_accuracy_2D_ItI_uniform: part_soln = %s", part_soln.shape
    )
    logging.debug(
        "check_fused_accuracy_2D_ItI_uniform: homog_soln = %s", homog_soln.shape
    )
    logging.debug(
        "check_fused_accuracy_2D_ItI_uniform: computed_soln = %s", computed_soln.shape
    )
    expected_soln = homog_soln + part_soln
    assert jnp.allclose(computed_soln, expected_soln, atol=ATOL, rtol=RTOL)


class Test_accuracy_fused_2D_DtN_uniform:
    def test_0(self, caplog) -> None:
        """Polynomial data with zero source term."""
        caplog.set_level(logging.DEBUG)
        check_fused_accuracy_2D_DtN_uniform(SOLVER_DTN, TEST_CASE_POLY_ZERO_SOURCE)

    def test_1(self, caplog) -> None:
        """Polynomial data with non-zero source term."""
        caplog.set_level(logging.DEBUG)
        check_fused_accuracy_2D_DtN_uniform(SOLVER_DTN, TEST_CASE_POLY_PART_HOMOG)


class Test_accuracy_baseline_2D_DtN_uniform:
    def test_0(self, caplog) -> None:
        """Polynomial data with zero source term."""
        caplog.set_level(logging.DEBUG)
        check_baseline_accuracy_2D_DtN_uniform(SOLVER_DTN, TEST_CASE_POLY_ZERO_SOURCE)

    def test_1(self, caplog) -> None:
        """Polynomial data with non-zero source term."""
        caplog.set_level(logging.DEBUG)
        check_baseline_accuracy_2D_DtN_uniform(SOLVER_DTN, TEST_CASE_POLY_PART_HOMOG)


class Test_accuracy_fused_2D_ItI_uniform:
    def test_0(self, caplog) -> None:
        """Polynomial data with zero source term."""
        caplog.set_level(logging.DEBUG)

        check_fused_accuracy_2D_ItI_uniform(SOLVER_ITI, TEST_CASE_POLY_ZERO_SOURCE)


if __name__ == "__main__":
    pytest.main()
