from typing import Callable
import jax.numpy as jnp

from hps.src.up_down_passes import build_stage, down_pass, local_solve_stage
from hps.src.solver_obj import create_solver_obj_2D
from hps.src.quadrature.trees import Node
from hps.src.solver_obj import create_solver_obj_3D
from hps.src.up_down_passes import (
    fused_pde_solve_2D,
    fused_pde_solve_2D_ItI,
)


def get_l_inf_error_2D(
    p: int,
    l: int,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_soln_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    I_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    use_fused_code: bool = True,
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

    q = p - 2

    # Set up the root of the domain
    root = Node(
        xmin=west,
        xmax=east,
        ymin=south,
        ymax=north,
        depth=0,
        zmin=None,
        zmax=None,
    )

    solver_obj = create_solver_obj_2D(p=p, q=q, root=root, uniform_levels=l)

    # Create coeffs for Laplacian
    d_xx_coeffs = d_xx_coeff_fn(solver_obj.leaf_cheby_points)
    d_yy_coeffs = d_yy_coeff_fn(solver_obj.leaf_cheby_points)
    if I_coeff_fn is not None:
        I_coeffs = I_coeff_fn(solver_obj.leaf_cheby_points)
    else:
        I_coeffs = None

    # Check this is an elliptic problem:
    assert jnp.all(d_xx_coeffs * d_yy_coeffs > 0)

    source = source_fn(solver_obj.leaf_cheby_points)

    if use_fused_code:
        # In this branch, we use the fused function call
        boundary_data = dirichlet_data_fn(solver_obj.root_boundary_points)
        fused_pde_solve_2D(
            solver_obj,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            I_coeffs=I_coeffs,
            boundary_data=boundary_data,
        )
    else:
        local_solve_stage(
            solver_obj,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )

        # Do the upward pass
        build_stage(solver_obj)

        # Define the boundary data
        boundary_points = solver_obj.root_boundary_points
        n_per_side = boundary_points.shape[0] // 4
        boundary_data_lst = [
            dirichlet_data_fn(boundary_points[:n_per_side]),
            dirichlet_data_fn(boundary_points[n_per_side : 2 * n_per_side]),
            dirichlet_data_fn(
                boundary_points[2 * n_per_side : 3 * n_per_side]
            ),
            dirichlet_data_fn(boundary_points[3 * n_per_side :]),
        ]

        # Do the downward pass
        down_pass(solver_obj, boundary_data_lst)

    all_cheby_pts = jnp.reshape(solver_obj.leaf_cheby_points, (-1, 2))
    all_soln_vals = jnp.reshape(solver_obj.interior_solns, (-1))
    expected_soln_vals = dirichlet_data_fn(all_cheby_pts).flatten()

    if particular_soln_fn is not None:
        expected_soln_vals = (
            expected_soln_vals + particular_soln_fn(all_cheby_pts).flatten()
        )

    return jnp.max(jnp.abs(all_soln_vals - expected_soln_vals)) / jnp.max(
        jnp.abs(expected_soln_vals)
    )


def get_l_inf_error_2D_ItI(
    p: int,
    l: int,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eta: float,
    particular_soln_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    I_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    return_expected_computed_tree: bool = False,
) -> float:
    """
    This function computes the L_inf error for a 2D problem with a known solution
    solved on a uniform mesh. The solution is constructed by merging ItI maps,
    so the problem tested is Helmholtz-like.

    Args:
        p (int): Chebyshev parameter
        l (int): Number of uniform refinement levels. There will be 4^l patches.
        dirichlet_data_fn (Callable[[jnp.ndarray], jnp.ndarray]): Prescribing the boundary data.
        d_xx_coeff_fn (Callable[[jnp.ndarray], jnp.ndarray]): Prescribing the coefficients for the Laplacian.
        d_yy_coeff_fn (Callable[[jnp.ndarray], jnp.ndarray]): Prescribing the coefficients for the Laplacian.
        source_fn (Callable[[jnp.ndarray], jnp.ndarray]): Prescribing the source term.
        dudx_fn (Callable[[jnp.ndarray], jnp.ndarray]): Prescribing the x-derivative of the boundary data. Used to compute the impedance.
        dudy_fn (Callable[[jnp.ndarray], jnp.ndarray]): Prescribing the y-derivative of the boundary data. Used to compute the impedance.
        eta (float): Parameter for impedance.
        particular_soln_fn (Callable[[jnp.ndarray], jnp.ndarray], optional): If there is a particular solution, this is it. Defaults to None.
        I_coeff_fn (Callable[[jnp.ndarray], jnp.ndarray], optional): Prescribing the coefficient on the u term. Defaults to None.

    Returns:
        float: L-inf error between computed and analytical solution.
    """

    # Set up the domain
    south = -jnp.pi / 2
    north = jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2
    corners = [(west, south), (east, south), (east, north), (west, north)]

    q = p - 2

    # Set up the Tree
    tree = create_solver_obj_2D(p, q, l, corners, eta=eta, use_ItI=True)

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

    # Assemble incoming impedance data on the boundary
    boundary_points = tree.root_boundary_points
    n_per_side = tree.root_boundary_points.shape[0] // 4
    boundary_f = dirichlet_data_fn(boundary_points)
    boundary_normals = jnp.concatenate(
        [
            -1 * dudy_fn(boundary_points[:n_per_side]),
            dudx_fn(boundary_points[n_per_side : 2 * n_per_side]),
            dudy_fn(boundary_points[2 * n_per_side : 3 * n_per_side]),
            -1 * dudx_fn(boundary_points[3 * n_per_side :]),
        ]
    )
    # This is incoming impedance data
    boundary_data = boundary_normals + 1j * eta * boundary_f

    # Evaluate the solution at all of the Chebyshev points
    # down_pass(tree, boundary_data)
    fused_pde_solve_2D_ItI(
        tree,
        source_term=source,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
        I_coeffs=I_coeffs,
        boundary_data=boundary_data,
    )

    # Reshape the quadrature points and the computed solution arrays
    all_cheby_pts = jnp.reshape(tree.leaf_cheby_points, (-1, 2))
    all_soln_vals = jnp.reshape(tree.interior_solns, (-1))

    expected_soln_vals = dirichlet_data_fn(all_cheby_pts).flatten()

    if particular_soln_fn is not None:
        part_soln_vals = particular_soln_fn(all_cheby_pts).flatten()
    else:
        part_soln_vals = 0
    expected_soln_vals = expected_soln_vals + part_soln_vals

    if return_expected_computed_tree:
        e = dirichlet_data_fn(tree.leaf_cheby_points)
        if particular_soln_fn is not None:
            e = e + particular_soln_fn(tree.leaf_cheby_points)
        return (e, tree.interior_solns, tree)
    else:
        return jnp.max(jnp.abs(all_soln_vals - expected_soln_vals)) / jnp.max(
            jnp.abs(expected_soln_vals.flatten())
        )


def get_l_inf_error_3D(
    p: int,
    l: int,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_zz_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
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
    xmax = jnp.pi / 2
    xmin = -jnp.pi / 2
    ymax = jnp.pi / 2
    ymin = -jnp.pi / 2
    zmax = jnp.pi / 2
    zmin = -jnp.pi / 2
    corners = jnp.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])

    q = p - 2

    # Set up the Tree
    tree = create_solver_obj_3D(p, q, l, corners)

    # Create coeffs for Laplacian
    d_xx_coeffs = d_xx_coeff_fn(tree.leaf_cheby_points)
    d_yy_coeffs = d_yy_coeff_fn(tree.leaf_cheby_points)
    d_zz_coeffs = d_zz_coeff_fn(tree.leaf_cheby_points)

    # Check this is an elliptic problem:
    assert jnp.all(d_xx_coeffs * d_yy_coeffs > 0)

    source = source_fn(tree.leaf_cheby_points)
    local_solve_stage(
        tree,
        source_term=source,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
        D_zz_coeffs=d_zz_coeffs,
    )

    # Do the upward pass
    build_stage(tree)

    # Define the boundary data
    boundary_points = tree.root_boundary_points
    boundary_data = dirichlet_data_fn(boundary_points)

    # Evaluate the solution at all of the Chebyshev points
    # Do the downward pass
    down_pass(tree, boundary_data)

    all_cheby_pts = jnp.reshape(tree.leaf_cheby_points, (-1, 3))
    all_soln_vals = jnp.reshape(tree.interior_solns, (-1))

    expected_soln_vals = dirichlet_data_fn(all_cheby_pts).flatten()

    if particular_soln_fn is not None:
        expected_soln_vals = (
            expected_soln_vals + particular_soln_fn(all_cheby_pts).flatten()
        )

    # print("get_l_inf_error_3D: all_soln_vals.shape", all_soln_vals.shape)
    # print("get_l_inf_error_3D: expected_soln_vals.shape", expected_soln_vals.shape)
    return jnp.max(jnp.abs(all_soln_vals - expected_soln_vals)) / jnp.max(
        jnp.abs(expected_soln_vals)
    )
