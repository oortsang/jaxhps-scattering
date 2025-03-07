from functools import partial
from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax

from hps.src.quadrature.quadrature_utils import (
    chebyshev_points,
    barycentric_lagrange_2d_interpolation_matrix,
    barycentric_lagrange_3d_interpolation_matrix,
    affine_transform,
)
from hps.src.quadrature.quad_3D.indexing import (
    rearrange_indices_ext_int,
    get_face_1_idxes,
    get_face_2_idxes,
    get_face_3_idxes,
    get_face_4_idxes,
    get_face_5_idxes,
    get_face_6_idxes,
    indexing_for_refinement_operator,
)
from hps.src.quadrature.quad_3D.differentiation import precompute_diff_operators
from hps.src.quadrature.trees import Node, get_all_leaves, add_four_children
from hps.src.quadrature.quad_3D.grid_creation import (
    get_all_boundary_gauss_legendre_points,
    get_all_boundary_gauss_legendre_points_uniform_refinement,
)


def precompute_refining_coarsening_ops(q: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    The refining operator maps from a Gauss-Legendre grid to four Gauss-Legendre grids on
    the same domain.

    For reference in this section, imagine the four panesl are laid out like this:

    --------
    | D | C |
    --------
    | A | B |
    --------

    The coarsening operator maps from four Gauss-Legendre grids to a single Gauss-Legendre grid.
    """

    if q % 2:
        raise ValueError("q must be even.")

    gauss_pts = np.polynomial.legendre.leggauss(q)[0]

    idxes = jnp.arange(q**2)

    first_half = affine_transform(gauss_pts, jnp.array([-1, 0]))
    second_half = affine_transform(gauss_pts, jnp.array([0, 1]))

    # Refining operator

    to_A = barycentric_lagrange_2d_interpolation_matrix(
        gauss_pts, gauss_pts, first_half, first_half
    )
    to_B = barycentric_lagrange_2d_interpolation_matrix(
        gauss_pts, gauss_pts, second_half, first_half
    )
    to_C = barycentric_lagrange_2d_interpolation_matrix(
        gauss_pts, gauss_pts, second_half, second_half
    )
    to_D = barycentric_lagrange_2d_interpolation_matrix(
        gauss_pts, gauss_pts, first_half, second_half
    )
    refining_operator = jnp.concatenate([to_A, to_B, to_C, to_D], axis=0)

    # Coarsening operator
    coarsening_op_out = jnp.zeros((q**2, 4 * q**2), dtype=jnp.float64)

    gauss_pts_first_half = gauss_pts[: q // 2]
    gauss_pts_second_half = gauss_pts[q // 2 :]

    from_A = barycentric_lagrange_2d_interpolation_matrix(
        first_half, first_half, gauss_pts_first_half, gauss_pts_first_half
    )
    A_bools = idxes % q < q / 2
    A_bools = jnp.logical_and(A_bools, idxes < q**2 / 2)
    coarsening_op_out = coarsening_op_out.at[A_bools, 0 : q**2].set(from_A)

    from_B = barycentric_lagrange_2d_interpolation_matrix(
        second_half, first_half, gauss_pts_second_half, gauss_pts_first_half
    )
    B_bools = idxes % q < q / 2
    B_bools = jnp.logical_and(B_bools, idxes >= q**2 / 2)
    coarsening_op_out = coarsening_op_out.at[B_bools, q**2 : 2 * (q**2)].set(from_B)

    from_C = barycentric_lagrange_2d_interpolation_matrix(
        second_half, second_half, gauss_pts_second_half, gauss_pts_second_half
    )
    C_bools = idxes % q >= q / 2
    C_bools = jnp.logical_and(C_bools, idxes >= q**2 / 2)
    coarsening_op_out = coarsening_op_out.at[C_bools, 2 * q**2 : 3 * (q**2)].set(from_C)

    from_D = barycentric_lagrange_2d_interpolation_matrix(
        first_half, second_half, gauss_pts_first_half, gauss_pts_second_half
    )
    D_bools = idxes % q >= q / 2
    D_bools = jnp.logical_and(D_bools, idxes < q**2 / 2)
    coarsening_op_out = coarsening_op_out.at[D_bools, 3 * q**2 :].set(from_D)

    return refining_operator, coarsening_op_out


def refinement_operator(p: int) -> jnp.array:
    cheby_pts_1d = chebyshev_points(p)[0]
    cheby_pts_refined = jnp.concatenate(
        [
            affine_transform(cheby_pts_1d, jnp.array([-1, 0])),
            affine_transform(cheby_pts_1d, jnp.array([0, 1])),
        ]
    )

    # This thing is scale and translation invariant so no need
    # to worry about recomputing it for different sizes
    I_refined = barycentric_lagrange_3d_interpolation_matrix(
        cheby_pts_1d,
        cheby_pts_1d,
        cheby_pts_1d,
        cheby_pts_refined,
        cheby_pts_refined,
        cheby_pts_refined,
    )

    r, c = indexing_for_refinement_operator(p)

    # I_refined has cols that are a standard meshgrid of (cheby_pts_1d, cheby_pts_1d, cheby_pts_1d).
    # We want to rearrange the columns so that the first p**3 - (p-2)**3 columns are the exterior points
    # and the rest are the interior points.
    I_refined = I_refined[:, c]

    # I_refined has rows that are a standard meshgrid of (cheby_pts_refined, cheby_pts_refined, cheby_pts_refined).
    # We first want to rearrange them so that each of the 8 blocks are together. Then we want to rearrange each
    # block so that the first p**3 - (p-2)**3 rows are the exterior points and the rest are the interior points.
    I_refined = I_refined[r, :]
    return I_refined


def precompute_P_matrix(p: int, q: int) -> jnp.ndarray:
    """Puts together an operator interpolating from the Gauss grid on the boundary to the Cheby points on the boundary of the 3D cube.

    Args:
        p (int): Number of Chebyshev points in 1D.
        q (int): Number of Gauss-Legendre points in 1D.

    Returns:
        jnp.ndarray: Has shape (p**3 - (p-2)**3, 6 * q**2)
    """
    gauss_pts = np.polynomial.legendre.leggauss(q)[0]
    cheby_pts = chebyshev_points(p)[0]

    n_cheby_bdry_pts = p**3 - (p - 2) ** 3

    P = barycentric_lagrange_2d_interpolation_matrix(
        gauss_pts, gauss_pts, cheby_pts, cheby_pts
    )

    I_P = jnp.zeros((n_cheby_bdry_pts, 6 * q**2), dtype=jnp.float64)

    # First face
    idxes = get_face_1_idxes(p)
    I_P = I_P.at[idxes, 0 : q**2].set(P)

    # Second face
    idxes = get_face_2_idxes(p)
    I_P = I_P.at[idxes, q**2 : 2 * q**2].set(P)

    # Third face
    idxes = get_face_3_idxes(p)
    I_P = I_P.at[idxes, 2 * q**2 : 3 * q**2].set(P)

    # Fourth face
    idxes = get_face_4_idxes(p)
    I_P = I_P.at[idxes, 3 * q**2 : 4 * q**2].set(P)

    # Fifth face
    idxes = get_face_5_idxes(p)
    I_P = I_P.at[idxes, 4 * q**2 : 5 * q**2].set(P)

    # Sixth face
    idxes = get_face_6_idxes(p)
    I_P = I_P.at[idxes, 5 * q**2 :].set(P)

    # Take averages of the corners of the cube. They occur at
    # indices 0, p-1, p**2 - p, p**2 - 1, p**2, p**2 + p - 1, 2 * p**2 - p, 2 * p**2 - 1.
    corner_idxes = jnp.array(
        [
            0,
            p - 1,
            p**2 - p,
            p**2 - 1,
            p**2,
            p**2 + p - 1,
            2 * p**2 - p,
            2 * p**2 - 1,
        ]
    )
    I_P = I_P.at[corner_idxes, :].set(I_P[corner_idxes, :] / 3)

    # Take averages of the edges of the cube. There are 12 edges. They are all of length p-2 because we have
    # already averaged the corners.
    edge_idxes = jnp.concatenate(
        [
            jnp.arange(1, p - 1),
            jnp.arange(p**2 + 1, p**2 + p - 1),
            jnp.arange(p**2 - p + 1, p**2 - 1),
            jnp.arange(2 * p**2 - p + 1, 2 * p**2 - 1),
            jnp.arange(1, p - 1) * p,
            jnp.arange(p, p**2 - p, p) + p**2,
            jnp.arange(2 * p**2, 3 * p**2 - 2 * p, p),
            jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p, p),
            jnp.arange(2 * p - 1, p**2 - p, p),
            jnp.arange(p**2 + 2 * p - 1, 2 * p**2 - p, p),
            jnp.arange(2 * p**2 + p - 1, 3 * p**2 - 2 * p, p),
            jnp.arange(3 * p**2 - 2 * p + p - 1, 4 * p**2 - 4 * p, p),
        ]
    )

    I_P = I_P.at[edge_idxes, :].set(I_P[edge_idxes, :] / 2)

    return I_P


def precompute_Q_D_matrix(
    p: int, q: int, du_dx: jnp.ndarray, du_dy: jnp.ndarray, du_dz: jnp.ndarray
) -> jnp.ndarray:
    """Precomputes an operator which maps a solution on the
    3D Chebyshev points to the outward normal derivative of the solution on the boundary Gauss grid.

    Args:
        p (int): Number of Chebyshev points in 1D.
        q (int): Number of Gauss-Legendre points in 1D.
        du_dx (jnp.ndarray): Precomputed derivative operator in the x-direction. Has shape (p**3, p**3).
        du_dy (jnp.ndarray): Precomputed derivative operator in the y-direction. Has shape (p**3, p**3).
        du_dz (jnp.ndarray): Precomputed derivative operator in the z-direction. Has shape (p**3, p**3).

    Returns:
        jnp.ndarray: Has shape (6 * q**2, p**3)
    """

    # First assemble D, a matrix of shape (6 p**2, p**3) which maps from a solution on all of the
    # Chebyshev points to the outward normal derivative of the solution on the boundary Chebyshev points.
    # Note this operator is double-counting
    # grid points on the edges.
    N = jnp.full((6 * p**2, p**3), np.nan, dtype=jnp.float64)
    # Grab the rows of the appropriate derivative operator and put them into N.
    face_1_idxes = get_face_1_idxes(p)
    N = N.at[: p**2].set(-1 * du_dx[face_1_idxes])
    face_2_idxes = get_face_2_idxes(p)
    N = N.at[p**2 : 2 * p**2].set(du_dx[face_2_idxes])
    face_3_idxes = get_face_3_idxes(p)
    N = N.at[2 * p**2 : 3 * p**2].set(-1 * du_dy[face_3_idxes])
    face_4_idxes = get_face_4_idxes(p)
    N = N.at[3 * p**2 : 4 * p**2].set(du_dy[face_4_idxes])
    face_5_idxes = get_face_5_idxes(p)
    N = N.at[4 * p**2 : 5 * p**2].set(-1 * du_dz[face_5_idxes])
    face_6_idxes = get_face_6_idxes(p)
    N = N.at[5 * p**2 :].set(du_dz[face_6_idxes])

    # Now assemble Q_D, a matrix of shape (6 * q**2, p**3) which maps from the Chebyshev points on the boundary to the Gauss points on the boundary.

    gauss_pts = np.polynomial.legendre.leggauss(q)[0]
    cheby_pts = chebyshev_points(p)[0]
    Q = barycentric_lagrange_2d_interpolation_matrix(
        cheby_pts, cheby_pts, gauss_pts, gauss_pts
    )
    Q_I = jnp.kron(jnp.eye(6), Q)
    # Q_I = jnp.zeros((6 * q**2, 6 * p**2), dtype=jnp.float64)
    # # Face 1
    # Q_I = Q_I.at[0 : q**2, 0 : p**2].set(jnp.flip(Q, axis=(0, 1)))
    # # Face 2
    # # Q_I = Q_I.at[q**2 : 2 * q**2].set(Q)
    # # # Face 3
    # # Q_I = Q_I.at[2 * q**2 : 3 * q**2].set(Q)
    # # # Face 4
    # # Q_I = Q_I.at[3 * q**2 : 4 * q**2].set(Q)
    # # # Face 5
    # # Q_I = Q_I.at[4 * q**2 : 5 * q**2].set(Q)
    # # # Face 6
    # # Q_I = Q_I.at[5 * q**2 :].set(Q)
    Q_D = Q_I @ N
    return Q_D


def interp_operator_to_uniform(
    root: Node,
    from_grid: jnp.array,
    to_x: jnp.array,
    to_y: jnp.array,
    to_z: jnp.array,
    p: int,
) -> jnp.array:
    """
    Given a non-uniform quadrature, specified by the tree root and the array of
    quadrature points from_grid, this function returns a matrix that interpolates
    the function values from the non-uniform grid to a uniform grid specified by
    the arrays to_x, to_y, to_z.

    The interpolation is local, i.e. the for a given point in the uniform grid,
    the interpolation matrix only interpolates from one of the leaves of the tree.
    There will be some edge cases where one of the output points is on the boundary
    of two or more leaves, in which case the output is the average of the interpolations
    from the leaves.

    Args:
        root (Node): Describes the non-uniform octree structure
        from_grid (jnp.array): Has shape (n_leaves, p**3, 3)
        to_x (jnp.array): Has shape (m,)
        to_y (jnp.array): Has shape (n,)
        to_z (jnp.array): Has shape (k,)
        p (int): Number of points per dimension in one leaf of the non-uniform grid

    Returns:
        jnp.array: Has shape (m*n*k, n_leaves*p**3)
    """

    m = to_x.shape[0]
    n = to_y.shape[0]
    k = to_z.shape[0]

    leaves_iter = get_all_leaves(root)
    n_leaves = len(leaves_iter)

    # Chebyshev panel with p points
    cheby_pts_1d = chebyshev_points(p)[0]

    # When we create the local interpolation matrices, we will need to
    # rearrange the columns because we order the points on each patch
    # to be exterior then interior, rather than the standard meshgrid
    # ordering.
    rearrange_idxes = rearrange_indices_ext_int(p)

    assert from_grid.shape[0] == n_leaves

    I = jnp.zeros((m * n * k, n_leaves * p**3), dtype=from_grid.dtype)

    # Make the output grid here
    X, Y, Z = jnp.meshgrid(to_x, to_y, to_z, indexing="ij")
    pts = jnp.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
    # print("interp_operator_to_uniform: pts.shape = ", pts.shape)

    # Loop through each point and compute the operator for that point
    for i in range(pts.shape[0]):

        leaves_containing_i = []
        to_x_i = pts[i, 0].reshape((1,))
        to_y_i = pts[i, 1].reshape((1,))
        to_z_i = pts[i, 2].reshape((1,))

        for leaf_idx, leaf in enumerate(leaves_iter):
            if _node_contains_point(leaf, pts[i]):
                leaves_containing_i.append(leaf_idx)

        n_leaves_containing = len(leaves_containing_i)

        for leaf_idx in leaves_containing_i:
            leaf = leaves_iter[leaf_idx]

            # Get the 3D Cheby panel for this leaf
            from_x = affine_transform(cheby_pts_1d, jnp.array([leaf.xmin, leaf.xmax]))
            from_y = affine_transform(cheby_pts_1d, jnp.array([leaf.ymin, leaf.ymax]))
            from_z = affine_transform(cheby_pts_1d, jnp.array([leaf.zmin, leaf.zmax]))

            I_local = barycentric_lagrange_3d_interpolation_matrix(
                from_x, from_y, from_z, to_x_i, to_y_i, to_z_i
            )

            # Rearrange the columns
            I_local = I_local[:, rearrange_idxes]

            # Put the local interpolation matrix into the global interpolation matrix
            I = I.at[i : i + 1, leaf_idx * p**3 : (leaf_idx + 1) * p**3].set(
                I_local / n_leaves_containing
                + I[i : i + 1, leaf_idx * p**3 : (leaf_idx + 1) * p**3]
            )

    return I


def _node_contains_point(node: Node, pt: jnp.ndarray) -> bool:
    """Returns True if the node contains the point pt, False otherwise."""
    return (
        node.xmin <= pt[0] <= node.xmax
        and node.ymin <= pt[1] <= node.ymax
        and node.zmin <= pt[2] <= node.zmax
    )


def interp_from_nonuniform_hps_to_uniform_grid(
    root: Node,
    p: int,
    f_evals: jnp.array,
    to_x: jnp.array,
    to_y: jnp.array,
    to_z: jnp.array,
) -> Tuple[jnp.array]:
    xmin = root.xmin
    xmax = root.xmax
    ymin = root.ymin
    ymax = root.ymax
    zmin = root.zmin
    zmax = root.zmax

    X, Y, Z = jnp.meshgrid(to_x, to_y, to_z)

    pts = jnp.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
    all_leaves = get_all_leaves(root)
    corners_lst = [
        jnp.array(
            [
                [node.xmin, node.ymin, node.zmin],
                [node.xmax, node.ymax, node.zmax],
            ]
        )
        for node in all_leaves
    ]
    corners_iter = jnp.stack(corners_lst)

    satisfies_xmin = pts[:, 0, None] >= corners_iter[None, :, 0, 0]
    satisfies_xmax = pts[:, 0, None] <= corners_iter[None, :, 1, 0]
    satisfies_ymin = pts[:, 1, None] >= corners_iter[None, :, 0, 1]
    satisfies_ymax = pts[:, 1, None] <= corners_iter[None, :, 1, 1]
    satisfies_zmin = pts[:, 2, None] >= corners_iter[None, :, 0, 2]
    satisfies_zmax = pts[:, 2, None] <= corners_iter[None, :, 1, 2]

    x_bools = jnp.logical_and(satisfies_xmin, satisfies_xmax)
    y_bools = jnp.logical_and(satisfies_ymin, satisfies_ymax)
    z_bools = jnp.logical_and(satisfies_zmin, satisfies_zmax)

    all_bools = jnp.logical_and(x_bools, jnp.logical_and(y_bools, z_bools))

    # Find the indexes of the patches that contain each point
    patch_idx = jnp.argmax(all_bools, axis=1)

    corners_for_vmap = corners_iter[patch_idx]
    f_for_vmap = f_evals[patch_idx]

    xvals_for_vmap = pts[:, 0]
    yvals_for_vmap = pts[:, 1]
    zvals_for_vmap = pts[:, 2]

    # Interpolate to the target points
    vals = vmapped_interp_to_point(
        xvals_for_vmap, yvals_for_vmap, zvals_for_vmap, corners_for_vmap, f_for_vmap, p
    )
    return vals, pts


@partial(jax.jit, static_argnums=(5,))
def _interp_to_point(
    xval: jnp.array,
    yval: jnp.array,
    zval: jnp.array,
    corners: jnp.array,
    f: jnp.array,
    p: int,
) -> jnp.array:
    """
    For a particular point (xval, yval) this function finds the patch
    that contains the point, constructs a polynomial interpolation matrix
    to that point, and evaluates the interpolant at the point.

    Args:
        xval (jnp.array): x coordinate of the target point. Has shape () or (n,)
        yval (jnp.array): y coordinate of the target point. Has shape () or (n,)
        p (int): Polynomial order of interpolation
        cheby_grid (jnp.array): HPS discretization. Has size (n_leaves, p^2, 2).
        all_corners (jnp.array): Corners of the patches. Has size (n_leaves, 4, 2).
        f (jnp.array): Function evals on the Cheby grid. Has size (n_leaves, p^2)

    Returns:
        jnp.array: Has shape (1,) or (n^2,) depending on the size of input xval and yval. The output
        is the interpolated value at the target point(s).
    """
    # If xval is a scalar, need to reshape it to be a 1D array for the barycentric interp function.
    # If xval is a 1D vector, this doesn't change anything.
    xval = xval.reshape(-1)
    yval = yval.reshape(-1)
    zval = zval.reshape(-1)
    # print("_interp_to_point: xval: ", xval, " xval shape: ", xval.shape)
    # print("_interp_to_point: yval: ", yval, " yval shape: ", yval.shape)

    cheby_pts = chebyshev_points(p)[0]

    out = jnp.zeros_like(xval)

    from_x = affine_transform(cheby_pts, corners[:, 0])
    from_y = affine_transform(cheby_pts, corners[:, 1])
    from_z = affine_transform(cheby_pts, corners[:, 2])
    # # Annoyingly this is how the y vals are ordered
    # from_y = jnp.flip(from_y)

    I = barycentric_lagrange_3d_interpolation_matrix(
        from_x, from_y, from_z, xval, yval, zval
    )

    rearrange_idxes = rearrange_indices_ext_int(p)
    I = I[:, rearrange_idxes]

    return I @ f


vmapped_interp_to_point = jax.vmap(_interp_to_point, in_axes=(0, 0, 0, 0, 0, None))
