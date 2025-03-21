"""This file has functions that create quadrature grids for a 3D cube. This includes a 3D Chebyshev quadrature of the volume and 
2D Gauss-Legendre quadrature of the faces.
"""

from typing import Tuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from hps.src.quadrature.quadrature_utils import chebyshev_points, affine_transform
from hps.src.quadrature.quad_3D.indexing import rearrange_indices_ext_int
from hps.src.quadrature.trees import (
    Node,
    get_all_leaves_jitted,
    get_all_leaves_special_ordering,
)


def get_ordered_lst_of_boundary_nodes(root: Node) -> Tuple[Tuple[Node]]:
    """When looking at each face of the cube from the positive direction, we
    want to return a list of leaves that are ordered like this:
    ---------
    | 3 | 2 |
    ---------
    | 0 | 1 |
    ---------

    Need some notion of tolerance to make sure I am getting leaves with
    """

    # To find the leaves in the (y,z) plane, we need to search through the
    # tree with node ordering [e, h, d, a, f, g, c, b] = [4, 7, 3, 0, 5, 6, 2, 1].
    leaf_ordering_12 = [4, 7, 3, 0, 5, 6, 2, 1]
    all_leaves_12 = get_all_leaves_special_ordering(
        root, child_traversal_order=leaf_ordering_12
    )
    # Find leaves in face 1 which is the x=xmin face.
    leaves_1 = [leaf for leaf in all_leaves_12 if leaf.xmin == root.xmin]
    # Find leaves in face 2 which is the x=xmax face.
    leaves_2 = [leaf for leaf in all_leaves_12 if leaf.xmax == root.xmax]

    # To find the leaves in the (x,z) plane, we need to search through the
    # tree with node ordering [e, f, b, a, h, g, c, d] = [4, 5, 1, 0, 7, 6, 2, 3].
    leaf_ordering_34 = [4, 5, 1, 0, 7, 6, 2, 3]
    all_leaves_34 = get_all_leaves_special_ordering(
        root, child_traversal_order=leaf_ordering_34
    )
    # Find leaves in face 3 which is the y=ymin face.
    leaves_3 = [leaf for leaf in all_leaves_34 if leaf.ymin == root.ymin]
    # Find leaves in face 4 which is the y=ymax face.
    leaves_4 = [leaf for leaf in all_leaves_34 if leaf.ymax == root.ymax]

    # To find the leaves in the (x,y) plane, we can use the original ordering
    all_leaves_56 = get_all_leaves_special_ordering(root)
    # Find leaves in face 5 which is the z=zmin face.
    leaves_5 = [leaf for leaf in all_leaves_56 if leaf.zmin == root.zmin]
    # Find leaves in face 6 which is the z=zmax face.
    leaves_6 = [leaf for leaf in all_leaves_56 if leaf.zmax == root.zmax]
    return leaves_1, leaves_2, leaves_3, leaves_4, leaves_5, leaves_6


def get_all_leaf_3d_cheby_points_uniform_refinement(
    p: int, l: int, corners: jnp.ndarray
) -> jnp.ndarray:
    """
    Given the parameters that describe a quadtree discretization, return an array that gives the locations
    of all of the Chebyshev points on the leaf nodes.

    Args:
        p (int): Number of Chebyshev nodes in one dimension for each leaf node.
        l (int): Number of octree levels. There will be 8**l leaf nodes.
        corners (jnp.ndarray): Has shape (3,2). First row is (x_min, y_min, z_min) and last row is (x_max, y_max, z_max).

    Returns:
        jnp.ndarray: Has shape (8**l, p**3, 3). Lists the Chebyshev points for each leaf node.
    """
    corners_iter = jnp.expand_dims(corners, axis=0)
    for level in range(l):
        corners_iter = vmapped_corners(corners_iter).reshape(-1, 2, 3)

    cheby_pts_1d = chebyshev_points(p)[0]
    all_cheby_points = vmapped_corners_to_cheby_points_lst(corners_iter, cheby_pts_1d)
    return all_cheby_points


def get_all_leaf_3d_cheby_points(p: int, root: Node) -> jnp.array:

    leaves_iter = get_all_leaves_jitted(root)
    bounds = jnp.array(
        [
            [[leaf.xmin, leaf.ymin, leaf.zmin], [leaf.xmax, leaf.ymax, leaf.zmax]]
            for leaf in leaves_iter
        ]
    )
    cheby_points_1d = chebyshev_points(p)[0]
    all_cheby_points = vmapped_corners_to_cheby_points_lst(bounds, cheby_points_1d)
    return all_cheby_points


def get_all_boundary_gauss_legendre_points(q: int, root: Node) -> jnp.ndarray:

    gauss_pts_1d = np.polynomial.legendre.leggauss(q)[0]

    side_leaves = get_ordered_lst_of_boundary_nodes(root)

    # Do first face.
    bounds_1 = jnp.array(
        [[[leaf.ymin, leaf.zmin], [leaf.ymax, leaf.zmax]] for leaf in side_leaves[0]]
    )
    bdry_pts_1 = vmapped_corners_to_gauss_face(bounds_1, gauss_pts_1d)
    # Add xmin in the first column.
    bdry_pts_1 = jnp.stack(
        [
            root.xmin * jnp.ones_like(bdry_pts_1[:, :, 0]),
            bdry_pts_1[:, :, 0],
            bdry_pts_1[:, :, 1],
        ],
        axis=-1,
    )

    # Do second face.
    bounds_2 = jnp.array(
        [[[leaf.ymin, leaf.zmin], [leaf.ymax, leaf.zmax]] for leaf in side_leaves[1]]
    )
    bdry_pts_2 = vmapped_corners_to_gauss_face(bounds_2, gauss_pts_1d)
    bdry_pts_2 = jnp.stack(
        [
            root.xmax * jnp.ones_like(bdry_pts_2[:, :, 0]),
            bdry_pts_2[:, :, 0],
            bdry_pts_2[:, :, 1],
        ],
        axis=-1,
    )

    # Do third face.
    bounds_3 = jnp.array(
        [[[leaf.xmin, leaf.zmin], [leaf.xmax, leaf.zmax]] for leaf in side_leaves[2]]
    )
    bdry_pts_3 = vmapped_corners_to_gauss_face(bounds_3, gauss_pts_1d)
    bdry_pts_3 = jnp.stack(
        [
            bdry_pts_3[:, :, 0],
            root.ymin * jnp.ones_like(bdry_pts_3[:, :, 1]),
            bdry_pts_3[:, :, 1],
        ],
        axis=-1,
    )

    # Do fourth face.
    bounds_4 = jnp.array(
        [[[leaf.xmin, leaf.zmin], [leaf.xmax, leaf.zmax]] for leaf in side_leaves[3]]
    )
    bdry_pts_4 = vmapped_corners_to_gauss_face(bounds_4, gauss_pts_1d)
    bdry_pts_4 = jnp.stack(
        [
            bdry_pts_4[:, :, 0],
            root.ymax * jnp.ones_like(bdry_pts_4[:, :, 1]),
            bdry_pts_4[:, :, 1],
        ],
        axis=-1,
    )

    # Do fifth face.
    bounds_5 = jnp.array(
        [[[leaf.xmin, leaf.ymin], [leaf.xmax, leaf.ymax]] for leaf in side_leaves[4]]
    )
    bdry_pts_5 = vmapped_corners_to_gauss_face(bounds_5, gauss_pts_1d)
    bdry_pts_5 = jnp.stack(
        [
            bdry_pts_5[:, :, 0],
            bdry_pts_5[:, :, 1],
            root.zmin * jnp.ones_like(bdry_pts_5[:, :, 1]),
        ],
        axis=-1,
    )
    # Do sixth face.
    bounds_6 = jnp.array(
        [[[leaf.xmin, leaf.ymin], [leaf.xmax, leaf.ymax]] for leaf in side_leaves[5]]
    )
    bdry_pts_6 = vmapped_corners_to_gauss_face(bounds_6, gauss_pts_1d)
    bdry_pts_6 = jnp.stack(
        [
            bdry_pts_6[:, :, 0],
            bdry_pts_6[:, :, 1],
            root.zmax * jnp.ones_like(bdry_pts_6[:, :, 1]),
        ],
        axis=-1,
    )
    all_bdry_pts = [
        bdry_pts_1,
        bdry_pts_2,
        bdry_pts_3,
        bdry_pts_4,
        bdry_pts_5,
        bdry_pts_6,
    ]
    return jnp.concatenate(all_bdry_pts, axis=0).reshape(-1, 3)


def get_all_boundary_gauss_legendre_points_uniform_refinement(
    q: int, l: int, corners: jnp.ndarray
) -> jnp.ndarray:
    """
    Given the corners of a cube and a number of Gauss-Legendre points, this function will return a list of points forming a 2D G-L grid on the exterior faces of the leaves.

    There are 2**l leaves in each direction, so there are
    2**(2l) leaves on each face.

    Args:
        q (int): Number of G-L nodes per dimension for each leaf.
        l (int): Number of oct-merge subdivisions. There will be 8**l leaf nodes.
        corners (jnp.ndarray): Has shape (3,2)

    Returns:
        jnp.ndarray: Has shape (6 * 2**(2l) * q**2, 3)
    """
    xmin, ymin, zmin = corners[0]
    xmax, ymax, zmax = corners[1]
    gauss_pts_1d = np.polynomial.legendre.leggauss(q)[0]

    corners_xy = jnp.expand_dims(corners[:, :2], axis=0)
    corners_yz = jnp.expand_dims(corners[:, 1:], axis=0)
    corners_xz = jnp.expand_dims(corners[:, [0, 2]], axis=0)

    for level in range(l):
        corners_xy = vmapped_corners_quad(corners_xy).reshape(-1, 2, 2)
        corners_yz = vmapped_corners_quad(corners_yz).reshape(-1, 2, 2)
        corners_xz = vmapped_corners_quad(corners_xz).reshape(-1, 2, 2)

    all_gauss_points_xy = vmapped_corners_to_gauss_face(
        corners_xy, gauss_pts_1d
    ).reshape((-1, 2))
    all_gauss_points_yz = vmapped_corners_to_gauss_face(
        corners_yz, gauss_pts_1d
    ).reshape((-1, 2))
    all_gauss_points_xz = vmapped_corners_to_gauss_face(
        corners_xz, gauss_pts_1d
    ).reshape((-1, 2))

    face_1 = jnp.column_stack(
        (jnp.full(all_gauss_points_yz.shape[0], xmin), all_gauss_points_yz)
    )
    face_2 = jnp.column_stack(
        (jnp.full(all_gauss_points_yz.shape[0], xmax), all_gauss_points_yz)
    )

    # Faces 3 and 4 lay parallel to the (x,z) plane.

    face_3 = jnp.column_stack(
        (
            all_gauss_points_xz[:, 0],
            jnp.full(all_gauss_points_xz.shape[0], ymin),
            all_gauss_points_xz[:, 1],
        )
    )
    face_4 = jnp.column_stack(
        (
            all_gauss_points_xz[:, 0],
            jnp.full(all_gauss_points_xz.shape[0], ymax),
            all_gauss_points_xz[:, 1],
        )
    )

    # Faces 5 and 6 lay parallel to the (x,y) plane.
    face_5 = jnp.column_stack(
        (all_gauss_points_xy, jnp.full(all_gauss_points_xy.shape[0], zmin))
    )
    face_6 = jnp.column_stack(
        (all_gauss_points_xy, jnp.full(all_gauss_points_xy.shape[0], zmax))
    )

    out = jnp.concatenate([face_1, face_2, face_3, face_4, face_5, face_6], axis=0)
    return out


@jax.jit
def _corners_for_oct_subdivision(corners: jnp.ndarray) -> jnp.ndarray:
    """Input has shape (2, 3) and output has shape (8, 2, 3)."""
    x_min, y_min, z_min = corners[0]
    x_max, y_max, z_max = corners[1]

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    corners_a = jnp.array(
        [
            [x_min, y_min, z_mid],
            [x_mid, y_mid, z_max],
        ]
    )
    corners_b = jnp.array(
        [
            [x_mid, y_min, z_mid],
            [x_max, y_mid, z_max],
        ]
    )
    corners_c = jnp.array(
        [
            [x_mid, y_mid, z_mid],
            [x_max, y_max, z_max],
        ]
    )
    corners_d = jnp.array(
        [
            [x_min, y_mid, z_mid],
            [x_mid, y_max, z_max],
        ]
    )
    corners_e = jnp.array(
        [
            [x_min, y_min, z_min],
            [x_mid, y_mid, z_mid],
        ]
    )
    corners_f = jnp.array(
        [
            [x_mid, y_min, z_min],
            [x_max, y_mid, z_mid],
        ]
    )
    corners_g = jnp.array(
        [
            [x_mid, y_mid, z_min],
            [x_max, y_max, z_mid],
        ]
    )
    corners_h = jnp.array(
        [
            [x_min, y_mid, z_min],
            [x_mid, y_max, z_mid],
        ]
    )

    out = jnp.stack(
        [
            corners_a,
            corners_b,
            corners_c,
            corners_d,
            corners_e,
            corners_f,
            corners_g,
            corners_h,
        ]
    )

    return out


@jax.jit
def _corners_for_quad_subdivision(corners: jnp.ndarray) -> jnp.ndarray:
    xmin, ymin = corners[0]
    xmax, ymax = corners[1]
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    corners_a = jnp.array([[xmin, ymin], [xmid, ymid]])
    corners_b = jnp.array([[xmin, ymid], [xmid, ymax]])
    corners_c = jnp.array([[xmid, ymin], [xmax, ymid]])
    corners_d = jnp.array([[xmid, ymid], [xmax, ymax]])
    return jnp.stack([corners_a, corners_b, corners_c, corners_d])


vmapped_corners_quad = jax.vmap(_corners_for_quad_subdivision, in_axes=(0,))

vmapped_corners = jax.vmap(_corners_for_oct_subdivision, in_axes=(0,))


@jax.jit
def corners_to_cheby_points_lst(
    corners: jnp.ndarray, cheby_pts_1d: jnp.ndarray
) -> jnp.ndarray:
    """
    Given a list of corners and 1D Chebyshev nodes, form a 3D Chebyshev tensor product grid on the cube.

    Args:
        corners (jnp.ndarray): Has shape (2,3). First row is (x_min, y_min, z_min) and last row is (x_max, y_max, z_max).
        cheby_pts_1d (jnp.ndarray): Has shape (p,)

    Returns:
        jnp.ndarray: Has shape (p**3, 3)
    """
    p = cheby_pts_1d.shape[0]

    x_min, y_min, z_min = corners[0]
    x_max, y_max, z_max = corners[1]

    x_lims = jnp.array([x_min, x_max])
    y_lims = jnp.array([y_min, y_max])
    z_lims = jnp.array([z_min, z_max])

    x_pts = affine_transform(cheby_pts_1d, x_lims)
    y_pts = affine_transform(cheby_pts_1d, y_lims)
    z_pts = affine_transform(cheby_pts_1d, z_lims)

    X, Y, Z = jnp.meshgrid(x_pts, y_pts, z_pts, indexing="ij")

    together = jnp.concatenate(
        [jnp.expand_dims(X, -1), jnp.expand_dims(Y, -1), jnp.expand_dims(Z, -1)],
        axis=-1,
    )
    out = jnp.reshape(together, (p**3, 3))

    r = rearrange_indices_ext_int(p)
    out = out[r]

    return out


vmapped_corners_to_cheby_points_lst = jax.vmap(
    corners_to_cheby_points_lst, in_axes=(0, None)
)


@jax.jit
def corners_to_gauss_face(corners: jnp.ndarray, gauss_pts_1d: jnp.array) -> jnp.ndarray:
    """
    Given a square defined by two opposing corners, this function will create a Gauss-Legendre grid on the square.

    The square lies in a (x,y), (y,z), or (x,z) plane. Either way, this function expects the corners to look like:
    [[min_first_coord, min_second_coord], [max_first_coord, max_second_coord]]

    Args:
        q (int): Number of G-L nodes per dimension.
        corners (jnp.ndarray): Has shape (2,2).

    Returns:
        jnp.ndarray: Has shape (q**2, 2)
    """
    x_min, y_min = corners[0]
    x_max, y_max = corners[1]

    x_lims = jnp.array([x_min, x_max])
    y_lims = jnp.array([y_min, y_max])

    q = gauss_pts_1d.shape[0]

    x_pts = affine_transform(gauss_pts_1d, x_lims)
    y_pts = affine_transform(gauss_pts_1d, y_lims)

    X, Y = jnp.meshgrid(x_pts, y_pts, indexing="ij")

    together = jnp.concatenate(
        [jnp.expand_dims(X, -1), jnp.expand_dims(Y, -1)], axis=-1
    )
    out = jnp.reshape(together, (q**2, 2))

    return out


vmapped_corners_to_gauss_face = jax.vmap(corners_to_gauss_face, in_axes=(0, None))


@partial(jax.jit, static_argnums=(0,))
def corners_to_gauss_points_lst(q: int, corners: jnp.ndarray) -> jnp.ndarray:
    """
    Given the corners of a cube and a number of Gauss-Legendre points, this
    function will return a list of points forming a 2D G-L grid on
    each face of the cube.

    Args:
        q (int): Number of G-L nodes per dimension.
        corners (jnp.ndarray): Has shape (2,3). First row is (x_min, y_min, z_min) and last row is (x_max, y_max, z_max).

    Returns:
        jnp.ndarray: Has shape (6 * q**2, 2). Lists the faces in a particular order; check
        the oct merge ordering specs for more details.
    """
    x_min, y_min, z_min = corners[0]
    x_max, y_max, z_max = corners[1]

    gauss_pts = np.polynomial.legendre.leggauss(q)[0]

    # Face 1 is the face lying parallel to the (y,z) plane farthest in the -x direction.
    corners_face_1 = corners[:, 1:]
    face_1_2 = corners_to_gauss_face(corners_face_1, gauss_pts)
    # append x_min to face_1
    face_1 = jnp.concatenate(
        [jnp.expand_dims(jnp.ones(q**2) * x_min, -1), face_1_2], axis=-1
    )
    face_2 = jnp.concatenate(
        [jnp.expand_dims(jnp.ones(q**2) * x_max, -1), face_1_2], axis=-1
    )

    corners_face_3 = corners[:, [0, 2]]
    face_3_4 = corners_to_gauss_face(corners_face_3, gauss_pts)
    face_3 = jnp.stack(
        [face_3_4[:, 0], jnp.ones(q**2) * y_min, face_3_4[:, 1]],
        axis=-1,
    )
    face_4 = jnp.stack(
        [face_3_4[:, 0], jnp.ones(q**2) * y_max, face_3_4[:, 1]],
        axis=-1,
    )

    corners_face_5 = corners[:, [0, 1]]
    face_5_6 = corners_to_gauss_face(corners_face_5, gauss_pts)
    face_5 = jnp.stack(
        [face_5_6[:, 0], face_5_6[:, 1], jnp.ones(q**2) * z_min],
        axis=-1,
    )
    face_6 = jnp.stack(
        [face_5_6[:, 0], face_5_6[:, 1], jnp.ones(q**2) * z_max],
        axis=-1,
    )

    out = jnp.concatenate([face_1, face_2, face_3, face_4, face_5, face_6], axis=0)

    return out
