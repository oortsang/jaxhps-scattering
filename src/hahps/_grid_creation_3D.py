from ._discretization_tree import DiscretizationNode3D, get_all_leaves
from ._discretization_tree_operations_3D import (
    get_all_leaves_special_ordering_3D,
)
from ._grid_creation_2D import vmapped_bounds_2D
from .quadrature import chebyshev_points, gauss_points, affine_transform
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, List


def compute_interior_Chebyshev_points_uniform_3D(
    root: DiscretizationNode3D, L: int, p: int
) -> jax.Array:
    bounds = jnp.array(
        [[root.xmin, root.xmax, root.ymin, root.ymax, root.zmin, root.zmax]]
    )
    for level in range(L):
        bounds = vmapped_bounds_3D(bounds).reshape(-1, 6)

    cheby_pts_1d = chebyshev_points(p)
    all_cheby_points = vmapped_bounds_to_cheby_points_3D(bounds, cheby_pts_1d)
    return all_cheby_points


@jax.jit
def bounds_for_oct_subdivision(bounds: jax.Array) -> jax.Array:
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    corners_a = jnp.array(
        [x_min, x_mid, y_min, y_mid, z_mid, z_max],
    )
    corners_b = jnp.array(
        [x_mid, x_max, y_min, y_mid, z_mid, z_max],
    )
    corners_c = jnp.array(
        [x_mid, x_max, y_mid, y_max, z_mid, z_max],
    )
    corners_d = jnp.array(
        [x_min, x_mid, y_mid, y_max, z_mid, z_max],
    )
    corners_e = jnp.array(
        [x_min, x_mid, y_min, y_mid, z_min, z_mid],
    )
    corners_f = jnp.array(
        [x_mid, x_max, y_min, y_mid, z_min, z_mid],
    )
    corners_g = jnp.array(
        [x_mid, x_max, y_mid, y_max, z_min, z_mid],
    )
    corners_h = jnp.array(
        [x_min, x_mid, y_mid, y_max, z_min, z_mid],
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


vmapped_bounds_3D = jax.vmap(bounds_for_oct_subdivision)


@jax.jit
def bounds_to_cheby_points_3D(
    bounds: jnp.ndarray, cheby_pts_1d: jnp.ndarray
) -> jnp.ndarray:
    p = cheby_pts_1d.shape[0]

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    x_lims = jnp.array([x_min, x_max])
    y_lims = jnp.array([y_min, y_max])
    z_lims = jnp.array([z_min, z_max])

    x_pts = affine_transform(cheby_pts_1d, x_lims)
    y_pts = affine_transform(cheby_pts_1d, y_lims)
    z_pts = affine_transform(cheby_pts_1d, z_lims)

    X, Y, Z = jnp.meshgrid(x_pts, y_pts, z_pts, indexing="ij")

    together = jnp.concatenate(
        [
            jnp.expand_dims(X, -1),
            jnp.expand_dims(Y, -1),
            jnp.expand_dims(Z, -1),
        ],
        axis=-1,
    )
    out = jnp.reshape(together, (p**3, 3))

    r = rearrange_indices_ext_int(p)
    out = out[r]

    return out


vmapped_bounds_to_cheby_points_3D = jax.vmap(
    bounds_to_cheby_points_3D, in_axes=(0, None)
)


def compute_boundary_Gauss_points_uniform_3D(
    root: DiscretizationNode3D, L: int, q: int
) -> jax.Array:
    # xmin, ymin, zmin = corners[0]
    # xmax, ymax, zmax = corners[1]
    gauss_pts_1d = gauss_points(q)

    corners_xy = jnp.array([[root.xmin, root.xmax, root.ymin, root.ymax]])
    corners_yz = jnp.array([[root.ymin, root.ymax, root.zmin, root.zmax]])
    corners_xz = jnp.array([[root.xmin, root.xmax, root.zmin, root.zmax]])
    # corners_yz = jnp.expand_dims(corners[:, 1:], axis=0)
    # corners_xz = jnp.expand_dims(corners[:, [0, 2]], axis=0)

    for level in range(L):
        corners_xy = vmapped_bounds_2D(corners_xy).reshape(-1, 4)
        corners_yz = vmapped_bounds_2D(corners_yz).reshape(-1, 4)
        corners_xz = vmapped_bounds_2D(corners_xz).reshape(-1, 4)

    all_gauss_points_xy = vmapped_bounds_to_gauss_face(
        corners_xy, gauss_pts_1d
    ).reshape((-1, 2))
    all_gauss_points_yz = vmapped_bounds_to_gauss_face(
        corners_yz, gauss_pts_1d
    ).reshape((-1, 2))
    all_gauss_points_xz = vmapped_bounds_to_gauss_face(
        corners_xz, gauss_pts_1d
    ).reshape((-1, 2))

    face_1 = jnp.column_stack(
        (
            jnp.full(all_gauss_points_yz.shape[0], root.xmin),
            all_gauss_points_yz,
        )
    )
    face_2 = jnp.column_stack(
        (
            jnp.full(all_gauss_points_yz.shape[0], root.xmax),
            all_gauss_points_yz,
        )
    )

    # Faces 3 and 4 lay parallel to the (x,z) plane.

    face_3 = jnp.column_stack(
        (
            all_gauss_points_xz[:, 0],
            jnp.full(all_gauss_points_xz.shape[0], root.ymin),
            all_gauss_points_xz[:, 1],
        )
    )
    face_4 = jnp.column_stack(
        (
            all_gauss_points_xz[:, 0],
            jnp.full(all_gauss_points_xz.shape[0], root.ymax),
            all_gauss_points_xz[:, 1],
        )
    )

    # Faces 5 and 6 lay parallel to the (x,y) plane.
    face_5 = jnp.column_stack(
        (
            all_gauss_points_xy,
            jnp.full(all_gauss_points_xy.shape[0], root.zmin),
        )
    )
    face_6 = jnp.column_stack(
        (
            all_gauss_points_xy,
            jnp.full(all_gauss_points_xy.shape[0], root.zmax),
        )
    )

    out = jnp.concatenate(
        [face_1, face_2, face_3, face_4, face_5, face_6], axis=0
    )
    return out


@jax.jit
def bounds_to_gauss_face(
    bounds: jax.Array, gauss_pts_1d: jax.Array
) -> jax.Array:
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
    x_min, x_max, y_min, y_max = bounds

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


vmapped_bounds_to_gauss_face = jax.vmap(
    bounds_to_gauss_face, in_axes=(0, None)
)


def compute_interior_Chebyshev_points_adaptive_3D(
    root: DiscretizationNode3D, p: int
) -> jax.Array:
    leaves_iter = get_all_leaves(root)
    bounds = jnp.array(
        [
            [leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax, leaf.zmin, leaf.zmax]
            for leaf in leaves_iter
        ]
    )
    cheby_points_1d = chebyshev_points(p)
    all_cheby_points = vmapped_bounds_to_cheby_points_3D(
        bounds, cheby_points_1d
    )
    return all_cheby_points


def compute_boundary_Gauss_points_adaptive_3D(
    root: DiscretizationNode3D, q: int
) -> jax.Array:
    gauss_pts_1d = np.polynomial.legendre.leggauss(q)[0]

    side_leaves = get_ordered_lst_of_boundary_nodes(root)

    # Do first face.
    bounds_1 = jnp.array(
        [
            [leaf.ymin, leaf.ymax, leaf.zmin, leaf.zmax]
            for leaf in side_leaves[0]
        ]
    )
    bdry_pts_1 = vmapped_bounds_to_gauss_face(bounds_1, gauss_pts_1d)
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
        [
            [leaf.ymin, leaf.ymax, leaf.zmin, leaf.zmax]
            for leaf in side_leaves[1]
        ]
    )
    bdry_pts_2 = vmapped_bounds_to_gauss_face(bounds_2, gauss_pts_1d)
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
        [
            [leaf.xmin, leaf.xmax, leaf.zmin, leaf.zmax]
            for leaf in side_leaves[2]
        ]
    )
    bdry_pts_3 = vmapped_bounds_to_gauss_face(bounds_3, gauss_pts_1d)
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
        [
            [leaf.xmin, leaf.xmax, leaf.zmin, leaf.zmax]
            for leaf in side_leaves[3]
        ]
    )
    bdry_pts_4 = vmapped_bounds_to_gauss_face(bounds_4, gauss_pts_1d)
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
        [
            [leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax]
            for leaf in side_leaves[4]
        ]
    )
    bdry_pts_5 = vmapped_bounds_to_gauss_face(bounds_5, gauss_pts_1d)
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
        [
            [leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax]
            for leaf in side_leaves[5]
        ]
    )
    bdry_pts_6 = vmapped_bounds_to_gauss_face(bounds_6, gauss_pts_1d)
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


def get_ordered_lst_of_boundary_nodes(
    root: DiscretizationNode3D,
) -> Tuple[Tuple[DiscretizationNode3D]]:
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
    all_leaves_12 = get_all_leaves_special_ordering_3D(
        root, child_traversal_order=leaf_ordering_12
    )
    # Find leaves in face 1 which is the x=xmin face.
    leaves_1 = [leaf for leaf in all_leaves_12 if leaf.xmin == root.xmin]
    # Find leaves in face 2 which is the x=xmax face.
    leaves_2 = [leaf for leaf in all_leaves_12 if leaf.xmax == root.xmax]

    # To find the leaves in the (x,z) plane, we need to search through the
    # tree with node ordering [e, f, b, a, h, g, c, d] = [4, 5, 1, 0, 7, 6, 2, 3].
    leaf_ordering_34 = [4, 5, 1, 0, 7, 6, 2, 3]
    all_leaves_34 = get_all_leaves_special_ordering_3D(
        root, child_traversal_order=leaf_ordering_34
    )
    # Find leaves in face 3 which is the y=ymin face.
    leaves_3 = [leaf for leaf in all_leaves_34 if leaf.ymin == root.ymin]
    # Find leaves in face 4 which is the y=ymax face.
    leaves_4 = [leaf for leaf in all_leaves_34 if leaf.ymax == root.ymax]

    # To find the leaves in the (x,y) plane, we can use the original ordering
    all_leaves_56 = get_all_leaves_special_ordering_3D(root)
    # Find leaves in face 5 which is the z=zmin face.
    leaves_5 = [leaf for leaf in all_leaves_56 if leaf.zmin == root.zmin]
    # Find leaves in face 6 which is the z=zmax face.
    leaves_6 = [leaf for leaf in all_leaves_56 if leaf.zmax == root.zmax]
    return leaves_1, leaves_2, leaves_3, leaves_4, leaves_5, leaves_6


@partial(jax.jit, static_argnums=(0,))
def rearrange_indices_ext_int(p: int) -> jnp.ndarray:
    # out = np.zeros(p**3, dtype=int)
    idxes = np.arange(p**3)

    # The first p^2 points and last p^2 points are the points where x=x_min and x=x_max
    # respectively. This happens when (idx // p**2) == 0 or (idx // p**2) == p-1.
    left_face = idxes // p**2 == 0
    right_face = idxes // p**2 == p - 1

    mask = np.logical_or(left_face, right_face)

    # The top and bottom faces of the cube are the faces where y=y_min and y=y_max respectively.
    # This happens when (idx // p) % p == 0 or (idx // p) % p == p-1.
    bottom_face = np.logical_and((idxes // p) % p == 0, ~mask)
    top_face = np.logical_and((idxes // p) % p == p - 1, ~mask)

    mask = np.logical_or(mask, np.logical_or(bottom_face, top_face))

    # The front and back faces of the cube are the faces where z=z_min and z=z_max respectively.
    # This happens when idx % p == 0 or idx % p == p-1.
    front_face = np.logical_and(idxes % p == 0, ~mask)
    back_face = np.logical_and(idxes % p == p - 1, ~mask)

    mask = np.logical_or(mask, np.logical_or(front_face, back_face))

    out = jnp.concatenate(
        [
            idxes[left_face],
            idxes[right_face],
            idxes[bottom_face],
            idxes[top_face],
            idxes[front_face],
            idxes[back_face],
            idxes[~mask],
        ]
    )

    return out


def get_all_uniform_leaves_3D(
    root: DiscretizationNode3D, L: int
) -> List[DiscretizationNode3D]:
    bounds = jnp.array(
        [[root.xmin, root.xmax, root.ymin, root.ymax, root.zmin, root.zmax]]
    )

    for _ in range(L):
        bounds = vmapped_bounds_3D(bounds).reshape(-1, 6)

    node_lst = [
        DiscretizationNode3D(
            xmin=x[0], xmax=x[1], ymin=x[2], ymax=x[3], zmin=x[4], zmax=x[5]
        )
        for x in bounds
    ]
    return node_lst
