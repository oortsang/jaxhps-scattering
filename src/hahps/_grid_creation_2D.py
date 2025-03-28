from ._discretization_tree import DiscretizationNode2D, get_all_leaves
from ._discretization_tree_operations_2D import (
    get_ordered_lst_of_boundary_nodes,
)
from .quadrature import (
    chebyshev_points,
    gauss_points,
    affine_transform,
    meshgrid_to_lst_of_pts,
)
import jax
import jax.numpy as jnp
from typing import List
import numpy as np
from functools import partial


# Putting this under jit compilation raises a tracer leak error when using the
# adaptive discretization routines.
# @partial(jax.jit, static_argnums=(1, 2))
def compute_interior_Chebyshev_points_uniform_2D(
    root: DiscretizationNode2D, L: int, p: int
) -> jax.Array:
    bounds = jnp.array([[root.xmin, root.xmax, root.ymin, root.ymax]])

    for level in range(L):
        bounds = vmapped_bounds_2D(bounds).reshape(-1, 4)

    cheby_pts_1D = chebyshev_points(p)
    all_cheby_points = vmapped_bounds_to_cheby_points_2D(bounds, cheby_pts_1D)
    return all_cheby_points


@jax.jit
def bounds_for_quad_subdivision(bounds: jax.Array) -> jax.Array:
    xmin, xmax, ymin, ymax = bounds
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2
    return jnp.array(
        [
            [xmin, x_mid, ymin, y_mid],  # SW
            [x_mid, xmax, ymin, y_mid],  # SE
            [x_mid, xmax, y_mid, ymax],  # NE
            [xmin, x_mid, y_mid, ymax],  # NW
        ]
    )


vmapped_bounds_2D = jax.vmap(bounds_for_quad_subdivision)


@jax.jit
def bounds_to_cheby_points_2D(
    bounds: jnp.ndarray, cheby_pts_1d: jnp.ndarray
) -> jnp.ndarray:
    xs = bounds[:2]
    ys = bounds[2:]

    x_pts = affine_transform(cheby_pts_1d, xs)
    y_pts = affine_transform(cheby_pts_1d, ys)

    X, Y = jnp.meshgrid(x_pts, jnp.flipud(y_pts), indexing="ij")
    cheby_pts = meshgrid_to_lst_of_pts(X, Y)

    r = rearrange_indices_ext_int(cheby_pts_1d.shape[0])
    cheby_pts = cheby_pts[r]

    return cheby_pts


vmapped_bounds_to_cheby_points_2D = jax.vmap(
    bounds_to_cheby_points_2D, in_axes=(0, None)
)


@partial(jax.jit, static_argnums=(1, 2))
def compute_boundary_Gauss_points_uniform_2D(
    root: DiscretizationNode2D, L: int, q: int
) -> jax.Array:
    gauss_pts_1d = gauss_points(q)
    n_patches_across_side = 2**L

    x_breakpoints = jnp.linspace(
        root.xmin, root.xmax, n_patches_across_side + 1
    )
    y_breakpoints = jnp.linspace(
        root.ymin, root.ymax, n_patches_across_side + 1
    )

    x_gauss_nodes = jnp.concatenate(
        [
            affine_transform(gauss_pts_1d, x_breakpoints[i : i + 2])
            for i in range(n_patches_across_side)
        ]
    )
    y_gauss_nodes = jnp.concatenate(
        [
            affine_transform(gauss_pts_1d, y_breakpoints[i : i + 2])
            for i in range(n_patches_across_side)
        ]
    )
    gauss_nodes = jnp.concatenate(
        [
            jnp.column_stack(
                (x_gauss_nodes, jnp.full(x_gauss_nodes.shape[0], root.ymin))
            ),
            jnp.column_stack(
                (jnp.full(y_gauss_nodes.shape[0], root.xmax), y_gauss_nodes)
            ),
            jnp.column_stack(
                (
                    jnp.flipud(x_gauss_nodes),
                    jnp.full(x_gauss_nodes.shape[0], root.ymax),
                )
            ),
            jnp.column_stack(
                (
                    jnp.full(y_gauss_nodes.shape[0], root.xmin),
                    jnp.flipud(y_gauss_nodes),
                )
            ),
        ]
    )
    return gauss_nodes


# @partial(jax.jit, static_argnums=(1,))
def compute_interior_Chebyshev_points_adaptive_2D(
    root: DiscretizationNode2D, p: int
) -> jax.Array:
    leaves_iter = get_all_leaves(root)
    bounds = jnp.array(
        [[leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax] for leaf in leaves_iter]
    )
    cheby_pts_1d = chebyshev_points(p)
    all_cheby_points = vmapped_bounds_to_cheby_points_2D(bounds, cheby_pts_1d)
    return all_cheby_points


# This can not be jax.jit because it calls find_node_at_corner with
# traced arrays, which raises an error.
def compute_boundary_Gauss_points_adaptive_2D(
    root: DiscretizationNode2D, q: int
) -> jax.Array:
    gauss_pts_1d = gauss_points(q)

    corners = get_ordered_lst_of_boundary_nodes(root)

    west = root.xmin
    east = root.xmax
    south = root.ymin
    north = root.ymax

    south_gauss_nodes = jnp.concatenate(
        [
            affine_transform(gauss_pts_1d, [node.xmin, node.xmax])
            for node in corners[0]
        ]
    )
    east_gauss_nodes = jnp.concatenate(
        [
            affine_transform(gauss_pts_1d, [node.ymin, node.ymax])
            for node in corners[1]
        ]
    )
    north_gauss_nodes = jnp.concatenate(
        [
            affine_transform(gauss_pts_1d, [node.xmax, node.xmin])
            for node in corners[2]
        ]
    )
    west_gauss_nodes = jnp.concatenate(
        [
            affine_transform(gauss_pts_1d, [node.ymax, node.ymin])
            for node in corners[3]
        ]
    )
    gauss_nodes = jnp.concatenate(
        [
            jnp.column_stack(
                (
                    south_gauss_nodes,
                    jnp.full(south_gauss_nodes.shape[0], south),
                )
            ),
            jnp.column_stack(
                (jnp.full(east_gauss_nodes.shape[0], east), east_gauss_nodes)
            ),
            jnp.column_stack(
                (
                    north_gauss_nodes,
                    jnp.full(north_gauss_nodes.shape[0], north),
                )
            ),
            jnp.column_stack(
                (jnp.full(west_gauss_nodes.shape[0], west), west_gauss_nodes)
            ),
        ]
    )
    return gauss_nodes


@partial(jax.jit, static_argnums=(0,))
def rearrange_indices_ext_int(n: int) -> jnp.ndarray:
    """This function gives the array indices to rearrange the 2D Cheby grid so that the
    4(p-1) boundary points are listed first, starting at the SW corner and going clockwise around the
    boundary. The interior points are listed after.
    """

    idxes = np.zeros(n**2, dtype=int)
    # S border
    for i, j in enumerate(range(n - 1, n**2, n)):
        idxes[i] = j
    # W border
    for i, j in enumerate(range(n**2 - 2, n**2 - n - 1, -1)):
        idxes[n + i] = j
    # N border
    for i, j in enumerate(range(n**2 - 2 * n, 0, -n)):
        idxes[2 * n - 1 + i] = j
    # S border
    for i, j in enumerate(range(1, n - 1)):
        idxes[3 * n - 2 + i] = j
    # Loop through the indices in column-rasterized form and fill in the ones from the interior.
    current_idx = 4 * n - 4
    nums = np.arange(n**2)
    for i in nums:
        if i not in idxes:
            idxes[current_idx] = i
            current_idx += 1
        else:
            continue

    return jnp.array(idxes)


def get_all_uniform_leaves_2D(
    root: DiscretizationNode2D, L: int
) -> List[DiscretizationNode2D]:
    bounds = jnp.array([[root.xmin, root.xmax, root.ymin, root.ymax]])

    for _ in range(L):
        bounds = vmapped_bounds_2D(bounds).reshape(-1, 4)

    node_lst = [
        DiscretizationNode2D(xmin=x[0], xmax=x[1], ymin=x[2], ymax=x[3])
        for x in bounds
    ]
    return node_lst
