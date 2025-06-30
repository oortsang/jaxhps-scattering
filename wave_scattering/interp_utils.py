# wave_scattering/interp_utils
# Helper functions for uniform-to-quadtree and quadtree-to-uniform
# 1. Quadtree leaf ordering-related helper functions
#     a. reorder_leaves_indices
#     b. morton_to_flatten_indices
#     c. prep_quadtree_to_unrolled_indices
# 2. Grid preparation
# Note: these all assume a domain of [-1, 1] or [-1, 1]^2 for simplicity.
#     a. _product_grid
#     b. _leaf_to_tree
#     c. prep_grids_cheb_2d
#     d. prep_grids_unif_2d
# 3. Quadtree leaf-level-related helper functions
# Note: these are taken from the MFISNets codebase ()
# but lightly modified to better leverage vectorized operations
#     a. prep_conv_interp_1d,
#     b. prep_conv_interp_2d,
#     c. apply_conv_interp_1d,
#     d. apply_conv_interp_2d,

import jax
import jax.numpy as jnp
import numpy as np

import scipy.sparse
from typing import Tuple

from src.jaxhps.quadrature import (
    chebyshev_points,
)


# 1. Helper functions for Quadtree leaf ordering
def reorder_leaves_indices(L: int, s: int = 1, new_order: tuple=(3,2,0,1)):
    """This function prepares the indices needed to recursively perform the
    reordering operation as described below:

    Suppose the inputs have children in labeled as [0,1,2,3]
        +-----+-----+-----+-----+
        |  0  |  1  |  2  |  3  |
        +-----+-----+-----+-----+
    This function will re-order the children within the flattened structure.
    For example, setting new_order=(3,2,0,1) would result in
        +-----+-----+-----+-----+
        |  3  |  2  |  0  |  1  |
        +-----+-----+-----+-----+
    which, in the quadtree, corresponds to an spatial organization of
        +-----+-----+
        |  3  |  2  |
        +-----+-----+
        |  0  |  1  |
        +-----+-----+
    in terms of the input.
    Note: this function can be used to put the quadtree into Morton or Z order, but this
    is different from row-major order, as far as the blocks are concerned.

    Parameters:
        L (int): number of levels in the quadtree
        s (int): number of entries per leaf in the quadtree
        new_order (tuple of ints): new relative ordering of the leaves
    Output:
        idcs (jax.array): re-ordered leaf index map with shape (4**L * s**2,)
            Can be used as jax.take(leaf_data, idcs, axis=<relevant axis>)
    """
    idcs = jnp.arange(4**L * s)
    new_order_array = jnp.array(new_order)
    for l in range(0, L):
        idcs = idcs.reshape(4**l, 4, 4**(L-l-1))
        idcs = jnp.take(idcs, new_order_array, axis=1)
    return idcs

def morton_to_flatten_indices(L: int, s: int=1, return_flat=True):
    """Converts the indices for morton ordering into a flattened structure
    Largely lifted from Matt Li's code; for our purposes s=1 seems sufficient
    Matt Li's repo is private, but there is a copy in Borong Zhang's
    repository of Inverse Scattering Problem baseline approaches:
        https://github.com/borongzhang/ISP_baseline
    in src/utils.py.
    """
    if L==0:
        res = jnp.arange(s**2).reshape(s,s) # no re-ordering necessary
    else:
        bsize = 4**(L-1) * s**2 # block size
        tmp = morton_to_flatten_indices(L-1, s, return_flat=False) # recurrence
        res = jnp.block([[tmp, tmp+bsize], [tmp+2*bsize, tmp+3*bsize]])
    return res.flatten() if return_flat else res

def prep_quadtree_to_unrolled_indices(L: int, s: int=1, new_order: Tuple=(0,1,3,2)):
    """Prepares the indices needed to go from the quadtree ordering of
        +-----+-----+
        |  0  |  1  |
        +-----+-----+
        |  3  |  2  |
        +-----+-----+
    to an unrolled ordering based on the spatial layout:
        [[0 1 2   ... 2^L-1]
         [1+2^L   ...      ]
         [        ...      ]
         [4^L-2^L ... 4^L-1]
    Parameters:
        L (int): number of levels in the quadtree
        s (int): number of elements per level in the quadtree
        new_order (tuple): the new ordering of the blocks
            default:   (0,1,3,2)
            alternate: (3,2,0,1) (depends on the desired first-axis ordering)
    """
    quadtree_to_morton   = reorder_leaves_indices(L, s, new_order=new_order)
    morton_to_unrolled   = morton_to_flatten_indices(L, s)
    quadtree_to_unrolled = jnp.take(quadtree_to_morton, morton_to_unrolled)
    return quadtree_to_unrolled

# def apply_quadtree_to_unrolled_indices(
#         data: jax.Array,
#         quadtree_to_unrolled: jax.Array,
#         axis=0
# ) -> jax.Array:
#     """Super basic function to apply the indices, mostly for a reminder of the usage"""
#     return jnp.take(data, quadtree_to_unrolled, axis=axis)

# 2. Grid-related helper functions
########################################
# # Should I duplicate the Chebyshev grid function??
# def _chebyshev_points(n: int) -> jax.Array:
#     """Lifted from src/jax/quadrature/_discretization.py
#     Returns n Chebyshev points over the interval [-1, 1]
#     out[i] = cos(pi * (n-1 - i) / (n-1)) for i={0,...,n-1}
#     The left side of the interval is returned first.
#     Args:
#         n (int): number of Chebyshev points to return
#     Returns:
#         jax.Array: The sampled points in [-1, 1] and the corresponding angles in [0, pi]
#     """
#     cos_args = jnp.arange(n, dtype=jnp.float64) / (n - 1)
#     angles = jnp.flipud(jnp.pi * cos_args)
#     pts = jnp.cos(angles)
#     return pts

def _product_grid(xs: jax.Array, ys: jax.Array) -> jax.Array:
    """Helper function to get the cartesian product of two grids
    Keeps the first grid as the first axis and second grid as the second axis,
    then unrolls them into a (N_x * N_y, 2) array
    """
    return (
        jnp.array(jnp.meshgrid(xs, ys, indexing="ij"))
        .transpose(1,2,0)
        .reshape(-1,2)
    )

def _leaf_to_tree(leaf_xs: jax.Array, L: int) -> jax.Array:
    """Helper function to generate tree-level grids
    Takes in leaf_xs as a leaf-level grid on [-1, 1]
    and maps to 2^L shrunken copies of the leaf-level grid on [-1, 1]
    (i.e., the domain of length 2 gets mapped to a length of 2*2^L)
    """
    return jnp.concatenate([
        (leaf_xs+1) * (1/2**L) - 1 + (2*i/2**L)
        for i in range(2**L)
    ])

def prep_grids_cheb_2d(L: int, p: int) -> tuple[jax.Array]:
    """Prepares chebyshev gridpoints on the domain [-1, 1]
    Returns grids for x, y, and their cartesian product
    Parameters:
        L (int): number of quadtree levels
        p (int): polynomial order of the chebyshev nodes
    Returns:
        (tree_cheb_x, tree_cheb_y, tree_cheb_xy):
            Tree-level chebyshev grids for x, y, and cartesian product (x,y)
            shapes: (2^L * p,), (2^L * p,), and (4^L * p^2, 2)
    """
    leaf_cheb_x = chebyshev_points(p)
    leaf_cheb_y = chebyshev_points(p)[::-1]

    tree_cheb_x  = _leaf_to_tree(leaf_cheb_x, L)
    tree_cheb_y  = _leaf_to_tree(leaf_cheb_y, L)
    tree_cheb_xy = _product_grid(tree_cheb_x, tree_cheb_y)
    return tree_cheb_x, tree_cheb_y, tree_cheb_xy

def prep_grids_unif_2d(
    L: int,
    n_per_leaf: int,
    rel_offset: float = 0,
) -> tuple[jax.Array]:
    """Prepares the uniform gridpoints on the domain [-1, 1]
    Parameters:
        L (int): number of quadtree levels
        n_per_leaf (int): number of gridpoints per leaf
        rel_offset (float): relative offset of where to sample the uniform grida
            If a leaf's chebyshev grid is sampled on the interval [-1, 1],
            the cell_offset values correspond to the following behavior:
            0:   default behavior, sample at [0, 1, 2, ..., n_per_leaf-1]*2/n_per_leaf-1
            0.5: cell-centered, sample at [0.5, 1.5, 2.5, ..., n_per_leaf-0.5]*2/n_per_leaf-1
            1:   sample on the opposite end, at [1, 2, ..., n_per_leaf]*2/n_per_leaf-1
    Returns:
        (tree_unif_x, tree_unif_y, tree_unif_xy):
            Tree-level uniform grids for x, y, and cartesian product (x,y)
            shapes: (2^L * p,), (2^L * p,), and (4^L * p^2, 2)
    """
    tree_n = 2**L * n_per_leaf
    tree_offset = rel_offset/tree_n
    tree_unif_x = tree_offset+jnp.linspace(-1, 1, tree_n, endpoint=False)
    tree_unif_y = tree_offset+jnp.linspace(-1, 1, tree_n, endpoint=False)
    tree_unif_xy = _product_grid(tree_unif_x, tree_unif_y)
    return tree_unif_x, tree_unif_y, tree_unif_xy

# 3. Helper functions for grid-to-grid interpolation
# Modified from the MFISNets repository
def prep_conv_interp_1d(
    points: np.ndarray,
    xi: np.ndarray,
    bc_mode: str = None,
    a_neg_half: bool = True,
) -> scipy.sparse.csr_array:
    """Alternate implementation written to use some vectorization
    Prepares a sparse array to apply convolution for cubic interpolation
    Operates in a single dimension and can be applied to each dimension independently
    to work with higher-dimension data

    Assumes that the entries in `points` are sorted and evenly spaced
    Args:
        points (ndarray): original data grid points
        xi (ndarray): array of points to be sampled as a (m,)-shaped array
        bc_mode (string): how to handle the boundary conditions
            options:
                "periodic": wrap values around
                "extend": extrapolate the missing out-of-boundary values
                    using a rule for cartesian points: f(-1) = 3*f(0) - 3*f(1) + f(2)
                    (see R. Keys 1981 paper below)
                "zero": sets values outside the boundary to zero
        a_neg_half (bool): whether to use a=-1/2 for the convolution filter (otherwise use a=-3/4)
    Returns:
        conv_filter (m by n): sparse linear filter to perform
            cubic interpolation (on padded data)
            Apply to data values with `apply_interp_{1,2}d`
            Note: padding should not be needed except for the inside edge of a polar grid

    For the choice of convolution filter for cubic convolution
    See https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    and R. Keys (1981). "Cubic convolution interpolation for digital image processing".
        IEEE Transactions on Acoustics, Speech, and Signal Processing.
        29 (6): 1153–1160. Bibcode:1981ITASS..29.1153K. CiteSeerX 10.1.1.320.776.
        doi:10.1109/TASSP.1981.1163711 .
    """
    bc_mode = bc_mode.lower() if bc_mode is not None else "zero"
    # Helper variables
    periodic_mode = bc_mode == "periodic"
    extend_mode = bc_mode == "extend"

    if a_neg_half:
        # with a=-1/2, standard choice (seems to be the Catmull-Rom filter?)
        cubic_conv_matrix = 0.5 * np.array(
            [[0, 2, 0, 0], [-1, 0, 1, 0], [2, -5, 4, -1], [-1, 3, -3, 1]]
        )
    else:
        # with a=-3/4, computed with sympy
        # Sometimes gives lower error but has weaker theoretical properties...
        cubic_conv_matrix = 0.25 * np.array(
            [[0, 4, 0, 0], [-3, 0, 3, 0], [6, -9, 6, -3], [-3, 5, -5, 3]]
        )

    n = points.shape[0]
    min_pt = points[0]
    # interval between regularly sampled points
    itvl = (points[-1] - points[0]) / ( n - 1 )
    m = xi.shape[0]

    # Faster to build in LIL form then convert later to CSR
    # This was previously.. probably coo is better now
    # interp_op = scipy.sparse.lil_array((m, n))
    interp_op = scipy.sparse.lil_array((m, n))

    js_float, xs_offset = np.divmod(xi - min_pt, itvl)
    js = js_float.astype(int)
    pos_rel_vec = xs_offset / itvl
    monomials_mat = np.stack(
        [
            np.ones(m),
            pos_rel_vec,
            pos_rel_vec**2,
            pos_rel_vec**3
        ], axis=1
    )
    filters_local = monomials_mat @ cubic_conv_matrix
    # First handle the cases fully in bounds...
    # Identify the target points that are fully in bounds...
    tgt_idcs = np.arange(m)
    tgt_pts_in_bounds  = np.logical_and(js>=1, js<=n-3) # boolean array
    tgt_idcs_in_bounds = tgt_idcs[tgt_pts_in_bounds]    # index array (target points in bounds)
    js_idcs_in_bounds  = js[tgt_pts_in_bounds]          # index array (~relevant source points)

    # Just load the values one-by-one to avoid unholy indexing sorcery
    filters_local_in_bounds = filters_local[tgt_pts_in_bounds]
    interp_op[tgt_idcs_in_bounds, js_idcs_in_bounds-1] = filters_local_in_bounds[:, 0]
    interp_op[tgt_idcs_in_bounds, js_idcs_in_bounds+0] = filters_local_in_bounds[:, 1]
    interp_op[tgt_idcs_in_bounds, js_idcs_in_bounds+1] = filters_local_in_bounds[:, 2]
    interp_op[tgt_idcs_in_bounds, js_idcs_in_bounds+2] = filters_local_in_bounds[:, 3]

    # Handle the boundary conditions
    tgt_pts_out_bounds  = np.logical_not(tgt_pts_in_bounds)
    tgt_idcs_out_bounds = tgt_idcs[tgt_pts_out_bounds]
    js_idcs_out_bounds  = js[tgt_pts_out_bounds]
    if periodic_mode:
        filters_local_out_bounds = filters_local[tgt_pts_out_bounds]
        interp_op[tgt_idcs_out_bounds, (js_idcs_out_bounds-1)%n] = filters_local_out_bounds[:, 0]
        interp_op[tgt_idcs_out_bounds, (js_idcs_out_bounds+0)%n] = filters_local_out_bounds[:, 1]
        interp_op[tgt_idcs_out_bounds, (js_idcs_out_bounds+1)%n] = filters_local_out_bounds[:, 2]
        interp_op[tgt_idcs_out_bounds, (js_idcs_out_bounds+2)%n] = filters_local_out_bounds[:, 3]
    else:
        for i in tgt_idcs_out_bounds:
            x = xi[i]
            j = js[i]
            filter_local = filters_local[i]
            # Assumes zero value beyond the extra single-cell padding
            # Extrapolation rule was linear anyway so fold down the extrapolation
            # into a reduced-length filter
            if extend_mode:
                if j < 1 and (j + 3) >= 0:
                    filter_folded = filter_local[1:] + filter_local[0] * np.array(
                        [3, -3, 1]
                    )
                    interp_op[i, : j + 3] = filter_folded[: j + 3]
                elif j < n and (j + 3) >= n:
                    filter_folded = filter_local[:-1] + filter_local[-1] * np.array(
                        [1, -3, 3]
                    )
                    interp_op[i, j - 1 :] = filter_folded[: n - j + 1]
            else:  # bc_mode == "zero" case
                if j < 1 and (j + 3) >= 0:
                    interp_op[i, : j + 3] = filter_local[: j + 3]
                elif j < n and (j + 3) >= n:
                    interp_op[i, j - 1 :] = filter_local[: n - j + 1]

    # Convert for slightly faster application
    interp_op = scipy.sparse.csr_array(interp_op)
    return interp_op

def prep_conv_interp_2d(
    points_x: np.ndarray,
    points_y: np.ndarray,
    xi: np.ndarray,
    bc_modes: None | str | Tuple = None,
    a_neg_half: bool = True,
) -> Tuple[scipy.sparse.csr_array, scipy.sparse.csr_array]:
    should_split = hasattr(bc_modes, "__len__") and len(bc_modes) == 2
    bc_mode_x = bc_modes[0] if should_split else bc_modes
    bc_mode_y = bc_modes[1] if should_split else bc_modes
    interp_op_x = prep_conv_interp_1d(
        points_x, xi[:, 0], bc_mode=bc_mode_x, a_neg_half=a_neg_half
    )
    interp_op_y = prep_conv_interp_1d(
        points_y, xi[:, 1], bc_mode=bc_mode_y, a_neg_half=a_neg_half
    )
    return interp_op_x, interp_op_y


def apply_conv_interp_1d(
    interp_op: scipy.sparse.csr_array, data_vals: np.ndarray
) -> np.ndarray:
    """Applies the x/y convolutional filter operators onto the padded values
        ~ diag(conv_x (vals) conv_y^T)
    interp_op is   m x nx
    data_vals is nx x {1, ny}
    result is m (or m x ny) interpolated values
    """
    res = interp_op @ data_vals  # .sum(1) # (m, ny+2) to (m,)
    return res


def apply_conv_interp_2d(
    interp_op_x: scipy.sparse.csr_array,
    interp_op_y: scipy.sparse.csr_array,
    data_vals: np.ndarray,
) -> np.ndarray:
    """Applies the x/y convolutional filters onto the padded values
        ~ diag(conv_x (vals) conv_y^T)
    Args:
        interp_op_x (sparse csr array): x-dim interpolation operator with shape (m, nx)
        interp_op_y (sparse csr array): y-dim interpolation operator with shape (m, ny)
        data_vals (ndarray): data values arranged in a grid (nx, ny)
    Returns:
        result (ndarray) is the interpolated values in a (m,) array
    """
    post_x = interp_op_x @ data_vals  # m x (ny+2)
    res = (post_x * interp_op_y).sum(1)
    return res
