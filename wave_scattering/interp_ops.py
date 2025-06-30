# wave_scattering/interp_ops.py
# Interpolation operator objects
# Classes:
# 1. QuadtreeToUniform
# 2. UniformToQuadtree

import jax
import jax.numpy as jnp
import numpy as np

from wave_scattering.interp_utils import (
    prep_quadtree_to_unrolled_indices,
    prep_grids_cheb_2d,
    prep_grids_unif_2d,
    prep_conv_interp_2d,
    apply_conv_interp_2d,
)

from src.jaxhps._grid_creation_2D import (
    rearrange_indices_ext_int as rearrange_indices_ext_int_2D,
)
from src.jaxhps.quadrature import (
    barycentric_lagrange_interpolation_matrix_2D,
    # chebyshev_points,
)


# 1. Quadtree-to-uniform object
########################################

# Quadtree (Chebyshev) -> Uniform
class QuadtreeToUniform:
    """Class to interpolate data from quadtree form with chebyshev-grid leaves
    to a contiguous form on a uniform grid.
    Simplifying assumptions:
    1. Assume each leaf contains an integer number of points from the uniform grid,
    and that these points are always in the same positions relative to the leaf boxes
    2. Assume that the domain size does not change
    """
    def __init__(self, L: int, p: int, n_per_leaf: int, cell_offset: float = 0):
        """Set up the reusable objects
        Parameters:
            L (int): number of levels in the tree
            p (int): polynomial order of the leaf-level chebyshev grids
            n_per_leaf (int): number of points in the uniform grid per leaf
            cell_offset (float): relative offset of where to sample the uniform grid
                If a leaf's chebyshev grid is sampled on the interval [-1, 1],
                the cell_offset values correspond to the following behavior:
                0:   default behavior, sample at    [0, 1, 2, ..., n_per_leaf-1]*2/n_per_leaf-1
                0.5: cell-centered, sample at       [0.5, 1.5, 2.5, ..., n_per_leaf-0.5]*2/n_per_leaf-1
                1:   sample on the opposite end, at [1, 2, ..., n_per_leaf]*2/n_per_leaf-1
        """
        self.L = L
        self.p = p
        self.n_per_leaf = n_per_leaf

        self.quadtree_to_unrolled_idcs = prep_quadtree_to_unrolled_indices(L)
        # self.leaf_cheb_x = chebyshev_points(p)
        # self.leaf_cheb_y = chebyshev_points(p)[::-1]
        # self.cell_offset = cell_offset
        # leaf_offset = cell_offset * 0.5/n_per_leaf
        # self.leaf_unif_x = leaf_offset+jnp.linspace(-1, 1, n_per_leaf, endpoint=False)
        # self.leaf_unif_y = leaf_offset+jnp.linspace(-1, 1, n_per_leaf, endpoint=False) # [::-1]

        leaf_cheb_grids = prep_grids_cheb_2d(0, p)
        self.leaf_cheb_x = leaf_cheb_grids[0]
        self.leaf_cheb_y = leaf_cheb_grids[1]
        leaf_unif_grids = prep_grids_unif_2d(0, n_per_leaf, cell_offset)
        self.leaf_unif_x = leaf_unif_grids[0]
        self.leaf_unif_y = leaf_unif_grids[1]

        tmp_interp_leaf_cheb_to_unif = barycentric_lagrange_interpolation_matrix_2D(
            from_pts_x=self.leaf_cheb_x,
            from_pts_y=self.leaf_cheb_y,
            to_pts_x=self.leaf_unif_x,
            to_pts_y=self.leaf_unif_y,
        )
        self.rearrange_leaf_idcs = rearrange_indices_ext_int_2D(p)
        self.interp_leaf_cheb_to_unif = (
            tmp_interp_leaf_cheb_to_unif
            [:, self.rearrange_leaf_idcs]
        )

    def apply(self, data_quadtree_cheb: jax.Array) -> jax.Array:
        """Applies the quadtree chebyshev-to-uniform operation
        Parameters:
            data_quadtree_cheb (jax.Array, shape: (4**L, p**2, ...)):
                Data from the HPS quadtree, sampled on Chebyshev grids at the leaf level
        Output:
            data_unif (jax.Array, shape (2**L * n_per_leaf, 2**L * n_per_leaf), ...):
                Data on the uniform grid without the quadtree structure
        """
        L = self.L
        p = self.p
        n_per_leaf = self.n_per_leaf
        leftover_shape = data_quadtree_cheb.shape[2:]
        leftover_idcs = 4 + jnp.arange(0, data_quadtree_cheb.ndim-2)

        # 1. Reorder the leaf nodes to a more standard ordering (~block-wise row-major-order)
        data_leaves_cheb = jnp.take(
            data_quadtree_cheb,
            self.quadtree_to_unrolled_idcs,
            axis = 0,
        )

        # 2. Map values from Chebyshev to Uniform grids on the leaf-level
        data_leaves_unif = jnp.einsum(
            "jl,il...->ij...",
            self.interp_leaf_cheb_to_unif,
            data_leaves_cheb
        )

        # 3. Reshape leaf-level data into square matrices, then rearrange
        data_unif = (
            data_leaves_unif
            .reshape(2**L, 2**L, n_per_leaf, n_per_leaf, *leftover_shape)
            .transpose(0,3,1,2, *leftover_idcs)
            .reshape(2**L * n_per_leaf, 2**L * n_per_leaf, *leftover_shape)

            # Not sure why the outputs need to be transposed here...
            .transpose(1,0, *(leftover_idcs-2))
        )
        return data_unif



# 2. Uniform-to-Quadtree (Chebyshev)
########################################
class UniformToQuadtree:
    """Class to interpolate data from a contiguous uniform grid to
    a quadtree form with chebyshev-grid leaves. Uses bicubic interpolation.
    Simplifying assumptions:
    1. Assume each leaf contains an integer number of points from the uniform grid,
    and that these points are always in the same positions relative to the leaf boxes
    2. Assume that the domain size does not change
    For now, fix it at [-1, 1, -1, 1] for both input/output since the true scaling does
    not really matter, other than if the domain were non-square.
    """
    def __init__(self, L: int, p: int, n_per_leaf: int, cell_offset: float = 0):
        """Set up the reusable objects
        Parameters:
            L (int): number of levels in the tree
            p (int): polynomial order of the leaf-level chebyshev grids
            n_per_leaf (int): number of points in the uniform grid per leaf
            cell_offset (float): relative offset of where to sample the uniform grid
                If a leaf's chebyshev grid is sampled on the interval [-1, 1],
                the cell_offset values correspond to the following behavior:
                0:   default behavior, sample at    [0, 1, 2, ..., n_per_leaf-1]*2/n_per_leaf-1
                0.5: cell-centered, sample at       [0.5, 1.5, 2.5, ..., n_per_leaf-0.5]*2/n_per_leaf-1
                1:   sample on the opposite end, at [1, 2, ..., n_per_leaf]*2/n_per_leaf-1
        """
        self.L = L
        self.p = p
        self.n_per_leaf = n_per_leaf
        n = 2**L * n_per_leaf
        self.n = n

        # Permutation maps for ordering leaves and chebyshev points within the leaves
        new_order = (0, 1, 3, 2)
        self.quadtree_to_unrolled_idcs = prep_quadtree_to_unrolled_indices(L, new_order=new_order)
        self.unrolled_to_quadtree_idcs = jnp.argsort(self.quadtree_to_unrolled_idcs) # invert
        self.rearrange_leaf_idcs       = rearrange_indices_ext_int_2D(p)
        self.inv_rearrange_leaf_idcs   = jnp.argsort(self.rearrange_leaf_idcs) # invert

        # leaf-level grids, for simplicity scaled on [-1, 1]
        leaf_cheb_grids = prep_grids_cheb_2d(0, p)
        leaf_unif_grids = prep_grids_unif_2d(0, n_per_leaf, cell_offset)
        self.leaf_cheb_x  = leaf_cheb_grids[0]
        self.leaf_cheb_y  = leaf_cheb_grids[1]
        self.leaf_cheb_xy = leaf_cheb_grids[2]
        self.leaf_unif_x  = leaf_unif_grids[0]
        self.leaf_unif_y  = leaf_unif_grids[1]
        self.leaf_unif_xy = leaf_unif_grids[2]

        # Tree-level grids, also scaled on [-1, 1]
        tree_cheb_grids = prep_grids_cheb_2d(L, p)
        tree_unif_grids = prep_grids_unif_2d(L, n_per_leaf, cell_offset)
        self.tree_cheb_x  = tree_cheb_grids[0]
        self.tree_cheb_y  = tree_cheb_grids[1]
        self.tree_cheb_xy = tree_cheb_grids[2]
        self.tree_unif_x  = tree_unif_grids[0]
        self.tree_unif_y  = tree_unif_grids[1]
        self.tree_unif_xy = tree_unif_grids[2]
        # print(f"tree_cheb_x.shape= {self.tree_cheb_x.shape}")
        # print(f"tree_cheb_xy.shape={self.tree_cheb_xy.shape}")
        # print(f"tree_unif_x.shape= {self.tree_unif_x.shape}")
        # print(f"tree_unif_xy.shape={self.tree_unif_xy.shape}")


        # Get the chebyshev grid with hps tree ordering
        self.hps_tree_cheb_xy = (
            self.tree_cheb_xy
            .reshape(2**L, p, 2**L, p, 2) # Get leaf/chebyshev structure
            .transpose(2, 0, 1, 3, 4)     # Put leaf ordering together
            .reshape(4**L, p**2, 2)       # Combine leaf ordering/content
            [self.unrolled_to_quadtree_idcs, :, :] # Rearrange leaf nodes
            [:, self.rearrange_leaf_idcs, :] # Rearrange Chebyshev nodes
            .reshape(-1, 2) # Unroll
        )

        # Interpolation operation, tree-wide
        tree_unif_to_cheb_x, tree_unif_to_cheb_y = prep_conv_interp_2d(
            self.tree_unif_x,
            self.tree_unif_y,
            self.hps_tree_cheb_xy,
            # use a zero-valued boundary condition to mimic empty neighboring leaves
            bc_modes=("zero", "zero"),
        )
        # Should/can I move the interpolation operators to the GPU?
        self.tree_unif_to_cheb_x = tree_unif_to_cheb_x
        self.tree_unif_to_cheb_y = tree_unif_to_cheb_y

    def apply(self, data_unif: jax.Array) -> jax.Array:
        """Take the data on the uniform grid, then map to the HPS quadtree format with leaf-level chebyshev grids
        Assume data_unif has shape (N, N)
        Parameters:
            data_unif (jax.Array, shape (N, N)): data on the uniform grid
        Output:
            data_tree_cheb (jax.Array, shape (4**L, p**2)): data in the quadtree/chebyshev-grid format
        """
        L = self.L
        p = self.p
        n_per_leaf = self.n_per_leaf
        N = 2**L * n_per_leaf
        # leftover_shape = data_unif.shape[2:]
        # leftover_idcs  = 4 + jnp.arange(0, data_unif.ndim-2)

        data_tree_cheb = apply_conv_interp_2d(
            self.tree_unif_to_cheb_x,
            self.tree_unif_to_cheb_y,
            data_unif.reshape(N, N),
        ).reshape(4**L, p**2)
        return data_tree_cheb
