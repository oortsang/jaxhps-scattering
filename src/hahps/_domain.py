from ._discretization_tree import (
    DiscretizationNode2D,
    DiscretizationNode3D,
    get_all_leaves,
)
from ._grid_creation_2D import (
    compute_interior_Chebyshev_points_uniform_2D,
    compute_interior_Chebyshev_points_adaptive_2D,
    compute_boundary_Gauss_points_uniform_2D,
    compute_boundary_Gauss_points_adaptive_2D,
    get_all_uniform_leaves_2D,
)
from ._grid_creation_3D import (
    compute_interior_Chebyshev_points_uniform_3D,
    compute_interior_Chebyshev_points_adaptive_3D,
    compute_boundary_Gauss_points_uniform_3D,
    compute_boundary_Gauss_points_adaptive_3D,
    get_all_uniform_leaves_3D,
)
from ._interpolation_methods import (
    interp_from_hps_2D,
    interp_from_hps_3D,
    interp_to_hps_2D,
    interp_to_hps_3D,
)
import jax
import jax.numpy as jnp
import logging
from typing import Callable


class Domain:
    def __init__(
        self,
        p: int,
        q: int,
        root: DiscretizationNode2D | DiscretizationNode3D,
        L: int | None = None,
    ):
        self.p = p
        self.q = q
        self.root = root
        self.L = L

        self.bool_2D = isinstance(root, DiscretizationNode2D)

        if self.L is not None:
            self.bool_uniform = True
            # Depending on whether root is a DiscretizationNode2D or
            # DiscretizationNode3D, we compute the grid points differently
            if self.bool_2D:
                self.interior_points = (
                    compute_interior_Chebyshev_points_uniform_2D(root, L, p)
                )
                self.boundary_points = (
                    compute_boundary_Gauss_points_uniform_2D(root, L, q)
                )
            else:
                self.interior_points = (
                    compute_interior_Chebyshev_points_uniform_3D(root, L, p)
                )
                self.boundary_points = (
                    compute_boundary_Gauss_points_uniform_3D(root, L, q)
                )

        else:
            # If L is None, we're using an adaptive discretization
            self.bool_uniform = False
            if self.bool_2D:
                self.interior_points = (
                    compute_interior_Chebyshev_points_adaptive_2D(root, p)
                )
                self.boundary_points = (
                    compute_boundary_Gauss_points_adaptive_2D(root, q)
                )
            else:
                self.interior_points = (
                    compute_interior_Chebyshev_points_adaptive_3D(root, p)
                )
                self.boundary_points = (
                    compute_boundary_Gauss_points_adaptive_3D(root, q)
                )

    def to_interior_points(
        self,
        values: jax.Array,
        sample_points_x: jax.Array,
        sample_points_y: jax.Array,
        sample_points_z: jax.Array = None,
    ) -> jax.Array:
        n_x = sample_points_x.shape[0]
        n_y = sample_points_y.shape[0]
        # 2D vs 3D checking
        if isinstance(self.root, DiscretizationNode2D):
            bool_2D = True
            assert sample_points_z is None
            # Check shape of values
            assert values.shape == (n_x, n_y)
        else:
            bool_2D = False
            assert sample_points_z is not None
            n_z = sample_points_z.shape[0]
            # Check shape of values
            assert values.shape == (n_x, n_y, n_z)

        if bool_2D:
            if self.bool_uniform:
                leaves = get_all_uniform_leaves_2D(self.root, self.L)
            else:
                leaves = get_all_leaves(self.root)
            logging.debug("to_interior_points: leaves: %s", len(leaves))
            leaf_bounds = jnp.array(
                [
                    [leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax]
                    for leaf in leaves
                ]
            )
            logging.debug(
                "to_interior_points: leaf_bounds: %s", leaf_bounds.shape
            )

            return interp_to_hps_2D(
                leaf_bounds, values, self.p, sample_points_x, sample_points_y
            )
        else:
            # 3D case
            if self.bool_uniform:
                leaves = get_all_uniform_leaves_3D(self.root, self.L)
            else:
                leaves = get_all_leaves(self.root)

            leaf_bounds = jnp.array(
                [
                    [
                        leaf.xmin,
                        leaf.xmax,
                        leaf.ymin,
                        leaf.ymax,
                        leaf.zmin,
                        leaf.zmax,
                    ]
                    for leaf in leaves
                ]
            )
            return interp_to_hps_3D(
                leaf_bounds,
                values,
                self.p,
                sample_points_x,
                sample_points_y,
                sample_points_z,
            )

    def to_boundary_points(
        self,
        sample_points: jax.Array,
        sample_values: jax.Array,
        sample_f: Callable,
    ) -> jax.Array:
        raise NotImplementedError("to_boundary_points is not implemented yet.")

    def from_interior_points(
        self,
        samples: jax.Array,
        eval_points_x: jax.Array,
        eval_points_y: jax.Array,
        eval_points_z: jax.Array = None,
    ) -> jax.Array:
        # 2D vs 3D checking
        if isinstance(self.root, DiscretizationNode2D):
            bool_2D = True
            assert eval_points_z is None
        else:
            bool_2D = False
            assert eval_points_z is not None

        n_leaves, n_per_leaf, _ = self.interior_points.shape
        assert samples.shape == (n_leaves, n_per_leaf)

        if bool_2D:
            if self.bool_uniform:
                leaves = get_all_uniform_leaves_2D(self.root, self.L)
            else:
                leaves = get_all_leaves(self.root)
            return interp_from_hps_2D(
                leaves=leaves,
                p=self.p,
                f_evals=samples,
                x_vals=eval_points_x,
                y_vals=eval_points_y,
            )

        else:
            if self.bool_uniform:
                leaves = get_all_uniform_leaves_3D(self.root, self.L)
            else:
                leaves = get_all_leaves(self.root)
            return interp_from_hps_3D(
                leaves=leaves,
                p=self.p,
                f_evals=samples,
                x_vals=eval_points_x,
                y_vals=eval_points_y,
                z_vals=eval_points_z,
            )

    def get_adaptive_boundary_data_lst(
        self, f: Callable[[jax.Array], jax.Array]
    ) -> list[jax.Array]:
        if self.bool_2D:
            side_0_pts = self.boundary_points[
                self.boundary_points[:, 1] == self.root.ymin
            ]

            side_1_pts = self.boundary_points[
                self.boundary_points[:, 0] == self.root.xmax
            ]
            side_2_pts = self.boundary_points[
                self.boundary_points[:, 1] == self.root.ymax
            ]
            side_3_pts = self.boundary_points[
                self.boundary_points[:, 0] == self.root.xmin
            ]

            bdry_data_lst = [
                f(side_0_pts),
                f(side_1_pts),
                f(side_2_pts),
                f(side_3_pts),
            ]
        else:
            face_1_pts = self.boundary_points[
                self.boundary_points[:, 0] == self.root.xmin
            ]
            face_2_pts = self.boundary_points[
                self.boundary_points[:, 0] == self.root.xmax
            ]
            face_3_pts = self.boundary_points[
                self.boundary_points[:, 1] == self.root.ymin
            ]
            face_4_pts = self.boundary_points[
                self.boundary_points[:, 1] == self.root.ymax
            ]
            face_5_pts = self.boundary_points[
                self.boundary_points[:, 2] == self.root.zmin
            ]
            face_6_pts = self.boundary_points[
                self.boundary_points[:, 2] == self.root.zmax
            ]

            bdry_data_lst = [
                f(face_1_pts),
                f(face_2_pts),
                f(face_3_pts),
                f(face_4_pts),
                f(face_5_pts),
                f(face_6_pts),
            ]

        return bdry_data_lst
