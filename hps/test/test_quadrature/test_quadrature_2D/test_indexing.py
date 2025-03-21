import jax.numpy as jnp

from hps.src.quadrature.quad_2D.indexing import (
    indexing_for_refinement_operator,
)
from hps.src.quadrature.quad_2D.grid_creation import (
    get_all_leaf_2d_cheby_points_uniform_refinement,
)
from hps.src.quadrature.quadrature_utils import chebyshev_points, affine_transform


class Test_indexing_for_refinement_operator:
    def test_0(self) -> None:
        p = 4
        row_idxes, col_idxes = indexing_for_refinement_operator(p)

        assert col_idxes.shape == (p**2,)
        assert row_idxes.shape == (4 * p**2,)
        assert jnp.unique(row_idxes).shape == (4 * p**2,)

    def test_1(self) -> None:
        p = 3
        cheby_pts_1d = chebyshev_points(p)[0]
        cheby_pts_refined = jnp.concatenate(
            [
                affine_transform(cheby_pts_1d, jnp.array([-1, 0])),
                affine_transform(cheby_pts_1d, jnp.array([0, 1])),
            ]
        )

        row_x, row_y = jnp.meshgrid(
            cheby_pts_refined, jnp.flipud(cheby_pts_refined), indexing="ij"
        )
        row_pts = jnp.stack([row_x.flatten(), row_y.flatten()], axis=-1)
        print("test_1: row_pts", row_pts)

        north = 1
        south = -1
        east = 1
        west = -1
        corners = jnp.array(
            [[west, south], [east, south], [east, north], [west, north]],
            dtype=jnp.float64,
        )

        pts_1 = get_all_leaf_2d_cheby_points_uniform_refinement(p, 1, corners).reshape(
            (-1, 2)
        )

        row_idxes, col_idxes = indexing_for_refinement_operator(p)

        row_pts_rearranged = row_pts[row_idxes]
        print("test_1: row_pts_rearranged", row_pts_rearranged)

        pts_1_trunc = pts_1[: row_pts_rearranged.shape[0]]
        print("test_1: pts_1", pts_1_trunc)

        assert jnp.allclose(row_pts_rearranged, pts_1_trunc)

        assert jnp.allclose(row_pts_rearranged, pts_1)
