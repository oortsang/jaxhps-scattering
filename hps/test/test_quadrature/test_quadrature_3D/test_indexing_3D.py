import pytest
import numpy as np
import jax.numpy as jnp

from hps.src.quadrature.quad_3D.indexing import (
    rearrange_indices_ext_int,
    get_face_1_idxes,
    get_face_2_idxes,
    get_face_3_idxes,
    get_face_4_idxes,
    get_face_5_idxes,
    get_face_6_idxes,
    into_column_first_order,
    indexing_for_refinement_operator,
)

from hps.src.quadrature.quad_3D.grid_creation import (
    corners_to_cheby_points_lst,
    get_all_leaf_3d_cheby_points,
    affine_transform,
)
from hps.src.quadrature.quadrature_utils import chebyshev_points
from hps.src.quadrature.trees import (
    Node,
    add_uniform_levels,
)


class Test_rearrange_indices_ext_int:
    def test_0(self) -> None:
        p = 4
        out = rearrange_indices_ext_int(p)
        assert out.shape == (p**3,), out.shape

    def test_1(self) -> None:
        p = 5
        out = rearrange_indices_ext_int(p)

        out_unique = np.unique(out)
        assert out_unique.shape == (p**3,), out_unique.shape


class Test_get_face_1_idxes:
    def test_0(self) -> None:
        p = 4
        out = get_face_1_idxes(p)
        assert out.shape == (p**2,), out.shape

        assert np.unique(out).shape == (p**2,), np.unique(out).shape

    def test_1(self) -> None:
        p = 5
        pts = chebyshev_points(p)[0]
        corners = jnp.array([[-1, -1, -1], [1, 1, 1]])
        cheby_points = corners_to_cheby_points_lst(corners, pts)

        out = get_face_1_idxes(p)
        face_1_pts = cheby_points[out]

        assert jnp.all(face_1_pts[:, 0] == -1), face_1_pts


class Test_get_face_2_idxes:
    def test_0(self) -> None:
        """Check the function returns the correct number of indices and the indices are unique."""
        p = 4
        out = get_face_2_idxes(p)
        assert out.shape == (p**2,), out.shape

        assert np.unique(out).shape == (p**2,), np.unique(out).shape

    def test_1(self) -> None:
        """Check the function returns indices on the correct side of the cube."""
        p = 5
        pts = chebyshev_points(p)[0]
        corners = jnp.array([[-1, -1, -1], [1, 1, 1]])
        cheby_points = corners_to_cheby_points_lst(corners, pts)
        out = get_face_2_idxes(p)
        face_2_pts = cheby_points[out]
        assert jnp.all(face_2_pts[:, 0] == 1), face_2_pts


class Test_get_face_3_idxes:
    def test_0(self) -> None:
        """Check the function returns the correct number of indices and the indices are unique."""
        p = 4
        out = get_face_3_idxes(p)
        assert out.shape == (p**2,), out.shape

        assert np.unique(out).shape == (p**2,), np.unique(out).shape

    def test_1(self) -> None:
        """Check the function returns indices on the correct side of the cube."""
        p = 5
        pts = chebyshev_points(p)[0]
        corners = jnp.array([[-1, -1, -1], [1, 1, 1]])
        cheby_points = corners_to_cheby_points_lst(corners, pts)
        out = get_face_3_idxes(p)
        face_3_pts = cheby_points[out]
        assert jnp.all(face_3_pts[:, 1] == -1), face_3_pts


class Test_get_face_4_idxes:
    def test_0(self) -> None:
        """Check the function returns the correct number of indices and the indices are unique."""
        p = 4
        out = get_face_4_idxes(p)
        assert out.shape == (p**2,), out.shape

        assert np.unique(out).shape == (p**2,), np.unique(out).shape

    def test_1(self) -> None:
        """Check the function returns indices on the correct side of the cube."""
        p = 5
        pts = chebyshev_points(p)[0]
        corners = jnp.array([[-1, -1, -1], [1, 1, 1]])
        cheby_points = corners_to_cheby_points_lst(corners, pts)
        out = get_face_4_idxes(p)
        face_4_pts = cheby_points[out]
        assert jnp.all(face_4_pts[:, 1] == 1), face_4_pts


class Test_get_face_5_idxes:
    def test_0(self) -> None:
        p = 4
        out = get_face_5_idxes(p)
        assert out.shape == (p**2,), out.shape

        assert np.unique(out).shape == (p**2,), np.unique(out).shape

    def test_1(self) -> None:
        """Check the function returns indices on the correct side of the cube."""
        p = 5
        pts = chebyshev_points(p)[0]
        corners = jnp.array([[-1, -1, -1], [1, 1, 1]])
        cheby_points = corners_to_cheby_points_lst(corners, pts)
        out = get_face_5_idxes(p)
        face_5_pts = cheby_points[out]
        assert jnp.all(face_5_pts[:, 2] == -1), face_5_pts


class Test_get_face_6_idxes:
    def test_0(self) -> None:
        p = 4
        out = get_face_6_idxes(p)
        assert out.shape == (p**2,), out.shape

        assert np.unique(out).shape == (p**2,), np.unique(out).shape

    def test_1(self) -> None:
        """Check the function returns indices on the correct side of the cube."""
        p = 4
        pts = chebyshev_points(p)[0]
        corners = jnp.array([[-1, -1, -1], [1, 1, 1]])
        cheby_points = corners_to_cheby_points_lst(corners, pts)
        out = get_face_6_idxes(p)
        print("test_1: out = ", out)
        face_6_pts = cheby_points[out]
        assert jnp.all(face_6_pts[:, 2] == 1), face_6_pts


class Test_into_column_first_order:
    def test_0(self) -> None:
        # Checks outputs are correct shapes.
        q = 2
        idxes_a = np.arange(q**2)
        idxes_b = q**2 + np.arange(q**2)
        idxes_c = 2 * q**2 + np.arange(q**2)
        idxes_d = 3 * q**2 + np.arange(q**2)
        print("test_0: idxes_a = ", idxes_a)
        print("test_0: idxes_b = ", idxes_b)
        print("test_0: idxes_c = ", idxes_c)
        print("test_0: idxes_d = ", idxes_d)

        q_idxes = np.arange(q).astype(int)

        out = into_column_first_order(
            q_idxes, idxes_a, idxes_b, idxes_c, idxes_d
        )
        print("test_0: out = ", out)
        assert out.shape == (4 * q**2,)
        assert np.unique(out).shape == (4 * q**2,)


class Test_indexing_for_refinement_operator:
    def test_0(self) -> None:
        p = 4
        row_idxes, col_idxes = indexing_for_refinement_operator(p)

        assert col_idxes.shape == (p**3,)
        assert row_idxes.shape == (8 * p**3,)

    def test_1(self) -> None:
        p = 2
        cheby_pts_1d = chebyshev_points(p)[0]
        cheby_pts_refined = jnp.concatenate(
            [
                affine_transform(cheby_pts_1d, jnp.array([-1, 0])),
                affine_transform(cheby_pts_1d, jnp.array([0, 1])),
            ]
        )

        col_x, col_y, col_z = jnp.meshgrid(
            cheby_pts_refined,
            cheby_pts_refined,
            cheby_pts_refined,
            indexing="ij",
        )
        col_pts = jnp.stack([col_x, col_y, col_z], axis=-1).reshape(-1, 3)
        print("test_1: col_pts", col_pts)

        root = Node(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, depth=0)
        add_uniform_levels(root, 1)

        # These are eight copies of the Chebyshev points
        pts_1 = get_all_leaf_3d_cheby_points(p, root).reshape((-1, 3))

        row_idxes, col_idxes = indexing_for_refinement_operator(p)

        col_pts_rearranged = col_pts[row_idxes]
        print("test_1: col_pts_rearranged", col_pts_rearranged)

        pts_1_trunc = pts_1[: col_pts_rearranged.shape[0]]
        print("test_1: pts_1", pts_1_trunc)

        assert jnp.allclose(col_pts_rearranged, pts_1_trunc)

        assert jnp.allclose(col_pts_rearranged, pts_1)


if __name__ == "__main__":
    pytest.main()
