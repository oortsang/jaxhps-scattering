import numpy as np
import jax.numpy as jnp
import pytest

from hps.src.quadrature.quadrature_utils import chebyshev_points
from hps.src.quadrature.quad_3D.grid_creation import (
    corners_to_cheby_points_lst,
    corners_to_gauss_points_lst,
    corners_to_gauss_face,
    _corners_for_oct_subdivision,
    get_all_boundary_gauss_legendre_points,
    get_all_leaf_3d_cheby_points,
)
from hps.src.quadrature.quad_3D.indexing import rearrange_indices_ext_int
from hps.src.quadrature.trees import (
    Node,
    add_eight_children,
    add_uniform_levels,
)


class Test_corners_to_cheby_points_lst:
    def test_0(self) -> None:
        p = 13
        pts = chebyshev_points(p)[0]

        corners = jnp.array(
            [
                [0, 0, 0],
                [1, 1, 1],
            ]
        )

        out = corners_to_cheby_points_lst(corners, pts)
        assert out.shape == (p**3, 3)

    def test_1(self) -> None:
        """Checks that the (p-2)^3 interior points are listed last."""

        p = 4
        pts = chebyshev_points(p)[0]

        corners = jnp.array(
            [
                [-1, -1, -1],
                [1, 1, 1],
            ]
        )
        out = corners_to_cheby_points_lst(corners, pts)

        idxes_out = rearrange_indices_ext_int(p)
        print("test_1: out", out)
        print("test_1: idxes_out shape", idxes_out.shape)
        n_ext_pts = p**3 - (p - 2) ** 3
        int_pts = out[n_ext_pts:]
        assert jnp.all(jnp.abs(int_pts) < 1)


class Test_corners_to_gauss_face:
    def test_0(self) -> None:
        q = 13
        gauss_pts = np.polynomial.legendre.leggauss(q)[0]
        corners = jnp.array([[-1, -1], [1, 1]])
        pts_lst = corners_to_gauss_face(corners, gauss_pts)
        assert pts_lst.shape == (q**2, 2)


class Test_corners_to_gauss_points_lst:
    def test_0(self) -> None:
        q = 4
        corners = jnp.array(
            [
                [-np.pi / 2, -np.pi / 2, -np.pi / 2],
                [np.pi / 2, np.pi / 2, np.pi / 2],
            ]
        )
        out = corners_to_gauss_points_lst(q, corners)
        assert out.shape == (6 * q**2, 3)
        assert out.min() == -np.pi / 2
        assert out.max() == np.pi / 2


class Test__corners_for_oct_subdivision:
    def test_0(self) -> None:
        """Makes sure it returns the correct size."""
        corners = jnp.array(
            [
                [0, 0, 0],
                [1, 1, 1],
            ]
        )
        out = _corners_for_oct_subdivision(corners)
        assert out.shape == (8, 2, 3)

    def test_1(self) -> None:
        corners = jnp.array(
            [
                [0, 0, 0],
                [1, 1, 1],
            ]
        )
        out = _corners_for_oct_subdivision(corners)

        for i in range(8):
            print("test_1: i:", i)
            print("test_1: out[i]", out[i])
            assert jnp.all(out[i, 0] <= out[i, 1])
            # Make sure it's a cube
            assert jnp.all(out[i, 1] - out[i, 0] == 0.5)

    def test_2(self) -> None:
        xmin, ymin, zmin = 0, 0, 0
        xmax, ymax, zmax = 1, 1, 1
        # xmid = (xmin + xmax) / 2
        # ymid = (ymin + ymax) / 2
        # zmid = (zmin + zmax) / 2
        corners = jnp.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        out = _corners_for_oct_subdivision(corners)

        # Corners for a and b should match in the y and z dimensions
        assert jnp.all(out[0, :, 1:] == out[1, :, 1:])

        # Corners for a and d should match in the x and z dimensions
        assert jnp.all(out[0, :, [0, 2]] == out[3, :, [0, 2]])

        # Corners for b and c should match in the x and z dimensions
        assert jnp.all(out[1, :, [0, 2]] == out[2, :, [0, 2]])


class Test_get_all_leaf_3d_cheby_points:
    def test_0(self) -> None:
        """Test things work with uniform grid."""
        p = 8
        l = 3
        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )

        add_eight_children(root)
        for c in root.children:
            add_eight_children(c)
            for gc in c.children:
                add_eight_children(gc)

        x = get_all_leaf_3d_cheby_points(p, root)
        n_leaves = 8**l
        assert x.shape == (n_leaves, p**3, 3)
        assert jnp.all(x[:, :, 0] >= 0)
        assert jnp.all(x[:, :, 0] <= 1)
        assert jnp.all(x[:, :, 1] >= 0)
        assert jnp.all(x[:, :, 1] <= 1)
        assert jnp.all(x[:, :, 2] >= 0)
        assert jnp.all(x[:, :, 2] <= 1)

    def test_1(self) -> None:
        """Tests non-uniform refinement"""
        p = 8
        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        add_eight_children(root)
        add_eight_children(root.children[0])

        x = get_all_leaf_3d_cheby_points(p, root)
        n_leaves = 8 + 7
        assert x.shape == (n_leaves, p**3, 3)


class Test_get_all_boundary_gauss_legendre_points:
    def test_0(self) -> None:
        q = 3
        l = 2
        root = Node(
            xmin=-jnp.pi / 2,
            xmax=jnp.pi / 2,
            ymin=-jnp.pi / 2,
            ymax=jnp.pi / 2,
            zmin=-jnp.pi / 2,
            zmax=jnp.pi / 2,
            depth=0,
        )
        add_uniform_levels(root, l)
        out = get_all_boundary_gauss_legendre_points(q, root)
        assert out.shape == (6 * 2 ** (2 * l) * q**2, 3)
        assert out.min() == -np.pi / 2
        assert out.max() == np.pi / 2

    def test_1(self) -> None:
        """Test individual faces against the corners_to_gauss_face function when l=0"""
        q = 6
        corners = jnp.array(
            [
                [-np.pi / 2, -np.pi / 2, -np.pi / 2],
                [np.pi / 2, np.pi / 2, np.pi / 2],
            ]
        )
        root = Node(
            xmin=-jnp.pi / 2,
            xmax=jnp.pi / 2,
            ymin=-jnp.pi / 2,
            ymax=jnp.pi / 2,
            zmin=-jnp.pi / 2,
            zmax=jnp.pi / 2,
            depth=0,
        )
        out = get_all_boundary_gauss_legendre_points(q, root)

        gauss_pts = np.polynomial.legendre.leggauss(q)[0]

        # Test face 1 and 2
        corners_12 = corners[:, 1:]
        pts_12_ref = corners_to_gauss_face(corners_12, gauss_pts)
        print("test_1: ref", pts_12_ref)
        print("test_1: out", out[: q**2, 1:])
        assert jnp.allclose(out[: q**2, 1:], pts_12_ref)
        assert jnp.allclose(out[q**2 : 2 * q**2, 1:], pts_12_ref)

        # Test face 3 and 4
        corners_34 = corners[:, [0, 2]]
        pts_34_ref = corners_to_gauss_face(corners_34, gauss_pts)
        assert jnp.allclose(out[2 * q**2 : 3 * q**2, [0, 2]], pts_34_ref)
        assert jnp.allclose(out[3 * q**2 : 4 * q**2, [0, 2]], pts_34_ref)

        # Test face 5 and 6
        corners_56 = corners[:, :2]
        pts_56_ref = corners_to_gauss_face(corners_56, gauss_pts)
        assert jnp.allclose(out[4 * q**2 : 5 * q**2, :2], pts_56_ref)
        assert jnp.allclose(out[5 * q**2 : 6 * q**2, :2], pts_56_ref)


if __name__ == "__main__":
    pytest.main()
