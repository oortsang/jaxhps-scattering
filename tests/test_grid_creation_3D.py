import jax.numpy as jnp

from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._discretization_tree_operations_3D import add_eight_children
from jaxhps._grid_creation_3D import (
    bounds_for_oct_subdivision,
    vmapped_bounds_3D,
    compute_boundary_Gauss_points_adaptive_3D,
    compute_boundary_Gauss_points_uniform_3D,
    compute_interior_Chebyshev_points_adaptive_3D,
    compute_interior_Chebyshev_points_uniform_3D,
    bounds_to_gauss_face,
)
from jaxhps.quadrature import gauss_points


class Test_compute_interior_Chebyshev_points_uniform_3D:
    def test_0(self) -> None:
        """Test things work with uniform grid."""
        p = 8
        l = 3
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )

        x = compute_interior_Chebyshev_points_uniform_3D(root=root, L=l, p=p)
        n_leaves = 8**l
        assert x.shape == (n_leaves, p**3, 3)
        assert jnp.all(x[:, :, 0] >= 0)
        assert jnp.all(x[:, :, 0] <= 1)
        assert jnp.all(x[:, :, 1] >= 0)
        assert jnp.all(x[:, :, 1] <= 1)
        assert jnp.all(x[:, :, 2] >= 0)
        assert jnp.all(x[:, :, 2] <= 1)


class Test_compute_interior_Chebyshev_points_adaptive_3D:
    def test_0(self) -> None:
        """Tests non-uniform refinement"""
        p = 8
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )
        add_eight_children(root)
        add_eight_children(root.children[0])

        x = compute_interior_Chebyshev_points_adaptive_3D(root, p)
        n_leaves = 8 + 7
        assert x.shape == (n_leaves, p**3, 3)


class Test_bounds_for_oct_subdivision:
    def test_0(self) -> None:
        """Makes sure it returns the correct size."""
        corners = jnp.array(
            [0, 1, 0, 1, 0, 1],
        )
        out = bounds_for_oct_subdivision(corners)
        assert out.shape == (8, 6)

    def test_1(self) -> None:
        corners = jnp.array(
            [0, 1, 0, 1, 0, 1],
        )
        out = bounds_for_oct_subdivision(corners)

        for i in range(8):
            print("test_1: i:", i)
            print("test_1: out[i]", out[i])
            # Make sure it's a cube
            for j in range(3):
                assert jnp.all(out[i, 2 * j + 1] - out[i, 2 * j] == 0.5)
                assert jnp.all(out[i, 2 * j] <= out[i, 2 * j + 1])

    def test_2(self) -> None:
        xmin, ymin, zmin = 0, 0, 0
        xmax, ymax, zmax = 1, 1, 1
        corners = jnp.array([xmin, xmax, ymin, ymax, zmin, zmax])
        out = bounds_for_oct_subdivision(corners)

        # Corners for a and b should match in the y and z dimensions
        assert jnp.all(out[0, 2:] == out[1, 2:])

        # Corners for a and d should match in the x and z dimensions
        assert jnp.all(out[0, [0, 1, 4, 5]] == out[3, [0, 1, 4, 5]])

        # Corners for b and c should match in the x and z dimensions
        assert jnp.all(out[1, [0, 1, 4, 5]] == out[2, [0, 1, 4, 5]])


class Test_vmapped_bounds_3D:
    def test_0(self) -> None:
        corners_0 = jnp.array([[-1.0, 0.0, -1.0, 0.0, -1.0, 0.0]])

        print("test_0: corners_0.shape: ", corners_0.shape)

        corners_1 = vmapped_bounds_3D(corners_0)
        assert corners_1.shape == (1, 8, 6)

        corners_2 = vmapped_bounds_3D(corners_1.reshape(-1, 6))
        assert corners_2.shape == (8, 8, 6)


class Test_compute_boundary_Gauss_points_uniform_3D:
    def test_0(self) -> None:
        q = 3
        l = 2
        root = DiscretizationNode3D(
            xmin=-jnp.pi / 2,
            xmax=jnp.pi / 2,
            ymin=-jnp.pi / 2,
            ymax=jnp.pi / 2,
            zmin=-jnp.pi / 2,
            zmax=jnp.pi / 2,
        )

        out = compute_boundary_Gauss_points_uniform_3D(root, l, q)
        assert out.shape == (6 * 2 ** (2 * l) * q**2, 3)
        assert out.min() == -jnp.pi / 2
        assert out.max() == jnp.pi / 2


class Test_compute_boundary_Gauss_points_adaptive_3D:
    def test_0(self) -> None:
        """Test individual faces against the corners_to_gauss_face function when l=0"""
        q = 6
        bounds = jnp.array(
            [
                -jnp.pi / 2,
                jnp.pi / 2,
                -jnp.pi / 2,
                jnp.pi / 2,
                -jnp.pi / 2,
                jnp.pi / 2,
            ],
        )
        root = DiscretizationNode3D(
            xmin=-jnp.pi / 2,
            xmax=jnp.pi / 2,
            ymin=-jnp.pi / 2,
            ymax=jnp.pi / 2,
            zmin=-jnp.pi / 2,
            zmax=jnp.pi / 2,
            depth=0,
        )
        out = compute_boundary_Gauss_points_adaptive_3D(root, q)

        gauss_pts = gauss_points(q)

        # Test face 1 and 2 (yz)
        corners_12 = bounds[2:]
        pts_12_ref = bounds_to_gauss_face(corners_12, gauss_pts)
        print("test_1: ref", pts_12_ref)
        print("test_1: out", out[: q**2, 1:])
        assert jnp.allclose(out[: q**2, 1:], pts_12_ref)
        assert jnp.allclose(out[q**2 : 2 * q**2, 1:], pts_12_ref)

        # Test face 3 and 4 (xz)
        xz_idxes = jnp.array([0, 1, 4, 5])
        corners_34 = bounds[xz_idxes]
        pts_34_ref = bounds_to_gauss_face(corners_34, gauss_pts)
        assert jnp.allclose(out[2 * q**2 : 3 * q**2, [0, 2]], pts_34_ref)
        assert jnp.allclose(out[3 * q**2 : 4 * q**2, [0, 2]], pts_34_ref)

        # Test face 5 and 6 (xy)
        corners_56 = bounds[:4]
        pts_56_ref = bounds_to_gauss_face(corners_56, gauss_pts)
        assert jnp.allclose(out[4 * q**2 : 5 * q**2, :2], pts_56_ref)
        assert jnp.allclose(out[5 * q**2 : 6 * q**2, :2], pts_56_ref)
