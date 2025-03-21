import pytest

from hps.src.solver_obj import (
    create_solver_obj_2D,
    create_solver_obj_3D,
)
from hps.src.quadrature.trees import (
    Node,
    add_four_children,
    add_uniform_levels,
)
import jax.numpy as jnp


class Test_create_solver_obj_2D:
    def test_0(self) -> None:
        p = 14
        q = 12
        l = 4

        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )

        n_leaf_nodes = 4**l

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)
        assert t.leaf_cheby_points.shape == (n_leaf_nodes, p**2, 2)
        assert t.root_boundary_points.shape == (4 * (2**l) * q, 2)

    def test_1(self) -> None:
        """Checks l=0"""
        p = 14
        q = 12
        l = 0

        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )

        add_uniform_levels(root, l)

        n_leaf_nodes = 2**l

        t = create_solver_obj_2D(p, q, root)
        assert t.leaf_cheby_points.shape == (n_leaf_nodes, p**2, 2)
        assert t.root_boundary_points.shape == (4 * q, 2)

    def test_2(self) -> None:
        """Checks some of the precomputed operators make sense."""
        p = 3
        q = 12
        l = 0

        root = Node(
            xmin=-1.0,
            xmax=1.0,
            ymin=-1.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)

        pts = t.leaf_cheby_points.squeeze()
        y_sq = jnp.square(pts[..., 1])

        two_y = 2 * pts[..., 1]

        out = t.D_y @ y_sq

        print("test_5: out = ", out)
        print("test_5: two_y = ", two_y)

        assert jnp.allclose(out, two_y)

    def test_3(self) -> None:
        """Nonuniform mesh."""

        p = 14
        q = 12

        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        add_four_children(root)
        add_four_children(root.children[0])
        add_four_children(root.children[0].children[0])

        n_leaf_nodes = 10

        t = create_solver_obj_2D(p, q, root)
        assert t.leaf_cheby_points.shape == (n_leaf_nodes, p**2, 2)
        assert t.root_boundary_points.shape == (12 * q, 2)

    def test_4(self) -> None:
        """Checks sizes are correct when using ItI maps."""
        p = 14
        q = 12
        l = 2
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )

        # n_leaf_nodes = 4**l

        t = create_solver_obj_2D(
            p, q, root, uniform_levels=l, use_ItI=True, eta=4.0
        )

        assert t.I_P_0.shape == (4 * (p - 1), 4 * q)
        assert t.Q_I.shape == (4 * q, 4 * p)
        assert t.F.shape == (4 * (p - 1), p**2)
        assert t.G.shape == (4 * p, p**2)


class Test_create_solver_obj_3D:
    def test_0(self) -> None:
        p = 14
        q = 12
        l = 2

        root = Node(
            xmin=-1.0,
            xmax=1.0,
            ymin=-1.0,
            ymax=1.0,
            zmin=-1.0,
            zmax=1.0,
            depth=0,
        )

        t = create_solver_obj_3D(p, q, root, uniform_levels=l)
        assert t.leaf_cheby_points.shape == (8**l, p**3, 3)


if __name__ == "__main__":
    pytest.main()
