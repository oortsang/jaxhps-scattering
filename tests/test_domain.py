from hahps._discretization_tree import DiscretizationNode2D
from hahps._discretization_tree_operations_2D import (
    add_four_children,
)
from hahps._domain import Domain


class Test_Domain:
    def test_0(self) -> None:
        """Tests Domain initialization in the uniform 2D case."""
        p = 16
        q = 14
        L = 2
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0

        root = DiscretizationNode2D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        domain = Domain(p=p, q=q, root=root, L=L)

        n_leaves = 4**L
        n_gauss_pts = 4 * q * (2**L)

        assert domain.interior_points.shape == (n_leaves, p**2, 2)
        assert domain.boundary_points.shape == (n_gauss_pts, 2)

    def test_1(self) -> None:
        """Tests Domain initialization in the adaptive 2D case."""
        p = 6
        q = 4

        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0

        root = DiscretizationNode2D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        add_four_children(root, root=root, q=q)
        add_four_children(root.children[0], root=root, q=q)

        n_leaves = 7
        n_gauss_panels = 10

        domain = Domain(p=p, q=q, root=root)
        assert domain.interior_points.shape == (n_leaves, p**2, 2)
        assert domain.boundary_points.shape == (n_gauss_panels * q, 2)
