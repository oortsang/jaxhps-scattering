from hahps._discretization_tree import (
    DiscretizationNode2D,
    DiscretizationNode3D,
)
from hahps._discretization_tree_operations_2D import (
    add_four_children,
)
from hahps._discretization_tree_operations_3D import (
    add_eight_children,
)
from hahps._domain import Domain
import jax
import jax.numpy as jnp
import logging
from hahps._utils import plot_soln_from_cheby_nodes

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Test_Domain_init:
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

    def test_2(self) -> None:
        """Tests Domain initialization in the 3D uniform case."""
        p = 6
        q = 4
        L = 2

        xmin = 0.0
        xmax = 1.0
        ymin = 0.0
        ymax = 1.0
        zmin = 0.0
        zmax = 1.0

        root = DiscretizationNode3D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
        )
        domain = Domain(p=p, q=q, root=root, L=L)
        n_leaves = 8**L
        n_gauss_pts = 6 * (q**2) * (4**L)
        assert domain.interior_points.shape == (n_leaves, p**3, 3)
        assert domain.boundary_points.shape == (n_gauss_pts, 3)

    def test_3(self) -> None:
        """Tests Domain initialization in the 3D adaptive case."""

        p = 6
        q = 4

        xmin = 0.0
        xmax = 1.0
        ymin = 0.0
        ymax = 1.0
        zmin = 0.0
        zmax = 1.0

        root = DiscretizationNode3D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
        )
        add_eight_children(root, root=root, q=q)
        add_eight_children(root.children[0], root=root, q=q)
        add_eight_children(root.children[0].children[0], root=root, q=q)
        n_leaves = 8 + 7 + 7
        n_gauss_panels = 10 + 10 + 10 + 4 + 4 + 4

        domain = Domain(p=p, q=q, root=root)
        assert domain.interior_points.shape == (n_leaves, p**3, 3)
        assert domain.boundary_points.shape == (n_gauss_panels * (q**2), 3)


class Test_from_interior_points:
    def test_0(self, caplog) -> None:
        """Initializes a uniform 2D domain and checks the from_interior_points method."""
        p = 6
        q = 4
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

        def f(x: jax.Array) -> jax.Array:  # x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        f_samples = f(domain.interior_points)
        n_x = 3
        xvals = jnp.linspace(xmin, xmax, n_x)
        yvals = jnp.linspace(ymin, ymax, n_x)

        samples, pts = domain.from_interior_points(f_samples, xvals, yvals)
        f_expected = f(pts)
        assert jnp.allclose(samples, f_expected)

    def test_1(self, caplog) -> None:
        """Initializes an adaptive 2D domain and checks the from_interior_points method."""
        p = 6
        q = 4
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
        add_four_children(root, root=root, q=q)
        add_four_children(root.children[0], root=root, q=q)
        domain = Domain(p=p, q=q, root=root, L=L)

        def f(x: jax.Array) -> jax.Array:  # x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        f_samples = f(domain.interior_points)
        n_x = 3
        xvals = jnp.linspace(xmin, xmax, n_x)
        yvals = jnp.linspace(ymin, ymax, n_x)

        samples, pts = domain.from_interior_points(f_samples, xvals, yvals)
        f_expected = f(pts)
        assert jnp.allclose(samples, f_expected)


class Test_to_interior_points:
    def test_0(self, caplog) -> None:
        """Initializes a uniform 2D domain and checks the to_interior_points method."""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
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

        n_x = 10
        n_y = 11
        xvals = jnp.linspace(xmin, xmax, n_x)
        yvals = jnp.linspace(ymin, ymax, n_y)
        # yvals = jnp.flip(yvals)  # Flip yvals to match the expected orientation
        X, Y = jnp.meshgrid(xvals, yvals, indexing="ij")
        logging.debug("X.shape: %s", X.shape)
        logging.debug("Y.shape: %s", Y.shape)
        pts = jnp.concatenate(
            (jnp.expand_dims(X, 2), jnp.expand_dims(Y, 2)), axis=2
        )
        logging.debug("pts.shape: %s", pts.shape)

        def f(x: jax.Array) -> jax.Array:  # x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        f_samples = f(pts)

        samples_on_hps = domain.to_interior_points(
            values=f_samples, sample_points_x=xvals, sample_points_y=yvals
        )
        f_expected = f(domain.interior_points)

        assert samples_on_hps.shape == f_expected.shape

        plot_soln_from_cheby_nodes(
            cheby_nodes=domain.interior_points.reshape(
                -1, 2
            ),  # Flatten for plotting
            computed_soln=samples_on_hps.reshape(-1),  # Flatten for plotting
            expected_soln=f_expected.reshape(-1),  # Flatten for plotting
            corners=None,
        )

        # Plot the two values for comparison
        assert jnp.allclose(samples_on_hps, f_expected)

    # def test_1(self, caplog) -> None:
    #     """Initializes an adaptive 2D domain and checks the from_interior_points method."""
    #     p = 6
    #     q = 4
    #     L = 2

    #     xmin = -1.0
    #     xmax = 1.0
    #     ymin = -1.0
    #     ymax = 1.0

    #     root = DiscretizationNode2D(
    #         xmin=xmin,
    #         xmax=xmax,
    #         ymin=ymin,
    #         ymax=ymax,
    #     )
    #     add_four_children(root, root=root, q=q)
    #     add_four_children(root.children[0], root=root, q=q)
    #     domain = Domain(p=p, q=q, root=root, L=L)

    #     def f(x: jax.Array) -> jax.Array:  # x^2 - 3y
    #         return x[..., 0] ** 2 - 3 * x[..., 1]

    #     f_samples = f(domain.interior_points)
    #     n_x = 3
    #     xvals = jnp.linspace(xmin, xmax, n_x)
    #     yvals = jnp.linspace(ymin, ymax, n_x)

    #     samples, pts = domain.from_interior_points(f_samples, xvals, yvals)
    #     f_expected = f(pts)
    #     assert jnp.allclose(samples, f_expected)
