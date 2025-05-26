from jaxhps._discretization_tree import (
    DiscretizationNode2D,
    DiscretizationNode3D,
)
from jaxhps._discretization_tree_operations_2D import (
    add_four_children,
)
from jaxhps._discretization_tree_operations_3D import (
    add_eight_children,
)
from jaxhps._domain import Domain
import jax
import jax.numpy as jnp
import logging

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


class Test_interp_from_interior_points:
    def test_0(self, caplog) -> None:
        """Initializes a uniform 2D domain and checks the interp_from_interior_points method."""
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

        samples, pts = domain.interp_from_interior_points(
            f_samples, xvals, yvals
        )
        f_expected = f(pts)
        assert jnp.allclose(samples, f_expected)

    def test_1(self, caplog) -> None:
        """Initializes an adaptive 2D domain and checks the interp_from_interior_points method."""
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

        samples, pts = domain.interp_from_interior_points(
            f_samples, xvals, yvals
        )
        f_expected = f(pts)
        assert jnp.allclose(samples, f_expected)


class Test_interp_to_interior_points:
    def test_0(self, caplog) -> None:
        """Initializes a uniform 2D domain and checks the interp_to_interior_points method."""
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

        # Use different number of points to catch errors
        # confusing x and y axes.
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

        samples_on_hps = domain.interp_to_interior_points(
            values=f_samples, sample_points_x=xvals, sample_points_y=yvals
        )
        f_expected = f(domain.interior_points)

        assert samples_on_hps.shape == f_expected.shape

        # Uncomment to plot the computed and expected solutions.

        # plot_soln_from_cheby_nodes(
        #     cheby_nodes=domain.interior_points.reshape(
        #         -1, 2
        #     ),  # Flatten for plotting
        #     computed_soln=samples_on_hps.reshape(-1),  # Flatten for plotting
        #     expected_soln=f_expected.reshape(-1),  # Flatten for plotting
        #     corners=None,
        # )

        assert jnp.allclose(samples_on_hps, f_expected)

    def test_1(self, caplog) -> None:
        """Initializes a non-uniform 2D domain and checks the interp_to_interior_points method."""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4

        xmin = 0.0
        xmax = 1.0
        ymin = 0.0
        ymax = 1.0

        root = DiscretizationNode2D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        add_four_children(root, root=root, q=q)
        add_four_children(root.children[0], root=root, q=q)
        domain = Domain(p=p, q=q, root=root)

        # Use different number of points to catch errors
        # confusing x and y axes.
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

        samples_on_hps = domain.interp_to_interior_points(
            values=f_samples, sample_points_x=xvals, sample_points_y=yvals
        )
        f_expected = f(domain.interior_points)

        assert samples_on_hps.shape == f_expected.shape

        # Uncomment to plot the computed and expected solutions.

        # plot_soln_from_cheby_nodes(
        #     cheby_nodes=domain.interior_points.reshape(
        #         -1, 2
        #     ),  # Flatten for plotting
        #     computed_soln=samples_on_hps.reshape(-1),  # Flatten for plotting
        #     expected_soln=f_expected.reshape(-1),  # Flatten for plotting
        #     corners=None,
        # )

        assert jnp.allclose(samples_on_hps, f_expected)

    def test_2(self, caplog) -> None:
        """Makes sure low-order polynomial interp is exact in non-uniform 3D case."""
        caplog.set_level(logging.DEBUG)

        p = 6
        q = 4

        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
        zmin = -1.0
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
        domain = Domain(p=p, q=q, root=root)

        # Use different number of points to catch errors
        # confusing x and y axes.
        n_x = 10
        n_y = 11
        n_z = 7
        xvals = jnp.linspace(xmin, xmax, n_x)
        yvals = jnp.linspace(ymin, ymax, n_y)
        zvals = jnp.linspace(zmin, zmax, n_z)
        # yvals = jnp.flip(yvals)  # Flip yvals to match the expected orientation
        X, Y, Z = jnp.meshgrid(xvals, yvals, zvals, indexing="ij")
        logging.debug("X.shape: %s", X.shape)
        logging.debug("Y.shape: %s", Y.shape)
        logging.debug("Z.shape: %s", Z.shape)
        pts = jnp.concatenate(
            (
                jnp.expand_dims(X, 3),
                jnp.expand_dims(Y, 3),
                jnp.expand_dims(Z, 3),
            ),
            axis=3,
        )
        # pts = jnp.concatenate(
        #     (jnp.expand_dims(X, 2), jnp.expand_dims(Y, 2)), axis=2
        # )
        logging.debug("pts.shape: %s", pts.shape)

        def f(x: jax.Array) -> jax.Array:  # x^2 - 3y + z
            return x[..., 0] ** 2 - 3 * x[..., 1] + x[..., 2]

        f_samples = f(pts)

        samples_on_hps = domain.interp_to_interior_points(
            values=f_samples,
            sample_points_x=xvals,
            sample_points_y=yvals,
            sample_points_z=zvals,
        )
        f_expected = f(domain.interior_points)

        assert samples_on_hps.shape == f_expected.shape

        # Uncomment to plot the computed and expected solutions.

        # plot_soln_from_cheby_nodes(
        #     cheby_nodes=domain.interior_points.reshape(
        #         -1, 2
        #     ),  # Flatten for plotting
        #     computed_soln=samples_on_hps.reshape(-1),  # Flatten for plotting
        #     expected_soln=f_expected.reshape(-1),  # Flatten for plotting
        #     corners=None,
        # )

        assert jnp.allclose(samples_on_hps, f_expected)


class Test_init_from_adaptive_discretization:
    def test_0(self, caplog) -> None:
        """3D adaptive discretization test."""
        caplog.set_level(logging.DEBUG)

        p = 6
        q = 4
        tol = 1e-05  # Chosen so we get exactly 1 level of refinement. If you change this, also change expected shape.

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = sin(y) + x**2"""
            return jnp.sin(x[..., 1] ** 2) + x[..., 0] ** 2

        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
        )

        domain = Domain.from_adaptive_discretization(
            p=p, q=q, root=root, f=f, tol=tol
        )

        assert domain.interior_points.shape == (8, p**3, 3)
