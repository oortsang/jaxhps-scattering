import numpy as np
import jax.numpy as jnp
import jax
from jaxhps.local_solve._adaptive_2D_DtN import (
    _prep_nonuniform_refinement_diff_operators_2D,
    vmapped_prep_nonuniform_refinement_diff_operators_2D,
)
from jaxhps._precompute_operators_2D import precompute_diff_operators_2D
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._discretization_tree_operations_2D import add_four_children
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem


class Test__prep_nonuniform_refinement_diff_operators_2D:
    def test_0(self) -> None:
        """Check accuracy"""
        p = 3
        q = 6
        half_side_len = 0.25

        # First precompute the diff operators without the sidelen scaling. This is how things
        # work in the local solve stage code
        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators_2D(p, 1.0)
        coeffs_arr = jnp.ones((2, p**2))
        diff_ops = jnp.stack([d_xx, d_xy, d_yy, d_x, d_y, jnp.eye(p**2)])

        coeffs_arr = coeffs_arr.at[0].set(4 * coeffs_arr[0])
        coeffs_arr = coeffs_arr.at[1].set(3 * coeffs_arr[1])
        which_coeffs = jnp.array([True, False, True, False, False, False])
        out, _ = _prep_nonuniform_refinement_diff_operators_2D(
            2 * half_side_len, coeffs_arr, which_coeffs, diff_ops, p, q
        )

        # Next, precompute diff operators with the correct sidelen scaling.
        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators_2D(
            p, half_side_len=half_side_len
        )
        expected_out = 4 * d_xx + 3 * d_yy

        print("test_1: out = ", out)
        print("test_1: expected_out = ", expected_out)
        assert jnp.allclose(out, expected_out)

    def test_1(self) -> None:
        """Checks accuracy with non-constant coefficients"""
        p = 3
        q = 6
        half_side_len = 0.25
        # First precompute the diff operators without the sidelen scaling. This is how things
        # work in the local solve stage code
        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators_2D(p, 1.0)
        coeffs_arr = np.random.normal(size=(2, p**2))
        diff_ops = jnp.stack([d_xx, d_xy, d_yy, d_x, d_y, jnp.eye(p**2)])
        which_coeffs = jnp.array([False, True, False, True, False, False])
        out, _ = _prep_nonuniform_refinement_diff_operators_2D(
            2 * half_side_len, coeffs_arr, which_coeffs, diff_ops, p, q
        )

        # Next, precompute diff operators with the correct sidelen scaling.
        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators_2D(
            p, half_side_len=half_side_len
        )
        expected_out = (
            jnp.diag(coeffs_arr[0]) @ d_xy + jnp.diag(coeffs_arr[1]) @ d_x
        )

        print("test_2: out = ", out)
        print("test_2: expected_out = ", expected_out)
        assert jnp.allclose(out, expected_out)

    def test_2(self) -> None:
        """Want to implement the operator
        3x * dxx + 4y * dyy

        when this is applied to f(x,y) = x^2 + y^2,
        we should get g(x,y) = 3x * 2 + 4y * 2 = 6x + 8y
        """

        p = 14
        q = 12
        l = 2

        sidelen = 1 / (2**l)

        node = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            depth=0,
        )
        add_four_children(node, root=node, q=q)
        for c in node.children:
            add_four_children(c, root=node, q=q)

        domain = Domain(
            p=p,
            q=q,
            root=node,
        )
        dxx_coeffs = 3 * domain.interior_points[..., 0]
        dyy_coeffs = 4 * domain.interior_points[..., 1]
        coeffs_arr = jnp.stack([dxx_coeffs, dyy_coeffs])
        print("test_3: coeffs_arr = ", coeffs_arr.shape)

        pde_problem = PDEProblem(
            domain,
            source=dxx_coeffs,
            D_xx_coefficients=dxx_coeffs,
            D_yy_coefficients=dyy_coeffs,
        )

        stacked_diff_operators = jnp.stack(
            [
                pde_problem.D_xx,
                pde_problem.D_xy,
                pde_problem.D_yy,
                pde_problem.D_x,
                pde_problem.D_y,
                jnp.eye(p**2, dtype=jnp.float64),
            ]
        )
        which_coeffs = jnp.array([True, False, True, False, False, False])

        f = (
            domain.interior_points[..., 0] ** 2
            + domain.interior_points[..., 1] ** 2
        )

        expected_g = (
            6 * domain.interior_points[..., 0]
            + 8 * domain.interior_points[..., 1]
        )

        # Loop over all of the nodes and check that the operator is applied correctly
        for i in range(domain.interior_points.shape[0]):
            op_i, _ = _prep_nonuniform_refinement_diff_operators_2D(
                sidelen,
                coeffs_arr[:, i],
                which_coeffs,
                stacked_diff_operators,
                p,
                q,
            )
            print("test_3: i = ", i)
            print("test_3: op_i norm = ", jnp.linalg.norm(op_i))
            prod = op_i @ f[i]
            print("test_3: prod nrm = ", jnp.linalg.norm(prod))
            print(
                "test_3: expected_g[i] nrm = ", jnp.linalg.norm(expected_g[i])
            )
            # plot_soln_from_cheby_nodes(
            #     domain.interior_points[i], None, prod, expected_g[i]
            # )
            assert jnp.allclose(op_i @ f[i], expected_g[i])

    def test_3(self) -> None:
        """Want to make sure Helmholtz operator looks good."""
        p = 20
        q = 18
        l = 2
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2
        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north, depth=0
        )

        add_four_children(root, root=root, q=q)
        for c in root.children:
            add_four_children(c, root=root, q=q)
        domain = Domain(p=p, q=q, root=root)
        side_len = jnp.pi / (2**l)
        k = jnp.pi

        d_xx_coeffs = jnp.ones_like(domain.interior_points[..., 0])
        d_yy_coeffs = jnp.ones_like(domain.interior_points[..., 0])
        i_coeffs = (k**2) * jnp.ones_like(domain.interior_points[..., 0])

        t = PDEProblem(
            domain=domain,
            source=i_coeffs,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            I_coefficients=i_coeffs,
        )

        coeffs_arr = jnp.stack([d_xx_coeffs, d_yy_coeffs, i_coeffs])
        which_coeffs = jnp.array([True, False, True, False, False, True])
        stacked_diff_operators = jnp.stack(
            [
                t.D_xx,
                t.D_xy,
                t.D_yy,
                t.D_x,
                t.D_y,
                jnp.eye(p**2, dtype=jnp.float64),
            ]
        )

        # Plane wave is e^{ikx}
        f = jnp.exp(1j * k * domain.interior_points[..., 0])

        for i in range(domain.interior_points.shape[0]):
            print("test_4: i = ", i)
            op_i, _ = _prep_nonuniform_refinement_diff_operators_2D(
                side_len,
                coeffs_arr[:, i],
                which_coeffs,
                stacked_diff_operators,
                p,
                q,
            )
            resid_i = op_i @ f[i]
            print("test_4: resid_i max = ", jnp.max(jnp.abs(resid_i)))
            assert jnp.allclose(op_i @ f[i], jnp.zeros_like(f[i]))
        jax.clear_caches()


class Test_vmapped_prep_nonuniform_refinement_diff_operators_2D:
    def test_0(self) -> None:
        """Tests that the function returns without error"""

        p = 4
        q = 2
        n_cheby_pts = p**2
        n_leaves = 13
        n_diff_terms = 6
        # n_cheby_bdry = 4 * (p - 1)
        n_gauss_bdry = 4 * q

        sidelens = np.random.normal(size=(n_leaves,))
        diff_ops_2D = np.random.normal(
            size=(n_diff_terms, n_cheby_pts, n_cheby_pts)
        )
        coeffs_arr = np.random.normal(size=(3, n_leaves, n_cheby_pts))
        which_coeffs = np.array([True, True, True, False, False, False])

        diff_operators, Q_Ds = (
            vmapped_prep_nonuniform_refinement_diff_operators_2D(
                sidelens, coeffs_arr, which_coeffs, diff_ops_2D, p, q
            )
        )

        assert diff_operators.shape == (n_leaves, n_cheby_pts, n_cheby_pts)
        assert Q_Ds.shape == (n_leaves, n_gauss_bdry, n_cheby_pts)
        jax.clear_caches()
