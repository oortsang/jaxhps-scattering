import numpy as np
import jax.numpy as jnp
import jax
from jaxhps._discretization_tree import (
    DiscretizationNode2D,
    get_all_leaves,
)
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
from jaxhps._discretization_tree_operations_2D import add_four_children
from jaxhps.merge._adaptive_2D_DtN import (
    merge_stage_adaptive_2D_DtN,
)
from jaxhps.local_solve._adaptive_2D_DtN import (
    local_solve_stage_adaptive_2D_DtN,
)
import logging

from jaxhps.down_pass._adaptive_2D_DtN import (
    down_pass_adaptive_2D_DtN,
    _propogate_down_quad,
)


class Test_down_pass_adaptive_2D_DtN:
    def test_0(self, caplog) -> None:
        """2D uniform case"""
        caplog.set_level(logging.DEBUG)
        p = 4
        q = 2
        l = 3
        num_leaves = 4**l

        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )

        # Manually add l=3 levels.
        add_four_children(root, root=root, q=q)
        for c in root.children:
            add_four_children(c, root=root, q=q)
            for gc in c.children:
                add_four_children(gc, root=root, q=q)
        logging.debug("test_0: root.n_0: %s", root.n_0)
        logging.debug("test_0: root.n_1: %s", root.n_1)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))

        # Create domain and PDE problem objects.
        domain = Domain(p=p, q=q, root=root)
        t = PDEProblem(
            domain=domain, source=source_term, D_xx_coefficients=d_xx_coeffs
        )

        Y_arr, T_arr, v_arr, h_arr = local_solve_stage_adaptive_2D_DtN(t)

        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.domain.root)):
            leaf.data.T = T_arr[i]
            leaf.data.h = h_arr[i]
            leaf.data.v = v_arr[i]
            leaf.data.Y = Y_arr[i]

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        logging.debug("test_0: T_arr.shape = %s", T_arr.shape)
        logging.debug("test_0: h_arr.shape = %s", h_arr.shape)
        merge_stage_adaptive_2D_DtN(t)

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = domain.get_adaptive_boundary_data_lst(f)

        leaf_solns = down_pass_adaptive_2D_DtN(
            pde_problem=t, boundary_data=boundary_data_lst
        )
        assert leaf_solns.shape == (num_leaves, p**2)

    def test_1(self, caplog) -> None:
        """2D Non-Uniform case."""
        caplog.set_level(logging.DEBUG)

        p = 4
        q = 2
        num_leaves = 7

        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(root, root=root, q=q)
        add_four_children(root.children[0], root=root, q=q)

        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        # Create domain and PDE problem objects.
        domain = Domain(p=p, q=q, root=root)
        t = PDEProblem(
            domain=domain, source=source_term, D_xx_coefficients=d_xx_coeffs
        )

        Y_arr, T_arr, v_arr, h_arr = local_solve_stage_adaptive_2D_DtN(t)

        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.domain.root)):
            leaf.data.T = T_arr[i]
            leaf.data.h = h_arr[i]
            leaf.data.v = v_arr[i]
            leaf.data.Y = Y_arr[i]

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)

        merge_stage_adaptive_2D_DtN(t)

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = domain.get_adaptive_boundary_data_lst(f)

        solns = down_pass_adaptive_2D_DtN(
            pde_problem=t, boundary_data=boundary_data_lst
        )
        assert solns.shape == (num_leaves, p**2)
        assert not jnp.any(jnp.isnan(solns))
        assert not jnp.any(jnp.isinf(solns))

    def test_2(self, caplog) -> None:
        """Difficult test case seen in the wild."""
        caplog.set_level(logging.DEBUG)
        p = 4
        q = 2
        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(add_to=root, root=root, q=q)
        for child in root.children:
            add_four_children(add_to=child, root=root, q=q)

        add_four_children(add_to=root.children[2].children[1], root=root, q=q)

        num_leaves = len(get_all_leaves(root))
        # print("test_3: Max depth = ", root.children[0].children[0].depth)

        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        # Create domain and PDE problem objects.
        domain = Domain(p=p, q=q, root=root)
        t = PDEProblem(
            domain=domain, source=source_term, D_xx_coefficients=d_xx_coeffs
        )

        Y_arr, T_arr, v_arr, h_arr = local_solve_stage_adaptive_2D_DtN(t)

        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.domain.root)):
            leaf.data.T = T_arr[i]
            leaf.data.h = h_arr[i]
            leaf.data.v = v_arr[i]
            leaf.data.Y = Y_arr[i]

        merge_stage_adaptive_2D_DtN(t)

        print("test_3: Completed build stage.")

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = domain.get_adaptive_boundary_data_lst(f)

        solns = down_pass_adaptive_2D_DtN(
            pde_problem=t, boundary_data=boundary_data_lst
        )
        assert solns.shape == (num_leaves, p**2)
        assert not jnp.any(jnp.isnan(solns))
        assert not jnp.any(jnp.isinf(solns))
        jax.clear_caches()


class Test__propogate_down_quad:
    def test_0(self) -> None:
        """Tests to make sure returns without error."""

        q = 4
        S_arr = np.random.normal(size=(4 * q, 8 * q))

        bdry_data_lst = [np.random.normal(size=(2 * q,)) for _ in range(4)]
        v_int = np.random.normal(size=(4 * q))

        compression_lsts = [jnp.array([False]) for _ in range(8)]

        refinement_op = np.random.normal(size=(2 * q, q))

        out = _propogate_down_quad(
            S_arr,
            bdry_data_lst,
            v_int,
            n_a_0=q,
            n_b_0=q,
            n_b_1=q,
            n_c_1=q,
            n_c_2=q,
            n_d_2=q,
            n_d_3=q,
            n_a_3=q,
            compression_lsts=compression_lsts,
            refinement_op=refinement_op,
        )
        expected_out_len = 4
        assert len(out) == expected_out_len
        # expected_out_shape = (4 * q,)
        for x in out:
            for z in x:
                assert z.shape == (q,)

    def test_1(self) -> None:
        """Uniform refinement; tests that the
        interfaces match up."""
        q = 4
        S_arr = np.random.normal(size=(4 * q, 8 * q))

        bdry_data_lst = [np.random.normal(size=(2 * q,)) for _ in range(4)]
        v_int = np.random.normal(size=(4 * q))

        compression_lsts = [jnp.array([False]) for _ in range(8)]

        refinement_op = np.random.normal(size=(2 * q, q))

        out = _propogate_down_quad(
            S_arr,
            bdry_data_lst,
            v_int,
            n_a_0=q,
            n_b_0=q,
            n_b_1=q,
            n_c_1=q,
            n_c_2=q,
            n_d_2=q,
            n_d_3=q,
            n_a_3=q,
            compression_lsts=compression_lsts,
            refinement_op=refinement_op,
        )

        g_a = out[0]
        g_b = out[1]
        g_c = out[2]
        g_d = out[3]

        # Check the interfaces match up

        # Edge 5
        assert jnp.allclose(g_a[1], jnp.flipud(g_b[3]))
        # Edge 6
        assert jnp.allclose(g_b[2], jnp.flipud(g_c[0]))
        # Edge 7
        assert jnp.allclose(g_c[3], jnp.flipud(g_d[1]))
        # Edge 8
        assert jnp.allclose(g_d[0], jnp.flipud(g_a[2]))
        jax.clear_caches()
