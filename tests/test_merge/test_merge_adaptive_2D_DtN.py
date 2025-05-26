import numpy as np
import jax.numpy as jnp
import jax

from jaxhps._discretization_tree import (
    DiscretizationNode2D,
    get_all_leaves,
    get_nodes_at_level,
)
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
from jaxhps._discretization_tree_operations_2D import add_four_children
from jaxhps.merge._adaptive_2D_DtN import (
    quad_merge_nonuniform_whole_level,
    _adaptive_quad_merge_2D_DtN,
    merge_stage_adaptive_2D_DtN,
)
from jaxhps.local_solve._adaptive_2D_DtN import (
    local_solve_stage_adaptive_2D_DtN,
)
import logging


class Test_merge_stage_adaptive_2D_DtN:
    def test_0(self, caplog) -> None:
        """Tests the _merge_stage function returns without error on a tree
        with uniform refinement.
        """
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

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        logging.debug("test_0: T_arr.shape = %s", T_arr.shape)
        logging.debug("test_0: h_arr.shape = %s", h_arr.shape)
        merge_stage_adaptive_2D_DtN(t)

        leaf0 = get_all_leaves(root)[0]
        logging.debug("test_0: l = %s", leaf0)
        logging.debug("test_0: l.data.T = %s", leaf0.data.T.shape)
        logging.debug("test_0: l.data.h = %s", leaf0.data.h.shape)

        for node in get_nodes_at_level(t.domain.root, l - 1):
            logging.debug("test_0: looking at node: %s", node)
            logging.debug("test_0: id(node.data) = %s", id(node.data))
            assert node.data.T.shape == (8 * q, 8 * q)
            assert node.data.h.shape == (8 * q,)
            assert node.data.S.shape == (4 * q, 8 * q)

        for node in get_nodes_at_level(t.domain.root, l - 2):
            assert node.data.T.shape == (16 * q, 16 * q)
            assert node.data.h.shape == (16 * q,)
            assert node.data.S.shape == (8 * q, 16 * q)

        for node in get_nodes_at_level(t.domain.root, l - 3):
            assert node.data.T.shape == (32 * q, 32 * q)
            assert node.data.h.shape == (32 * q,)
            assert node.data.S.shape == (16 * q, 32 * q)

    def test_1(self, caplog) -> None:
        """Tests the _merge_stage function returns without error on a tree
        with non-uniform refinement.
        """
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

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)

        merge_stage_adaptive_2D_DtN(t)

        # Check that the DtN and v_prime arrays are set in the tree object for levels 0, 1, 2.
        for node in get_nodes_at_level(t.domain.root, 0):
            assert node.data.T is not None
            assert node.data.h is not None
            if len(node.children):
                assert node.data.S is not None

        for node in get_nodes_at_level(t.domain.root, 1):
            assert node.data.T is not None
            assert node.data.h is not None
            if len(node.children):
                assert node.data.S is not None

        for node in get_nodes_at_level(t.domain.root, 2):
            assert node.data.T is not None
            assert node.data.h is not None

    def test_2(self) -> None:
        """Tests the _merge_stage function returns without error on a tree
        with non-uniform refinement.
        """
        p = 4
        q = 2
        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )

        # Manually add l=2 levels
        add_four_children(root, root=root, q=q)
        for c in root.children:
            add_four_children(c, root=root, q=q)

        # Add some non-unifority to have the grandchildren near the center of the domain
        # be refined more than the others.
        add_four_children(root.children[0].children[2], root=root, q=q)
        add_four_children(root.children[1].children[3], root=root, q=q)
        add_four_children(root.children[2].children[0], root=root, q=q)
        add_four_children(root.children[3].children[1], root=root, q=q)
        num_leaves = len(get_all_leaves(root))

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

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)

        merge_stage_adaptive_2D_DtN(t)

        # Check that the DtN and v_prime arrays are set in the tree object for levels 0, 1, 2.
        for node in get_nodes_at_level(t.domain.root, 0):
            assert node.data.T is not None
            assert node.data.h is not None
            if len(node.children):
                assert node.data.S is not None

        for node in get_nodes_at_level(t.domain.root, 1):
            assert node.data.T is not None
            assert node.data.h is not None
            if len(node.children):
                assert node.data.S is not None

        for node in get_nodes_at_level(t.domain.root, 2):
            assert node.data.T is not None
            assert node.data.h is not None
            if len(node.children):
                assert node.data.S is not None
        jax.clear_caches()


class Test__adaptive_quad_merge_2D_DtN:
    def test_0(self) -> None:
        """Tests the _quad_merge function returns without error when none of the
        input arrays need interpolation."""

        n_bdry = 28
        n_bdry_int = n_bdry // 4
        n_bdry_ext = 2 * (n_bdry // 4)
        T_a = np.random.normal(size=(n_bdry, n_bdry))
        T_b = np.random.normal(size=(n_bdry, n_bdry))
        T_c = np.random.normal(size=(n_bdry, n_bdry))
        T_d = np.random.normal(size=(n_bdry, n_bdry))
        v_prime_a = np.random.normal(size=(n_bdry))
        v_prime_b = np.random.normal(size=(n_bdry))
        v_prime_c = np.random.normal(size=(n_bdry))
        v_prime_d = np.random.normal(size=(n_bdry))

        need_interps = (jnp.array([False]),) * 8
        L_2f1 = np.random.normal(size=(n_bdry_ext, n_bdry_int))
        L_1f2 = np.random.normal(size=(n_bdry_int, n_bdry_ext))

        side_lens = jnp.array([n_bdry_int, n_bdry_int, n_bdry_int, n_bdry_int])

        S, T, v_prime_ext, v_int = _adaptive_quad_merge_2D_DtN(
            T_a,
            T_b,
            T_c,
            T_d,
            v_prime_a,
            v_prime_b,
            v_prime_c,
            v_prime_d,
            L_2f1=L_2f1,
            L_1f2=L_1f2,
            need_interp_lsts=need_interps,
            side_lens_a=side_lens,
            side_lens_b=side_lens,
            side_lens_c=side_lens,
            side_lens_d=side_lens,
        )

        assert S.shape == (4 * n_bdry_int, 4 * n_bdry_ext)
        assert T.shape == (4 * n_bdry_ext, 4 * n_bdry_ext)
        assert v_prime_ext.shape == (4 * n_bdry_ext,)
        assert v_int.shape == (4 * n_bdry_int,)

    def test_1(self) -> None:
        """Tests the _quad_merge function returns without error when just a needs interpolation."""

        q = 7
        T_a = np.random.normal(size=(2 * 4 * q, 2 * 4 * q))
        T_b = np.random.normal(size=(4 * q, 4 * q))
        T_c = np.random.normal(size=(4 * q, 4 * q))
        T_d = np.random.normal(size=(4 * q, 4 * q))
        v_prime_a = np.random.normal(size=(2 * 4 * q))
        v_prime_b = np.random.normal(size=(4 * q))
        v_prime_c = np.random.normal(size=(4 * q))
        v_prime_d = np.random.normal(size=(4 * q))

        need_interps = [
            jnp.array([False]),
        ] * 8
        need_interps[0] = jnp.array([True, True])
        need_interps[7] = jnp.array([True, True])
        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))

        side_lens_a = jnp.array([2 * q, 2 * q, 2 * q, 2 * q])
        side_lens = jnp.array([q, q, q, q])

        S, T, v_prime_ext, v_int = _adaptive_quad_merge_2D_DtN(
            T_a,
            T_b,
            T_c,
            T_d,
            v_prime_a,
            v_prime_b,
            v_prime_c,
            v_prime_d,
            L_2f1=L_2f1,
            L_1f2=L_1f2,
            need_interp_lsts=need_interps,
            side_lens_a=side_lens_a,
            side_lens_b=side_lens,
            side_lens_c=side_lens,
            side_lens_d=side_lens,
        )
        print("test_1: q", q)
        print("test_1: S shape", S.shape)
        print("test_1: T shape", T.shape)
        print("test_1: v_prime_ext shape", v_prime_ext.shape)
        print("test_1: v_int shape", v_int.shape)

        assert S.shape == (4 * q, 8 * q + 2 * q)
        assert T.shape == (8 * q + 2 * q, 8 * q + 2 * q)
        assert v_prime_ext.shape == (8 * q + 2 * q,)
        assert v_int.shape == (4 * q,)

    def test_2(self) -> None:
        """Tests the _quad_merge function returns without error when a and b need interpolation."""

        q = 7
        T_a = np.random.normal(size=(2 * 4 * q, 2 * 4 * q))
        T_b = np.random.normal(size=(2 * 4 * q, 2 * 4 * q))
        T_c = np.random.normal(size=(4 * q, 4 * q))
        T_d = np.random.normal(size=(4 * q, 4 * q))
        v_prime_a = np.random.normal(size=(2 * 4 * q))
        v_prime_b = np.random.normal(size=(2 * 4 * q))
        v_prime_c = np.random.normal(size=(4 * q))
        v_prime_d = np.random.normal(size=(4 * q))

        need_interps = [
            jnp.array([False]),
        ] * 8
        need_interps[0] = jnp.array([False, False])
        need_interps[1] = jnp.array([False, False])
        need_interps[2] = jnp.array([True, True])
        need_interps[7] = jnp.array([True, True])
        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))

        side_lens_a = jnp.array([2 * q, 2 * q, 2 * q, 2 * q])
        side_lens = jnp.array([q, q, q, q])

        S, T, v_prime_ext, v_int = _adaptive_quad_merge_2D_DtN(
            T_a,
            T_b,
            T_c,
            T_d,
            v_prime_a,
            v_prime_b,
            v_prime_c,
            v_prime_d,
            L_2f1=L_2f1,
            L_1f2=L_1f2,
            need_interp_lsts=need_interps,
            side_lens_a=side_lens_a,
            side_lens_b=side_lens_a,
            side_lens_c=side_lens,
            side_lens_d=side_lens,
        )
        print("test_2: q", q)
        print("test_2: S shape", S.shape)
        print("test_2: T shape", T.shape)
        print("test_2: v_prime_ext shape", v_prime_ext.shape)
        print("test_2: v_int shape", v_int.shape)

        assert S.shape == (4 * q + q, 8 * q + 4 * q)
        assert T.shape == (8 * q + 4 * q, 8 * q + 4 * q)
        assert v_prime_ext.shape == (8 * q + 4 * q,)
        assert v_int.shape == (4 * q + q,)

    def test_3(self) -> None:
        """Tests the _quad_merge function returns without error when a, b, and c need interpolation."""

        q = 7
        T_a = np.random.normal(size=(2 * 4 * q, 2 * 4 * q))
        T_b = np.random.normal(size=(2 * 4 * q, 2 * 4 * q))
        T_c = np.random.normal(size=(2 * 4 * q, 2 * 4 * q))
        T_d = np.random.normal(size=(4 * q, 4 * q))
        v_prime_a = np.random.normal(size=(2 * 4 * q))
        v_prime_b = np.random.normal(size=(2 * 4 * q))
        v_prime_c = np.random.normal(size=(2 * 4 * q))
        v_prime_d = np.random.normal(size=(4 * q))
        need_interps = [
            jnp.array([False, False]),
        ] * 8

        need_interps[4] = jnp.array([True, True])
        need_interps[7] = jnp.array([True, True])

        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))

        side_lens_a = jnp.array([2 * q, 2 * q, 2 * q, 2 * q])
        side_lens = jnp.array([q, q, q, q])

        S, T, v_prime_ext, v_int = _adaptive_quad_merge_2D_DtN(
            T_a,
            T_b,
            T_c,
            T_d,
            v_prime_a,
            v_prime_b,
            v_prime_c,
            v_prime_d,
            L_2f1=L_2f1,
            L_1f2=L_1f2,
            need_interp_lsts=need_interps,
            side_lens_a=side_lens_a,
            side_lens_b=side_lens_a,
            side_lens_c=side_lens_a,
            side_lens_d=side_lens,
        )
        print("test_1: q", q)
        print("test_1: S shape", S.shape)
        print("test_1: T shape", T.shape)
        print("test_1: v_prime_ext shape", v_prime_ext.shape)
        print("test_1: v_int shape", v_int.shape)

        assert S.shape == (4 * q + 2 * q, 8 * q + 6 * q)
        assert T.shape == (8 * q + 6 * q, 8 * q + 6 * q)
        assert v_prime_ext.shape == (8 * q + 6 * q,)
        assert v_int.shape == (4 * q + 2 * q,)
        jax.clear_caches()


class Test_quad_merge_nonuniform_whole_level:
    def test_0(self) -> None:
        """Make sure things work and return the correct shapes."""

        q = 4
        n_bdry = 8 * q
        # n_bdry_int = 2 * q
        n_out_quads = 1
        n_leaves_input = n_out_quads * 4
        T_in = [
            np.random.normal(size=(n_bdry, n_bdry))
            for _ in range(n_leaves_input)
        ]
        v_prime_in = [
            np.random.normal(size=(n_bdry,)) for _ in range(n_leaves_input)
        ]

        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))

        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(root, root=root, q=q)
        for child in root.children:
            add_four_children(child, root=root, q=q)
        nodes_this_level = [
            root.children[0],
            root.children[1],
            root.children[2],
            root.children[3],
        ]

        S, T, v_prime_ext, v_int = quad_merge_nonuniform_whole_level(
            T_in=T_in,
            h_in=v_prime_in,
            L_2f1=L_2f1,
            L_1f2=L_1f2,
            nodes_this_level=nodes_this_level,
        )

        assert len(S) == n_out_quads
        assert len(T) == n_out_quads
        assert len(v_prime_ext) == n_out_quads
        assert len(v_int) == n_out_quads
        jax.clear_caches()
