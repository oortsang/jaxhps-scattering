import logging
import pytest
import numpy as np
import jax.numpy as jnp

from hps.src.solver_obj import (
    create_solver_obj_2D,
    create_solver_obj_3D,
    SolverObj,
)
from hps.src.quadrature.trees import (
    Node,
    add_four_children,
    get_all_leaves,
    get_all_leaves_jitted,
    add_uniform_levels,
    get_nodes_at_level,
    add_eight_children,
)

from hps.src.methods.adaptive_build_stage import (
    _build_stage_2D,
    _quad_merge,
    _build_stage_3D,
    _oct_merge,
    quad_merge_nonuniform_whole_level,
    oct_merge_nonuniform_whole_level,
    node_to_oct_merge_outputs,
    is_node_type,
)
from hps.src.methods.local_solve_stage import (
    _local_solve_stage_2D,
    _local_solve_stage_3D,
)


class Test__build_stage_2D:
    def test_0(self, caplog) -> None:
        """Tests the _build_stage function returns without error on a tree
        with uniform refinement.
        """
        caplog.set_level(logging.DEBUG)
        p = 4
        q = 2
        l = 3
        num_leaves = 4**l

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=None, zmax=None
        )
        add_uniform_levels(root=root, l=l, q=q)
        print("test_0: root.n_0: ", root.n_0)
        print("test_0: root.n_1: ", root.n_1)
        t = create_solver_obj_2D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))

        sidelens = jnp.array([leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)])

        Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            p=t.p,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
            sidelens=sidelens,
        )

        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.root)):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        print("test_0: DtN_arr.shape = ", DtN_arr.shape)
        print("test_0: v_prime_arr.shape = ", v_prime_arr.shape)
        _build_stage_2D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
        )

        for node in get_nodes_at_level(t.root, l - 1):
            assert node.DtN.shape == (8 * q, 8 * q)
            assert node.v_prime.shape == (8 * q,)
            assert node.S.shape == (4 * q, 8 * q)

        for node in get_nodes_at_level(t.root, l - 2):
            assert node.DtN.shape == (16 * q, 16 * q)
            assert node.v_prime.shape == (16 * q,)
            assert node.S.shape == (8 * q, 16 * q)

        for node in get_nodes_at_level(t.root, l - 3):
            assert node.DtN.shape == (32 * q, 32 * q)
            assert node.v_prime.shape == (32 * q,)
            assert node.S.shape == (16 * q, 32 * q)

    def test_1(self, caplog) -> None:
        """Tests the _build_stage function returns without error on a tree
        with non-uniform refinement.
        """
        caplog.set_level(logging.DEBUG)

        p = 4
        q = 2
        num_leaves = 7

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=None, zmax=None
        )
        add_four_children(root, root=root, q=q)
        add_four_children(root.children[0], root=root, q=q)
        print("test_1: Max depth = ", root.children[0].children[0].depth)

        t = create_solver_obj_2D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        sidelens = jnp.array([leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)])
        Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            Q_D=t.Q_D,
            P=t.P,
            p=t.p,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
            sidelens=sidelens,
        )

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)

        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.root)):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]

        print("test_1: DtN_arr.shape = ", DtN_arr.shape)
        print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
        _build_stage_2D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
        )

        # Check that the DtN and v_prime arrays are set in the tree object for levels 0, 1, 2.
        for node in get_nodes_at_level(t.root, 0):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None

        for node in get_nodes_at_level(t.root, 1):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None

        for node in get_nodes_at_level(t.root, 2):
            assert node.DtN is not None
            assert node.v_prime is not None

    def test_2(self) -> None:
        """Tests the _build_stage function returns without error on a tree
        with non-uniform refinement.
        """
        p = 4
        q = 2

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=None, zmax=None
        )
        add_uniform_levels(root, 2, q=q)

        # Add some non-unifority to have the grandchildren near the center of the domain
        # be refined more than the others.
        add_four_children(root.children[0].children[2], root=root, q=q)
        add_four_children(root.children[1].children[3], root=root, q=q)
        add_four_children(root.children[2].children[0], root=root, q=q)
        add_four_children(root.children[3].children[1], root=root, q=q)
        num_leaves = len(get_all_leaves(root))

        t = create_solver_obj_2D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        sidelens = jnp.array([leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)])
        Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            p=t.p,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
            sidelens=sidelens,
        )

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)

        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.root)):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]

        print("test_1: DtN_arr.shape = ", DtN_arr.shape)
        print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
        _build_stage_2D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
        )

        # Check that the DtN and v_prime arrays are set in the tree object for levels 0, 1, 2.
        for node in get_nodes_at_level(t.root, 0):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None

        for node in get_nodes_at_level(t.root, 1):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None

        for node in get_nodes_at_level(t.root, 2):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None

        for node in get_nodes_at_level(t.root, 3):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None

    def test_3(self) -> None:
        """Tests the _build_stage function returns without error on a tree
        with non-uniform refinement.
        """
        p = 4
        q = 2

        print("test_3: q = ", q)

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=None, zmax=None
        )
        add_four_children(add_to=root, root=root, q=q)
        add_four_children(add_to=root.children[0], root=root, q=q)
        add_four_children(add_to=root.children[0].children[0], root=root, q=q)

        num_leaves = len(get_all_leaves(root))
        print("test_3: Max depth = ", root.children[0].children[0].depth)

        t = create_solver_obj_2D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        sidelens = jnp.array([leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)])
        Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            p=t.p,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
            sidelens=sidelens,
        )

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)

        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.root)):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]

        print("test_1: DtN_arr.shape = ", DtN_arr.shape)
        print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
        _build_stage_2D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
        )

    def test_4(self) -> None:
        """Tests the _build_stage function returns without error on a tree
        with non-uniform refinement.

        Found this test case in the wild.
        """
        p = 4
        q = 2

        print("test_3: q = ", q)

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=None, zmax=None
        )
        add_four_children(add_to=root, root=root, q=q)
        add_four_children(add_to=root.children[1], root=root, q=q)
        add_four_children(add_to=root.children[2], root=root, q=q)
        add_four_children(add_to=root.children[3], root=root, q=q)

        num_leaves = len(get_all_leaves(root))
        # print("test_3: Max depth = ", root.children[0].children[0].depth)

        t = create_solver_obj_2D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        sidelens = jnp.array([leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)])
        Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            p=t.p,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
            sidelens=sidelens,
        )

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)

        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.root)):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]

        print("test_1: DtN_arr.shape = ", DtN_arr.shape)
        print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
        _build_stage_2D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
        )

    def test_5(self) -> None:
        """Tests the _build_stage function returns without error on a tree
        with non-uniform refinement.

        Found this test case in the wild.
        """
        p = 4
        q = 2

        print("test_3: q = ", q)

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=None, zmax=None
        )
        add_four_children(add_to=root, root=root, q=q)
        for child in root.children:
            add_four_children(add_to=child, root=root, q=q)

        add_four_children(add_to=root.children[2].children[1], root=root, q=q)

        num_leaves = len(get_all_leaves(root))
        # print("test_3: Max depth = ", root.children[0].children[0].depth)

        t = create_solver_obj_2D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        sidelens = jnp.array([leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)])
        Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            p=t.p,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
            sidelens=sidelens,
        )

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)

        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.root)):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]

        print("test_1: DtN_arr.shape = ", DtN_arr.shape)
        print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
        _build_stage_2D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
        )


class Test__build_stage_3D:
    def test_0(self) -> None:
        """Makes sure things run correctly under uniform refinement."""
        p = 6
        q = 4
        l = 2
        num_leaves = 8**l
        root = Node(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, depth=0)
        add_uniform_levels(root=root, l=l, q=q)
        print("test_0: root.n_0", root.n_0)
        t = create_solver_obj_3D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**3))
        source_term = np.random.normal(size=(num_leaves, p**3))
        DtN_arr, v_arr, v_prime_arr, Y_arr = _local_solve_stage_3D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_xz=t.D_xz,
            D_yz=t.D_yz,
            D_zz=t.D_zz,
            D_x=t.D_x,
            D_y=t.D_y,
            D_z=t.D_z,
            P=t.P,
            p=t.p,
            q=t.q,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
            sidelens=jnp.array(
                [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
            ),
        )
        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.root)):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]

        assert Y_arr.shape == (num_leaves, p**3, 6 * q**2)
        _build_stage_3D(
            root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
            DtN_arr=DtN_arr,
            v_prime_arr=v_prime_arr,
            q=q,
        )
        # Check that the DtN and v_prime arrays are set in the tree object for levels 0, 1, 2.
        for node in get_nodes_at_level(t.root, 0):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None

        for node in get_nodes_at_level(t.root, 1):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None

    def test_1(self) -> None:
        """Makes sure things run correctly under non-uniform refinement."""
        for child_idx in range(6):
            # Run the test once for each child to make sure all of the indexing
            # is working correctly.
            print("test_1: child_idx = ", child_idx)
            p = 4
            q = 2
            print("test_1: q = ", q)
            print("test_1: q**2 = ", q**2)
            root = Node(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, depth=0)
            add_eight_children(root, root=root, q=q)
            add_eight_children(root.children[child_idx], root=root, q=q)
            add_eight_children(
                root.children[child_idx].children[child_idx], root=root, q=q
            )
            t = create_solver_obj_3D(p, q, root)
            num_leaves = len(get_all_leaves(root))
            d_xx_coeffs = np.random.normal(size=(num_leaves, p**3))
            source_term = np.random.normal(size=(num_leaves, p**3))
            DtN_arr, v_arr, v_prime_arr, Y_arr = _local_solve_stage_3D(
                D_xx=t.D_xx,
                D_xy=t.D_xy,
                D_yy=t.D_yy,
                D_xz=t.D_xz,
                D_yz=t.D_yz,
                D_zz=t.D_zz,
                D_x=t.D_x,
                D_y=t.D_y,
                D_z=t.D_z,
                P=t.P,
                p=t.p,
                q=t.q,
                D_xx_coeffs=d_xx_coeffs,
                source_term=source_term,
                sidelens=jnp.array(
                    [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
                ),
            )
            # Set the output DtN and v_prime arrays in the tree object.
            for i, leaf in enumerate(get_all_leaves(t.root)):
                leaf.DtN = DtN_arr[i]
                leaf.v_prime = v_prime_arr[i]

            assert Y_arr.shape == (num_leaves, p**3, 6 * q**2)
            _build_stage_3D(
                root,
                refinement_op=t.refinement_op,
                coarsening_op=t.coarsening_op,
                DtN_arr=DtN_arr,
                v_prime_arr=v_prime_arr,
                q=q,
            )
            for node in get_nodes_at_level(t.root, 0):
                assert node.DtN is not None
                assert node.v_prime is not None
                if len(node.children):
                    assert node.S is not None

            for node in get_nodes_at_level(t.root, 1):
                assert node.DtN is not None
                assert node.v_prime is not None
                if len(node.children):
                    assert node.S is not None

    def test_2(self) -> None:
        """Makes sure things run correctly under non-uniform refinement."""
        # Run the test once for each child to make sure all of the indexing
        # is working correctly.
        p = 4
        q = 2
        print("test_1: q = ", q)
        print("test_1: q**2 = ", q**2)
        root = Node(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, depth=0)
        add_eight_children(root, root=root, q=q)
        add_eight_children(root.children[1], root=root, q=q)
        add_eight_children(root.children[2], root=root, q=q)
        add_eight_children(root.children[3], root=root, q=q)
        t = create_solver_obj_3D(p, q, root)
        num_leaves = len(get_all_leaves(root))
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**3))
        source_term = np.random.normal(size=(num_leaves, p**3))
        DtN_arr, v_arr, v_prime_arr, Y_arr = _local_solve_stage_3D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_xz=t.D_xz,
            D_yz=t.D_yz,
            D_zz=t.D_zz,
            D_x=t.D_x,
            D_y=t.D_y,
            D_z=t.D_z,
            P=t.P,
            p=t.p,
            q=t.q,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
            sidelens=jnp.array(
                [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
            ),
        )
        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.root)):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]

        assert Y_arr.shape == (num_leaves, p**3, 6 * q**2)
        _build_stage_3D(
            root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
            DtN_arr=DtN_arr,
            v_prime_arr=v_prime_arr,
            q=q,
        )
        for node in get_nodes_at_level(t.root, 0):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None

        for node in get_nodes_at_level(t.root, 1):
            assert node.DtN is not None
            assert node.v_prime is not None
            if len(node.children):
                assert node.S is not None


class Test__quad_merge:
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

        S, T, v_prime_ext, v_int = _quad_merge(
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

        S, T, v_prime_ext, v_int = _quad_merge(
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

        S, T, v_prime_ext, v_int = _quad_merge(
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

        S, T, v_prime_ext, v_int = _quad_merge(
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


class Test__oct_merge:
    def test_0(self):
        """Tests the _oct_merge function returns without error when none of the
        input arrays need interpolation."""
        q = 2
        n_gauss_bdry = 6 * q**2
        n_gauss_bdry_refined = 2 * n_gauss_bdry
        T_a = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_b = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_c = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_d = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_e = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_f = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_g = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_h = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        v_prime_a = np.random.normal(size=(n_gauss_bdry))
        v_prime_b = np.random.normal(size=(n_gauss_bdry))
        v_prime_c = np.random.normal(size=(n_gauss_bdry))
        v_prime_d = np.random.normal(size=(n_gauss_bdry))
        v_prime_e = np.random.normal(size=(n_gauss_bdry))
        v_prime_f = np.random.normal(size=(n_gauss_bdry))
        v_prime_g = np.random.normal(size=(n_gauss_bdry))
        v_prime_h = np.random.normal(size=(n_gauss_bdry))
        L_2f1 = np.random.normal(size=(n_gauss_bdry_refined, n_gauss_bdry))
        L_1f2 = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry_refined))
        q_idxes = np.arange(q)
        sidelens = jnp.array([q**2, q**2, q**2, q**2, q**2, q**2])
        need_interp_lsts = [
            jnp.array([False]),
        ] * 24
        S, T, v_prime_ext, v_int = _oct_merge(
            T_a=T_a,
            T_b=T_b,
            T_c=T_c,
            T_d=T_d,
            T_e=T_e,
            T_f=T_f,
            T_g=T_g,
            T_h=T_h,
            v_prime_a=v_prime_a,
            v_prime_b=v_prime_b,
            v_prime_c=v_prime_c,
            v_prime_d=v_prime_d,
            v_prime_e=v_prime_e,
            v_prime_f=v_prime_f,
            v_prime_g=v_prime_g,
            v_prime_h=v_prime_h,
            L_1f2=L_1f2,
            L_2f1=L_2f1,
            need_interp_lsts=need_interp_lsts,
            side_lens_a=sidelens,
            side_lens_b=sidelens,
            side_lens_c=sidelens,
            side_lens_d=sidelens,
            side_lens_e=sidelens,
            side_lens_f=sidelens,
            side_lens_g=sidelens,
            side_lens_h=sidelens,
        )

        assert S.shape == (12 * q**2, 24 * q**2)
        assert T.shape == (24 * q**2, 24 * q**2)
        assert v_prime_ext.shape == (24 * q**2,)
        assert v_int.shape == (12 * q**2,)

    def test_1(self):
        """Tests the _oct_merge function returns without error when just a
        need interpolation."""
        q = 2
        n_per_face = q**2
        n_gauss_bdry = 6 * n_per_face
        n_gauss_bdry_refined = 24 * n_per_face
        T_a = np.random.normal(size=(n_gauss_bdry_refined, n_gauss_bdry_refined))
        T_b = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_c = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_d = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_e = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_f = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_g = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_h = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        v_prime_a = np.random.normal(size=(n_gauss_bdry_refined))
        v_prime_b = np.random.normal(size=(n_gauss_bdry))
        v_prime_c = np.random.normal(size=(n_gauss_bdry))
        v_prime_d = np.random.normal(size=(n_gauss_bdry))
        v_prime_e = np.random.normal(size=(n_gauss_bdry))
        v_prime_f = np.random.normal(size=(n_gauss_bdry))
        v_prime_g = np.random.normal(size=(n_gauss_bdry))
        v_prime_h = np.random.normal(size=(n_gauss_bdry))
        L_2f1 = np.random.normal(size=(4 * n_per_face, n_per_face))
        L_1f2 = np.random.normal(size=(n_per_face, 4 * n_per_face))
        sidelens = jnp.array([q**2, q**2, q**2, q**2, q**2, q**2])
        sidelens_a = 4 * sidelens

        need_interp_lsts = [
            jnp.array([False]),
        ] * 24
        # Specify that the faces in a need interpolation.
        need_interp_lsts[0] = jnp.array([True, True, True, True])
        need_interp_lsts[7] = jnp.array([True, True, True, True])
        need_interp_lsts[16] = jnp.array([True, True, True, True])
        print("test_1: n_per_face: ", n_per_face)
        q_idxes = np.arange(q)
        S, T, v_prime_ext, v_int = _oct_merge(
            T_a=T_a,
            T_b=T_b,
            T_c=T_c,
            T_d=T_d,
            T_e=T_e,
            T_f=T_f,
            T_g=T_g,
            T_h=T_h,
            v_prime_a=v_prime_a,
            v_prime_b=v_prime_b,
            v_prime_c=v_prime_c,
            v_prime_d=v_prime_d,
            v_prime_e=v_prime_e,
            v_prime_f=v_prime_f,
            v_prime_g=v_prime_g,
            v_prime_h=v_prime_h,
            L_1f2=L_1f2,
            L_2f1=L_2f1,
            need_interp_lsts=need_interp_lsts,
            side_lens_a=sidelens_a,
            side_lens_b=sidelens,
            side_lens_c=sidelens,
            side_lens_d=sidelens,
            side_lens_e=sidelens,
            side_lens_f=sidelens,
            side_lens_g=sidelens,
            side_lens_h=sidelens,
        )

        n_faces_out = 24 + 9

        assert S.shape == (12 * q**2, n_faces_out * n_per_face)
        assert T.shape == (n_faces_out * n_per_face, n_faces_out * n_per_face)
        assert v_prime_ext.shape == (n_faces_out * n_per_face,)
        assert v_int.shape == (12 * q**2,)


class Test_quad_merge_nonuniform_whole_level:

    def test_0(self) -> None:
        """Make sure things work and return the correct shapes."""

        q = 4
        n_bdry = 8 * q
        n_bdry_int = 2 * q
        n_out_quads = 1
        n_leaves_input = n_out_quads * 4
        T_in = [np.random.normal(size=(n_bdry, n_bdry)) for _ in range(n_leaves_input)]
        v_prime_in = [np.random.normal(size=(n_bdry,)) for _ in range(n_leaves_input)]

        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=None, zmax=None, depth=0
        )
        add_uniform_levels(root, l=2, q=q)
        nodes_this_level = [
            root.children[0],
            root.children[1],
            root.children[2],
            root.children[3],
        ]

        S, T, v_prime_ext, v_int = quad_merge_nonuniform_whole_level(
            T_in=T_in,
            v_prime=v_prime_in,
            L_2f1=L_2f1,
            L_1f2=L_1f2,
            nodes_this_level=nodes_this_level,
        )

        assert len(S) == n_out_quads
        assert len(T) == n_out_quads
        assert len(v_prime_ext) == n_out_quads
        assert len(v_int) == n_out_quads

        # for S_matrix in S:
        #     print("test_0: S_matrix.shape = ", S_matrix.shape)
        #     assert S_matrix.shape == (2 * n_bdry_int, n_bdry_int)
        # for T_matrix in T:
        #     assert T_matrix.shape == (n_bdry, n_bdry)
        # for v_prime_ext_vec in v_prime_ext:
        #     assert v_prime_ext_vec.shape == (n_bdry,)
        # for v_int_vec in v_int:
        #     assert v_int_vec.shape == (n_bdry_int,)


class Test_node_to_oct_merge_outputs:
    def test_0(self) -> None:
        q = 2
        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0)
        n_per_panel = q**2
        add_eight_children(root, root, q)

        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))

        root.L_1f2 = L_1f2
        root.L_2f1 = L_2f1

        for child in root.children:
            # Set DtN and v_prime attributes.
            child.DtN = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
            child.v_prime = np.random.normal(size=(6 * n_per_panel,))
            # child.L_2f1 = L_2f1
            # child.L_1f2 = L_1f2

        S, T, v_prime_ext, v_int = node_to_oct_merge_outputs(
            root,
        )

        assert T.shape == (24 * n_per_panel, 24 * n_per_panel)
        assert S.shape == (12 * n_per_panel, 24 * n_per_panel)


class Test_oct_merge_nonuniform_whole_level:
    def test_0(self) -> None:

        q = 2
        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0)
        n_per_panel = q**2
        add_eight_children(root, root, q)
        for child in root.children:
            add_eight_children(child, root, q)

        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))

        for l in get_all_leaves(root):
            # Set DtN and v_prime attributes.
            l.DtN = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
            l.v_prime = np.random.normal(size=(6 * n_per_panel,))

        nodes_level_1 = get_nodes_at_level(root, 1)
        assert len(nodes_level_1) == 8

        oct_merge_nonuniform_whole_level(
            L_1f2=L_1f2,
            L_2f1=L_2f1,
            nodes_this_level=nodes_level_1,
        )


class Test_is_node_type:
    def test_0(self) -> None:
        x = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0)
        x.DtN = np.random.normal(size=(10, 10))

        assert is_node_type(x)
        assert not is_node_type(x.DtN)


if __name__ == "__main__":
    pytest.main()
