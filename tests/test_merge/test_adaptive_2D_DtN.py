import numpy as np
import jax.numpy as jnp


from hahps._discretization_tree import (
    DiscretizationNode2D,
)
from hahps._discretization_tree_operations_2D import add_four_children
from hahps.merge._adaptive_2D_DtN import (
    quad_merge_nonuniform_whole_level,
    _adaptive_quad_merge_2D_DtN,
)


# class Test_build_stage_adaptive_2D_DtN:
#     def test_0(self, caplog) -> None:
#         """Tests the _build_stage function returns without error on a tree
#         with uniform refinement.
#         """
#         caplog.set_level(logging.DEBUG)
#         p = 4
#         q = 2
#         l = 3
#         num_leaves = 4**l

#         root = Node(
#             xmin=0.0,
#             xmax=1.0,
#             ymin=0.0,
#             ymax=1.0,
#             depth=0,
#             zmin=None,
#             zmax=None,
#         )
#         add_uniform_levels(root=root, l=l, q=q)
#         print("test_0: root.n_0: ", root.n_0)
#         print("test_0: root.n_1: ", root.n_1)
#         t = create_solver_obj_2D(p, q, root)
#         d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
#         source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))

#         sidelens = jnp.array(
#             [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
#         )

#         Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
#             D_xx=t.D_xx,
#             D_xy=t.D_xy,
#             D_yy=t.D_yy,
#             D_x=t.D_x,
#             D_y=t.D_y,
#             P=t.P,
#             p=t.p,
#             D_xx_coeffs=d_xx_coeffs,
#             source_term=source_term,
#             sidelens=sidelens,
#         )

#         # Set the output DtN and v_prime arrays in the tree object.
#         for i, leaf in enumerate(get_all_leaves(t.root)):
#             leaf.DtN = DtN_arr[i]
#             leaf.v_prime = v_prime_arr[i]

#         assert Y_arr.shape == (num_leaves, p**2, 4 * q)
#         # n_leaves, n_bdry, _ = DtN_arr.shape
#         # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
#         # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

#         print("test_0: DtN_arr.shape = ", DtN_arr.shape)
#         print("test_0: v_prime_arr.shape = ", v_prime_arr.shape)
#         _build_stage_2D(
#             root=t.root,
#             refinement_op=t.refinement_op,
#             coarsening_op=t.coarsening_op,
#         )

#         for node in get_nodes_at_level(t.root, l - 1):
#             assert node.DtN.shape == (8 * q, 8 * q)
#             assert node.v_prime.shape == (8 * q,)
#             assert node.S.shape == (4 * q, 8 * q)

#         for node in get_nodes_at_level(t.root, l - 2):
#             assert node.DtN.shape == (16 * q, 16 * q)
#             assert node.v_prime.shape == (16 * q,)
#             assert node.S.shape == (8 * q, 16 * q)

#         for node in get_nodes_at_level(t.root, l - 3):
#             assert node.DtN.shape == (32 * q, 32 * q)
#             assert node.v_prime.shape == (32 * q,)
#             assert node.S.shape == (16 * q, 32 * q)

#     def test_1(self, caplog) -> None:
#         """Tests the _build_stage function returns without error on a tree
#         with non-uniform refinement.
#         """
#         caplog.set_level(logging.DEBUG)

#         p = 4
#         q = 2
#         num_leaves = 7

#         root = Node(
#             xmin=0.0,
#             xmax=1.0,
#             ymin=0.0,
#             ymax=1.0,
#             depth=0,
#             zmin=None,
#             zmax=None,
#         )
#         add_four_children(root, root=root, q=q)
#         add_four_children(root.children[0], root=root, q=q)
#         print("test_1: Max depth = ", root.children[0].children[0].depth)

#         t = create_solver_obj_2D(p, q, root)
#         d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
#         source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
#         sidelens = jnp.array(
#             [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
#         )
#         Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
#             D_xx=t.D_xx,
#             D_xy=t.D_xy,
#             D_yy=t.D_yy,
#             D_x=t.D_x,
#             D_y=t.D_y,
#             Q_D=t.Q_D,
#             P=t.P,
#             p=t.p,
#             D_xx_coeffs=d_xx_coeffs,
#             source_term=source_term,
#             sidelens=sidelens,
#         )

#         assert Y_arr.shape == (num_leaves, p**2, 4 * q)

#         # Set the output DtN and v_prime arrays in the tree object.
#         for i, leaf in enumerate(get_all_leaves(t.root)):
#             leaf.DtN = DtN_arr[i]
#             leaf.v_prime = v_prime_arr[i]

#         print("test_1: DtN_arr.shape = ", DtN_arr.shape)
#         print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
#         _build_stage_2D(
#             root=t.root,
#             refinement_op=t.refinement_op,
#             coarsening_op=t.coarsening_op,
#         )

#         # Check that the DtN and v_prime arrays are set in the tree object for levels 0, 1, 2.
#         for node in get_nodes_at_level(t.root, 0):
#             assert node.DtN is not None
#             assert node.v_prime is not None
#             if len(node.children):
#                 assert node.S is not None

#         for node in get_nodes_at_level(t.root, 1):
#             assert node.DtN is not None
#             assert node.v_prime is not None
#             if len(node.children):
#                 assert node.S is not None

#         for node in get_nodes_at_level(t.root, 2):
#             assert node.DtN is not None
#             assert node.v_prime is not None

#     def test_2(self) -> None:
#         """Tests the _build_stage function returns without error on a tree
#         with non-uniform refinement.
#         """
#         p = 4
#         q = 2

#         root = Node(
#             xmin=0.0,
#             xmax=1.0,
#             ymin=0.0,
#             ymax=1.0,
#             depth=0,
#             zmin=None,
#             zmax=None,
#         )
#         add_uniform_levels(root, 2, q=q)

#         # Add some non-unifority to have the grandchildren near the center of the domain
#         # be refined more than the others.
#         add_four_children(root.children[0].children[2], root=root, q=q)
#         add_four_children(root.children[1].children[3], root=root, q=q)
#         add_four_children(root.children[2].children[0], root=root, q=q)
#         add_four_children(root.children[3].children[1], root=root, q=q)
#         num_leaves = len(get_all_leaves(root))

#         t = create_solver_obj_2D(p, q, root)
#         d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
#         source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
#         sidelens = jnp.array(
#             [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
#         )
#         Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
#             D_xx=t.D_xx,
#             D_xy=t.D_xy,
#             D_yy=t.D_yy,
#             D_x=t.D_x,
#             D_y=t.D_y,
#             P=t.P,
#             p=t.p,
#             D_xx_coeffs=d_xx_coeffs,
#             source_term=source_term,
#             sidelens=sidelens,
#         )

#         assert Y_arr.shape == (num_leaves, p**2, 4 * q)

#         # Set the output DtN and v_prime arrays in the tree object.
#         for i, leaf in enumerate(get_all_leaves(t.root)):
#             leaf.DtN = DtN_arr[i]
#             leaf.v_prime = v_prime_arr[i]

#         print("test_1: DtN_arr.shape = ", DtN_arr.shape)
#         print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
#         _build_stage_2D(
#             root=t.root,
#             refinement_op=t.refinement_op,
#             coarsening_op=t.coarsening_op,
#         )

#         # Check that the DtN and v_prime arrays are set in the tree object for levels 0, 1, 2.
#         for node in get_nodes_at_level(t.root, 0):
#             assert node.DtN is not None
#             assert node.v_prime is not None
#             if len(node.children):
#                 assert node.S is not None

#         for node in get_nodes_at_level(t.root, 1):
#             assert node.DtN is not None
#             assert node.v_prime is not None
#             if len(node.children):
#                 assert node.S is not None

#         for node in get_nodes_at_level(t.root, 2):
#             assert node.DtN is not None
#             assert node.v_prime is not None
#             if len(node.children):
#                 assert node.S is not None

#         for node in get_nodes_at_level(t.root, 3):
#             assert node.DtN is not None
#             assert node.v_prime is not None
#             if len(node.children):
#                 assert node.S is not None

#     def test_3(self) -> None:
#         """Tests the _build_stage function returns without error on a tree
#         with non-uniform refinement.
#         """
#         p = 4
#         q = 2

#         print("test_3: q = ", q)

#         root = Node(
#             xmin=0.0,
#             xmax=1.0,
#             ymin=0.0,
#             ymax=1.0,
#             depth=0,
#             zmin=None,
#             zmax=None,
#         )
#         add_four_children(add_to=root, root=root, q=q)
#         add_four_children(add_to=root.children[0], root=root, q=q)
#         add_four_children(add_to=root.children[0].children[0], root=root, q=q)

#         num_leaves = len(get_all_leaves(root))
#         print("test_3: Max depth = ", root.children[0].children[0].depth)

#         t = create_solver_obj_2D(p, q, root)
#         d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
#         source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
#         sidelens = jnp.array(
#             [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
#         )
#         Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
#             D_xx=t.D_xx,
#             D_xy=t.D_xy,
#             D_yy=t.D_yy,
#             D_x=t.D_x,
#             D_y=t.D_y,
#             P=t.P,
#             p=t.p,
#             D_xx_coeffs=d_xx_coeffs,
#             source_term=source_term,
#             sidelens=sidelens,
#         )

#         assert Y_arr.shape == (num_leaves, p**2, 4 * q)

#         # Set the output DtN and v_prime arrays in the tree object.
#         for i, leaf in enumerate(get_all_leaves(t.root)):
#             leaf.DtN = DtN_arr[i]
#             leaf.v_prime = v_prime_arr[i]

#         print("test_1: DtN_arr.shape = ", DtN_arr.shape)
#         print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
#         _build_stage_2D(
#             root=t.root,
#             refinement_op=t.refinement_op,
#             coarsening_op=t.coarsening_op,
#         )

#     def test_4(self) -> None:
#         """Tests the _build_stage function returns without error on a tree
#         with non-uniform refinement.

#         Found this test case in the wild.
#         """
#         p = 4
#         q = 2

#         print("test_3: q = ", q)

#         root = Node(
#             xmin=0.0,
#             xmax=1.0,
#             ymin=0.0,
#             ymax=1.0,
#             depth=0,
#             zmin=None,
#             zmax=None,
#         )
#         add_four_children(add_to=root, root=root, q=q)
#         add_four_children(add_to=root.children[1], root=root, q=q)
#         add_four_children(add_to=root.children[2], root=root, q=q)
#         add_four_children(add_to=root.children[3], root=root, q=q)

#         num_leaves = len(get_all_leaves(root))
#         # print("test_3: Max depth = ", root.children[0].children[0].depth)

#         t = create_solver_obj_2D(p, q, root)
#         d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
#         source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
#         sidelens = jnp.array(
#             [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
#         )
#         Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
#             D_xx=t.D_xx,
#             D_xy=t.D_xy,
#             D_yy=t.D_yy,
#             D_x=t.D_x,
#             D_y=t.D_y,
#             P=t.P,
#             p=t.p,
#             D_xx_coeffs=d_xx_coeffs,
#             source_term=source_term,
#             sidelens=sidelens,
#         )

#         assert Y_arr.shape == (num_leaves, p**2, 4 * q)

#         # Set the output DtN and v_prime arrays in the tree object.
#         for i, leaf in enumerate(get_all_leaves(t.root)):
#             leaf.DtN = DtN_arr[i]
#             leaf.v_prime = v_prime_arr[i]

#         print("test_1: DtN_arr.shape = ", DtN_arr.shape)
#         print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
#         _build_stage_2D(
#             root=t.root,
#             refinement_op=t.refinement_op,
#             coarsening_op=t.coarsening_op,
#         )

#     def test_5(self) -> None:
#         """Tests the _build_stage function returns without error on a tree
#         with non-uniform refinement.

#         Found this test case in the wild.
#         """
#         p = 4
#         q = 2

#         print("test_3: q = ", q)

#         root = Node(
#             xmin=0.0,
#             xmax=1.0,
#             ymin=0.0,
#             ymax=1.0,
#             depth=0,
#             zmin=None,
#             zmax=None,
#         )
#         add_four_children(add_to=root, root=root, q=q)
#         for child in root.children:
#             add_four_children(add_to=child, root=root, q=q)

#         add_four_children(add_to=root.children[2].children[1], root=root, q=q)

#         num_leaves = len(get_all_leaves(root))
#         # print("test_3: Max depth = ", root.children[0].children[0].depth)

#         t = create_solver_obj_2D(p, q, root)
#         d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
#         source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
#         sidelens = jnp.array(
#             [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
#         )
#         Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
#             D_xx=t.D_xx,
#             D_xy=t.D_xy,
#             D_yy=t.D_yy,
#             D_x=t.D_x,
#             D_y=t.D_y,
#             P=t.P,
#             p=t.p,
#             D_xx_coeffs=d_xx_coeffs,
#             source_term=source_term,
#             sidelens=sidelens,
#         )

#         assert Y_arr.shape == (num_leaves, p**2, 4 * q)

#         # Set the output DtN and v_prime arrays in the tree object.
#         for i, leaf in enumerate(get_all_leaves(t.root)):
#             leaf.DtN = DtN_arr[i]
#             leaf.v_prime = v_prime_arr[i]

#         print("test_1: DtN_arr.shape = ", DtN_arr.shape)
#         print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
#         _build_stage_2D(
#             root=t.root,
#             refinement_op=t.refinement_op,
#             coarsening_op=t.coarsening_op,
#         )


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
