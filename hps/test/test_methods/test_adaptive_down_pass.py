import pytest
import numpy as np
import jax.numpy as jnp

from hps.src.solver_obj import (
    create_solver_obj_2D,
    create_solver_obj_3D,
    get_bdry_data_evals_lst_2D,
    get_bdry_data_evals_lst_3D,
)
from hps.src.methods.adaptive_down_pass import (
    _down_pass_3D,
    _down_pass_2D,
    _propogate_down_quad,
    _propogate_down_oct,
)
from hps.src.methods.local_solve_stage import (
    _local_solve_stage_2D,
    _local_solve_stage_3D,
)
from hps.src.methods.adaptive_build_stage import (
    _build_stage_2D,
    _build_stage_3D,
)
from hps.src.quadrature.trees import (
    Node,
    get_all_leaves,
    add_four_children,
    add_eight_children,
    add_uniform_levels,
)


class Test__down_pass_2D:
    def test_0(self) -> None:
        """2D uniform case"""
        p = 4
        q = 2
        l = 4
        num_leaves = 4**l

        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            depth=0,
            zmin=None,
            zmax=None,
        )
        add_uniform_levels(root=root, l=l, q=q)

        t = create_solver_obj_2D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))

        sidelens = jnp.array(
            [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
        )

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

        # Set the data in the Node objects
        leaves = get_all_leaves(t.root)
        for i, leaf in enumerate(leaves):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]
            leaf.Y = Y_arr[i]
            leaf.v = v_arr[i]

        _build_stage_2D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
        )

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = get_bdry_data_evals_lst_2D(t, f)

        leaf_solns = _down_pass_2D(
            root=t.root,
            boundary_data=boundary_data_lst,
            refinement_op=t.refinement_op,
        )
        assert leaf_solns.shape == (num_leaves, p**2)

    def test_1(self) -> None:
        """2D Non-Uniform case."""
        p = 4
        q = 2
        num_leaves = 7

        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            depth=0,
            zmin=None,
            zmax=None,
        )
        add_four_children(add_to=root, root=root, q=q)
        add_four_children(add_to=root.children[2], root=root, q=q)

        t = create_solver_obj_2D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        sidelens = jnp.array(
            [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
        )
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
            leaf.Y = Y_arr[i]
            leaf.v = v_arr[i]
        print("test_1: DtN_arr.shape = ", DtN_arr.shape)
        print("test_1: v_prime_arr.shape = ", v_prime_arr.shape)
        _build_stage_2D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
        )

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = get_bdry_data_evals_lst_2D(t, f)

        solns = _down_pass_2D(
            t.root, boundary_data_lst, refinement_op=t.refinement_op
        )
        assert solns.shape == (num_leaves, p**2)
        assert not jnp.any(jnp.isnan(solns))
        assert not jnp.any(jnp.isinf(solns))

    def test_2(self) -> None:
        """Difficult test case seen in the wild."""

        p = 4
        q = 2

        print("test_3: q = ", q)

        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            depth=0,
            zmin=None,
            zmax=None,
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
        sidelens = jnp.array(
            [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
        )
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
            leaf.Y = Y_arr[i]
            leaf.v = v_arr[i]

        _build_stage_2D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
        )
        print("test_3: Completed build stage.")

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = get_bdry_data_evals_lst_2D(t, f)

        solns = _down_pass_2D(
            t.root, boundary_data_lst, refinement_op=t.refinement_op
        )
        assert solns.shape == (num_leaves, p**2)
        assert not jnp.any(jnp.isnan(solns))
        assert not jnp.any(jnp.isinf(solns))


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


class Test__down_pass_3D:
    def test_0(self) -> None:
        """3D uniform case"""
        p = 4
        q = 2
        l = 2
        num_leaves = 8**l

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=0.0, zmax=1.0
        )

        add_uniform_levels(root=root, l=l, q=q)
        t = create_solver_obj_3D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**3))
        source_term = np.random.normal(size=(num_leaves, p**3))

        sidelens = jnp.array(
            [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
        )

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
            sidelens=sidelens,
        )

        # Set the data in the Node objects
        leaves = get_all_leaves(t.root)
        for i, leaf in enumerate(leaves):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]
            leaf.Y = Y_arr[i]
            leaf.v = v_arr[i]

        _build_stage_3D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
            DtN_arr=DtN_arr,
            v_prime_arr=v_prime_arr,
            q=q,
        )

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = get_bdry_data_evals_lst_3D(t, f)
        print(
            "test_0: boundary_data_lst shapes = ",
            [x.shape for x in boundary_data_lst],
        )
        leaf_solns = _down_pass_3D(
            root=t.root,
            boundary_data=boundary_data_lst,
            refinement_op=t.refinement_op,
        )
        assert leaf_solns.shape == (num_leaves, p**3)

    def test_1(self) -> None:
        """3D non-uniform case"""
        p = 4
        q = 2

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=0.0, zmax=1.0
        )

        add_eight_children(add_to=root, root=root, q=q)

        add_eight_children(add_to=root.children[0], root=root, q=q)
        add_eight_children(add_to=root.children[0].children[0], root=root, q=q)

        num_leaves = len(get_all_leaves(root))
        print("test_1: num_leaves = ", num_leaves)

        t = create_solver_obj_3D(p, q, root)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**3))
        source_term = np.random.normal(size=(num_leaves, p**3))
        print("test_1: source_term.shape = ", source_term.shape)
        print("test_1: d_xx_coeffs.shape = ", d_xx_coeffs.shape)

        sidelens = jnp.array(
            [leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)]
        )

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
            sidelens=sidelens,
        )

        # Set the data in the Node objects
        leaves = get_all_leaves(t.root)
        for i, leaf in enumerate(leaves):
            leaf.DtN = DtN_arr[i]
            leaf.v_prime = v_prime_arr[i]
            leaf.Y = Y_arr[i]
            leaf.v = v_arr[i]

        _build_stage_3D(
            root=t.root,
            refinement_op=t.refinement_op,
            coarsening_op=t.coarsening_op,
            DtN_arr=DtN_arr,
            v_prime_arr=v_prime_arr,
            q=q,
        )

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = get_bdry_data_evals_lst_3D(t, f)
        print(
            "test_0: boundary_data_lst shapes = ",
            [x.shape for x in boundary_data_lst],
        )
        leaf_solns = _down_pass_3D(
            root=t.root,
            boundary_data=boundary_data_lst,
            refinement_op=t.refinement_op,
        )
        assert leaf_solns.shape == (num_leaves, p**3)


class Test__propogate_down_oct:
    def test_0(self) -> None:
        n_per_face = 3
        S_arr = np.random.normal(size=(12 * n_per_face, 24 * n_per_face))
        v_int_data = np.random.normal(size=(12 * n_per_face))

        bdry_data = [
            np.random.normal(size=(4 * n_per_face,)) for _ in range(6)
        ]

        out = _propogate_down_oct(
            S_arr,
            bdry_data,
            v_int_data,
            n_a_0=n_per_face,
            n_a_2=n_per_face,
            n_a_5=n_per_face,
            n_b_1=n_per_face,
            n_b_2=n_per_face,
            n_b_5=n_per_face,
            n_c_1=n_per_face,
            n_c_3=n_per_face,
            n_c_5=n_per_face,
            n_d_0=n_per_face,
            n_d_3=n_per_face,
            n_e_0=n_per_face,
            n_e_2=n_per_face,
            n_e_4=n_per_face,
            n_f_1=n_per_face,
            n_f_2=n_per_face,
            n_f_4=n_per_face,
            n_g_1=n_per_face,
            n_g_3=n_per_face,
            n_g_4=n_per_face,
            n_h_0=n_per_face,
            n_h_3=n_per_face,
            compression_lsts=[jnp.array([False]) for _ in range(24)],
            refinement_op=np.random.normal(size=(4 * n_per_face, n_per_face)),
        )
        expected_out_shape = (n_per_face,)
        assert len(out) == 8
        for i, x in enumerate(out):
            print("test_0: i = ", i, " len(x) = ", len(x))
            assert len(x) == 6
            for j, y in enumerate(x):
                print("test_0: j = ", j, " y.shape = ", y.shape)
                assert y.shape == expected_out_shape


if __name__ == "__main__":
    pytest.main()
