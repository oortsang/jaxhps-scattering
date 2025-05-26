import numpy as np
import jax.numpy as jnp
import jax
from jaxhps.local_solve._adaptive_3D_DtN import (
    local_solve_stage_adaptive_3D_DtN,
)
import logging
from jaxhps._discretization_tree import (
    DiscretizationNode3D,
    get_all_leaves,
    get_nodes_at_level,
)
from jaxhps._domain import Domain
from jaxhps._discretization_tree_operations_3D import add_eight_children
from jaxhps._pdeproblem import PDEProblem

from jaxhps.merge._adaptive_3D_DtN import (
    merge_stage_adaptive_3D_DtN,
    _oct_merge,
    oct_merge_nonuniform_whole_level,
    node_to_oct_merge_outputs,
    is_node_type,
)


def check_node_data(node: DiscretizationNode3D) -> None:
    n_bdry = node.n_0 + node.n_1 + node.n_2 + node.n_3 + node.n_4 + node.n_5
    assert node.data.T is not None
    assert node.data.T.shape == (n_bdry, n_bdry)
    assert node.data.h is not None
    assert node.data.h.shape == (n_bdry,)
    if len(node.children):
        assert node.data.S is not None
        assert node.data.g_tilde is not None
        assert node.data.S.shape[0] == node.data.g_tilde.shape[0]
        assert node.data.S.shape[-1] == n_bdry


class Test_merge_stage_adaptive_3D_DtN:
    def test_0(self, caplog) -> None:
        """Makes sure things run correctly under uniform refinement."""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        l = 2
        num_leaves = 8**l
        root = DiscretizationNode3D(
            xmin=-1,
            xmax=1,
            ymin=-1,
            ymax=1,
            zmin=-1,
            zmax=1,
        )
        # Manually do l=2 levels
        add_eight_children(root, root=root, q=q)
        for c in root.children:
            add_eight_children(c, root=root, q=q)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**3))
        source_term = np.random.normal(size=(num_leaves, p**3))

        domain = Domain(p=p, q=q, root=root)
        t = PDEProblem(
            domain=domain, D_xx_coefficients=d_xx_coeffs, source=source_term
        )

        Y_arr, T_arr, v, h = local_solve_stage_adaptive_3D_DtN(t)
        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.domain.root)):
            leaf.data.T = T_arr[i]
            leaf.data.h = h[i]

        assert Y_arr.shape == (num_leaves, p**3, 6 * q**2)
        merge_stage_adaptive_3D_DtN(t, T_arr, h)
        # Check that the DtN and v_prime arrays are set in the tree object for levels 0, 1, 2.
        for node in get_nodes_at_level(t.domain.root, 0):
            check_node_data(node)

        for node in get_nodes_at_level(t.domain.root, 1):
            check_node_data(node)

    def test_1(self, caplog) -> None:
        """Makes sure things run correctly under non-uniform refinement."""
        caplog.set_level(logging.DEBUG)

        for child_idx in range(6):
            # Run the test once for each child to make sure all of the indexing
            # is working correctly.
            print("test_1: child_idx = ", child_idx)
            p = 6
            q = 4
            root = DiscretizationNode3D(
                xmin=-1,
                xmax=1,
                ymin=-1,
                ymax=1,
                zmin=-1,
                zmax=1,
            )
            print("test_1: q = ", q)
            print("test_1: q**2 = ", q**2)

            add_eight_children(root, root=root, q=q)
            add_eight_children(root.children[child_idx], root=root, q=q)
            add_eight_children(
                root.children[child_idx].children[child_idx], root=root, q=q
            )
            domain = Domain(p=p, q=q, root=root)

            num_leaves = len(get_all_leaves(root))
            d_xx_coeffs = np.random.normal(size=(num_leaves, p**3))
            source_term = np.random.normal(size=(num_leaves, p**3))
            t = PDEProblem(
                domain=domain,
                D_xx_coefficients=d_xx_coeffs,
                source=source_term,
            )
            Y_arr, T_arr, v, h = local_solve_stage_adaptive_3D_DtN(t)
            # Set the output DtN and v_prime arrays in the tree object.
            for i, leaf in enumerate(get_all_leaves(t.domain.root)):
                leaf.data.T = T_arr[i]
                leaf.data.h = h[i]

            assert Y_arr.shape == (num_leaves, p**3, 6 * q**2)
            merge_stage_adaptive_3D_DtN(t, T_arr, h)
            for node in get_nodes_at_level(t.domain.root, 0):
                check_node_data(node)

            for node in get_nodes_at_level(t.domain.root, 1):
                check_node_data(node)

    def test_2(self) -> None:
        """Makes sure things run correctly under non-uniform refinement."""
        # Run the test once for each child to make sure all of the indexing
        # is working correctly.
        p = 6
        q = 4
        root = DiscretizationNode3D(
            xmin=-1,
            xmax=1,
            ymin=-1,
            ymax=1,
            zmin=-1,
            zmax=1,
        )
        add_eight_children(root, root=root, q=q)
        add_eight_children(root.children[1], root=root, q=q)
        add_eight_children(root.children[2], root=root, q=q)
        add_eight_children(root.children[3], root=root, q=q)
        num_leaves = len(get_all_leaves(root))
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**3))
        source_term = np.random.normal(size=(num_leaves, p**3))

        domain = Domain(p=p, q=q, root=root)
        t = PDEProblem(
            domain=domain, D_xx_coefficients=d_xx_coeffs, source=source_term
        )
        Y_arr, T_arr, v, h = local_solve_stage_adaptive_3D_DtN(t)
        # Set the output DtN and v_prime arrays in the tree object.
        for i, leaf in enumerate(get_all_leaves(t.domain.root)):
            leaf.data.T = T_arr[i]
            leaf.data.h = h[i]

        assert Y_arr.shape == (num_leaves, p**3, 6 * q**2)
        merge_stage_adaptive_3D_DtN(t, T_arr, h)

        for node in get_nodes_at_level(t.domain.root, 0):
            check_node_data(node)

        for node in get_nodes_at_level(t.domain.root, 1):
            check_node_data(node)
        jax.clear_caches()


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
        T_a = np.random.normal(
            size=(n_gauss_bdry_refined, n_gauss_bdry_refined)
        )
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
        jax.clear_caches()


class Test_node_to_oct_merge_outputs:
    def test_0(self) -> None:
        q = 2
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        n_per_panel = q**2
        add_eight_children(root, root, q)

        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))

        root.L_1f2 = L_1f2
        root.L_2f1 = L_2f1

        for child in root.children:
            # Set DtN and v_prime attributes.
            child.data.T = np.random.normal(
                size=(6 * n_per_panel, 6 * n_per_panel)
            )
            child.data.h = np.random.normal(size=(6 * n_per_panel,))
            # child.L_2f1 = L_2f1
            # child.L_1f2 = L_1f2

        S, T, v_prime_ext, v_int = node_to_oct_merge_outputs(
            root,
        )

        assert T.shape == (24 * n_per_panel, 24 * n_per_panel)
        assert S.shape == (12 * n_per_panel, 24 * n_per_panel)

        jax.clear_caches()


class Test_oct_merge_nonuniform_whole_level:
    def test_0(self) -> None:
        q = 2
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        n_per_panel = q**2
        add_eight_children(root, root, q)
        for child in root.children:
            add_eight_children(child, root, q)

        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))

        for l in get_all_leaves(root):
            # Set DtN and v_prime attributes.
            l.data.T = np.random.normal(
                size=(6 * n_per_panel, 6 * n_per_panel)
            )
            l.data.h = np.random.normal(size=(6 * n_per_panel,))

        nodes_level_1 = get_nodes_at_level(root, 1)
        assert len(nodes_level_1) == 8

        oct_merge_nonuniform_whole_level(
            L_1f4=L_1f2,
            L_4f1=L_2f1,
            nodes_this_level=nodes_level_1,
        )
        jax.clear_caches()


class Test_is_node_type:
    def test_0(self) -> None:
        x = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        x.DtN = np.random.normal(size=(10, 10))

        assert is_node_type(x)
        assert not is_node_type(x.data.T)
        jax.clear_caches()
