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
)
from jaxhps._domain import Domain
from jaxhps._discretization_tree_operations_3D import add_eight_children
from jaxhps._pdeproblem import PDEProblem

from jaxhps.merge._adaptive_3D_DtN import (
    merge_stage_adaptive_3D_DtN,
)

from jaxhps.down_pass._adaptive_3D_DtN import (
    down_pass_adaptive_3D_DtN,
    _propagate_down_oct,
)


class Test_down_pass_adaptive_3D_DtN:
    def test_0(self, caplog) -> None:
        """3D uniform case"""
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
            leaf.data.v = v[i]
            leaf.data.Y = Y_arr[i]

        assert Y_arr.shape == (num_leaves, p**3, 6 * q**2)
        merge_stage_adaptive_3D_DtN(t, T_arr, h)

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = domain.get_adaptive_boundary_data_lst(f)
        print(
            "test_0: boundary_data_lst shapes = ",
            [x.shape for x in boundary_data_lst],
        )
        leaf_solns = down_pass_adaptive_3D_DtN(
            pde_problem=t, boundary_data=boundary_data_lst
        )
        assert leaf_solns.shape == (num_leaves, p**3)

    def test_1(self, caplog) -> None:
        """3D non-uniform case"""
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
                leaf.data.Y = Y_arr[i]
                leaf.data.v = v[i]

            assert Y_arr.shape == (num_leaves, p**3, 6 * q**2)
            merge_stage_adaptive_3D_DtN(t, T_arr, h)

        def f(x):
            return jnp.ones_like(x[..., 0])

        boundary_data_lst = domain.get_adaptive_boundary_data_lst(f)
        print(
            "test_0: boundary_data_lst shapes = ",
            [x.shape for x in boundary_data_lst],
        )
        leaf_solns = down_pass_adaptive_3D_DtN(
            pde_problem=t, boundary_data=boundary_data_lst
        )
        assert leaf_solns.shape == (num_leaves, p**3)
        jax.clear_caches()


class Test__propagate_down_oct:
    def test_0(self) -> None:
        n_per_face = 3
        S_arr = np.random.normal(size=(12 * n_per_face, 24 * n_per_face))
        v_int_data = np.random.normal(size=(12 * n_per_face))

        bdry_data = [
            np.random.normal(size=(4 * n_per_face,)) for _ in range(6)
        ]

        out = _propagate_down_oct(
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
            projection_lsts=[jnp.array([False]) for _ in range(24)],
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
        jax.clear_caches()
