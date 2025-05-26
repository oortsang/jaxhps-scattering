import logging
import jax.numpy as jnp
import jax
import numpy as np

from jaxhps.down_pass._uniform_2D_DtN import (
    down_pass_uniform_2D_DtN,
    _propagate_down_2D_DtN,
)


from jaxhps.merge._uniform_2D_DtN import (
    merge_stage_uniform_2D_DtN,
)

from jaxhps.local_solve._uniform_2D_DtN import (
    local_solve_stage_uniform_2D_DtN,
)
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem


class Test_down_pass_uniform_2D_DtN:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        l = 3
        num_leaves = 4**l
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx_coeffs = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
        )
        (
            Y_arr,
            T_arr,
            v_arr,
            h_arr,
        ) = local_solve_stage_uniform_2D_DtN(pde_problem=t)

        S_arr_lst, g_tilde_lst = merge_stage_uniform_2D_DtN(
            T_arr=T_arr, h_arr=h_arr, l=l
        )

        boundary_data = jnp.ones_like(t.domain.boundary_points[..., 0])

        leaf_solns = down_pass_uniform_2D_DtN(
            boundary_data=boundary_data,
            S_lst=S_arr_lst,
            g_tilde_lst=g_tilde_lst,
            Y_arr=Y_arr,
            v_arr=v_arr,
        )
        assert leaf_solns.shape == (num_leaves, p**2)
        jax.clear_caches()

    def test_1(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        l = 3
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        n_bdry = domain.boundary_points.shape[0]

        S_lst = []
        S_lst.insert(0, jnp.ones((1, n_bdry // 2, n_bdry)))
        S_lst.insert(0, jnp.ones((4, n_bdry // 4, n_bdry // 2)))

        g_tilde_lst = []
        g_tilde_lst.insert(0, jnp.ones((1, n_bdry // 2)))
        g_tilde_lst.insert(0, jnp.ones((4, n_bdry // 4)))

        bdry_data = jnp.ones((n_bdry))
        out = down_pass_uniform_2D_DtN(
            boundary_data=bdry_data,
            S_lst=S_lst,
            g_tilde_lst=g_tilde_lst,
            Y_arr=None,
            v_arr=None,
        )
        expected_out_shape = (16, n_bdry // 4)
        assert out.shape == expected_out_shape


class Test__propagate_down_2D_DtN:
    def test_0(self) -> None:
        """Tests to make sure returns without error."""
        n_child = 8
        S_arr = np.random.normal(size=(4 * n_child, 8 * n_child))
        bdry_data = np.random.normal(size=(8 * n_child))
        v_int = np.random.normal(size=(4 * n_child))

        out = _propagate_down_2D_DtN(S_arr, bdry_data, v_int)
        expected_out_shape = (4, 4 * n_child)
        assert out.shape == expected_out_shape

    def test_1(self) -> None:
        n_child = 8
        S_arr = np.random.normal(size=(4 * n_child, 8 * n_child))
        bdry_data = np.random.normal(size=(8 * n_child))
        v_int = np.random.normal(size=(4 * n_child))

        out = _propagate_down_2D_DtN(S_arr, bdry_data, v_int)

        g_a = out[0]
        g_b = out[1]
        g_c = out[2]
        g_d = out[3]

        # Check the interfaces match up

        # Edge 5
        assert jnp.allclose(
            g_a[n_child : 2 * n_child], jnp.flipud(g_b[3 * n_child :])
        )
        # Edge 6
        assert jnp.allclose(
            g_b[2 * n_child : 3 * n_child], jnp.flipud(g_c[:n_child])
        )
        # Edge 7
        assert jnp.allclose(
            g_c[3 * n_child :], jnp.flipud(g_d[n_child : 2 * n_child])
        )
        # Edge 8
        assert jnp.allclose(
            g_d[:n_child], jnp.flipud(g_a[2 * n_child : 3 * n_child])
        )
        jax.clear_caches()
