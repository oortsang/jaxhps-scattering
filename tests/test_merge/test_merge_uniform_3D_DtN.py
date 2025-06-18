import numpy as np
import logging
from jaxhps.merge._uniform_3D_DtN import (
    merge_stage_uniform_3D_DtN,
    _uniform_oct_merge_DtN,
)
import jax

from jaxhps.local_solve._uniform_3D_DtN import (
    local_solve_stage_uniform_3D_DtN,
)
from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem


class Test_merge_stage_uniform_3D_DtN:
    def test_0(self, caplog) -> None:
        # Make sure it runs with correct inputs and outputs.
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        l = 1
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        n_leaves = 8**l
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx_coeffs = np.random.normal(size=(n_leaves, p**3))
        source_term = np.random.normal(size=(n_leaves, p**3))

        t = PDEProblem(
            domain=domain, D_xx_coefficients=d_xx_coeffs, source=source_term
        )

        Y_arr, T_arr, v, h = local_solve_stage_uniform_3D_DtN(t)
        assert Y_arr.shape == (n_leaves, p**3, 6 * q**2)

        S_arr_lst, g_tilde_lst = merge_stage_uniform_3D_DtN(
            T_arr=T_arr, h_arr=h, l=l
        )
        assert len(S_arr_lst) == l
        assert len(g_tilde_lst) == l
        for i in range(l):
            assert S_arr_lst[i].shape[-2] == g_tilde_lst[i].shape[-1]
        jax.clear_caches()

    def test_1(self, caplog) -> None:
        # Make sure it runs with correct inputs and outputs.
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        l = 1
        nsrc = 3
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        n_leaves = 8**l
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx_coeffs = np.random.normal(size=(n_leaves, p**3))
        source_term = np.random.normal(size=(n_leaves, p**3, nsrc))

        t = PDEProblem(
            domain=domain, D_xx_coefficients=d_xx_coeffs, source=source_term
        )

        Y_arr, T_arr, v, h = local_solve_stage_uniform_3D_DtN(t)
        assert Y_arr.shape == (n_leaves, p**3, 6 * q**2)

        S_arr_lst, g_tilde_lst = merge_stage_uniform_3D_DtN(
            T_arr=T_arr, h_arr=h, l=l
        )
        assert len(S_arr_lst) == l
        assert len(g_tilde_lst) == l
        for i in range(l):
            logging.debug(
                "S_arr_lst[%d].shape = %s, g_tilde_lst[%d].shape = %s",
                i,
                S_arr_lst[i].shape,
                i,
                g_tilde_lst[i].shape,
            )
            assert S_arr_lst[i].shape[-2] == g_tilde_lst[i].shape[0]
        jax.clear_caches()


class Test__uniform_oct_merge_DtN:
    def test_0(self):
        q = 2
        n_gauss_bdry = 6 * q**2
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
        q_idxes = np.arange(q)
        S, T, v_prime_ext, v_int = _uniform_oct_merge_DtN(
            q_idxes=q_idxes,
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
        )

        assert S.shape == (12 * q**2, 24 * q**2)
        assert T.shape == (24 * q**2, 24 * q**2)
        assert v_prime_ext.shape == (24 * q**2,)
        assert v_int.shape == (12 * q**2,)
        jax.clear_caches()
