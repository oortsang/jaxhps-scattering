import logging
import jax.numpy as jnp
import numpy as np
import jax
from jaxhps.merge._uniform_3D_DtN import (
    merge_stage_uniform_3D_DtN,
)

from jaxhps.local_solve._uniform_3D_DtN import (
    local_solve_stage_uniform_3D_DtN,
)
from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
from jaxhps.down_pass._uniform_3D_DtN import (
    down_pass_uniform_3D_DtN,
    _propogate_down_oct_DtN,
)


class Test_down_pass_uniform_3D_DtN:
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

        logging.debug("test_0: S_arr_lst[-1] shape: %s", S_arr_lst[-1].shape)

        boundary_data = jnp.ones_like(t.domain.boundary_points[..., 0])

        logging.debug("test_0:bdry_data shape: %s", boundary_data.shape)

        solns = down_pass_uniform_3D_DtN(
            boundary_data=boundary_data,
            S_lst=S_arr_lst,
            g_tilde_lst=g_tilde_lst,
            Y_arr=Y_arr,
            v_arr=v,
        )
        assert solns.shape == (n_leaves, p**3)
        jax.clear_caches()


class Test__propogate_down_oct_DtN:
    def test_0(self) -> None:
        n_per_face = 3
        S_arr = np.random.normal(size=(12 * n_per_face, 24 * n_per_face))
        bdry_data = np.random.normal(size=(24 * n_per_face))
        v_int_data = np.random.normal(size=(12 * n_per_face))

        out = _propogate_down_oct_DtN(S_arr, bdry_data, v_int_data)
        expected_out_shape = (8, 6 * n_per_face)
        assert out.shape == expected_out_shape
        jax.clear_caches()
