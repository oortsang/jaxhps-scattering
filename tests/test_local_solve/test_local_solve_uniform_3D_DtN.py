import numpy as np
import jax
from jaxhps.local_solve._uniform_3D_DtN import (
    local_solve_stage_uniform_3D_DtN,
    _gather_coeffs_3D,
)
from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem


class Test_local_solve_stage_uniform_3D_DtN:
    def test_0(self) -> None:
        """Tests the _local_solve_stage_3D function returns without error and returns the correct shape."""
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

        n_gauss_bdry = 6 * q**2

        assert Y_arr.shape == (n_leaves, p**3, n_gauss_bdry)
        assert T_arr.shape == (n_leaves, n_gauss_bdry, n_gauss_bdry)
        assert v.shape == (n_leaves, p**3)
        assert h.shape == (n_leaves, n_gauss_bdry)
        jax.clear_caches()

    def test_1(self) -> None:
        """Tests the _local_solve_stage_3D function returns without error and returns the correct shape
        when using multiple source terms."""
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

        n_gauss_bdry = 6 * q**2

        assert Y_arr.shape == (n_leaves, p**3, n_gauss_bdry)
        assert T_arr.shape == (n_leaves, n_gauss_bdry, n_gauss_bdry)
        assert v.shape == (n_leaves, p**3, nsrc)
        assert h.shape == (n_leaves, n_gauss_bdry, nsrc)
        jax.clear_caches()


class Test__gather_coeffs_3D:
    def test_0(self) -> None:
        """Make sure the function returns the correct shape."""
        n_leaf_nodes = 30
        p_cubed = 17

        xx_coeffs = np.random.normal(size=(n_leaf_nodes, p_cubed))
        yy_coeffs = np.random.normal(size=(n_leaf_nodes, p_cubed))
        i_coeffs = np.random.normal(size=(n_leaf_nodes, p_cubed))

        out_coeffs, out_bools = _gather_coeffs_3D(
            D_xx_coeffs=xx_coeffs, D_yy_coeffs=yy_coeffs, I_coeffs=i_coeffs
        )
        assert out_coeffs.shape == (3, n_leaf_nodes, p_cubed)
        assert out_bools.shape == (10,)
        assert out_bools[0].item() is True
        assert out_bools[1].item() is False
        assert out_bools[2].item() is True
        assert out_bools[3].item() is False
        assert out_bools[4].item() is False
        assert out_bools[5].item() is False
        assert out_bools[6].item() is False
        assert out_bools[7].item() is False
        assert out_bools[8].item() is False
        assert out_bools[9].item() is True
        jax.clear_caches()
