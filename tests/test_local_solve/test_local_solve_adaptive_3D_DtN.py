import numpy as np
import jax
from jaxhps.local_solve._adaptive_3D_DtN import (
    local_solve_stage_adaptive_3D_DtN,
)
from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._domain import Domain
from jaxhps._discretization_tree_operations_3D import add_eight_children
from jaxhps._pdeproblem import PDEProblem


class Test__local_solve_stage_adaptive_3D_DtN:
    def test_0(self) -> None:
        """Tests the _local_solve_stage_3D function returns without error and returns the correct shape."""
        p = 6
        q = 4
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        n_leaves = 8 + 7
        add_eight_children(root)
        add_eight_children(root.children[0])
        domain = Domain(p=p, q=q, root=root)

        d_xx_coeffs = np.random.normal(size=(n_leaves, p**3))
        source_term = np.random.normal(size=(n_leaves, p**3))

        t = PDEProblem(
            domain=domain, D_xx_coefficients=d_xx_coeffs, source=source_term
        )

        Y_arr, T_arr, v, h = local_solve_stage_adaptive_3D_DtN(t)

        n_gauss_bdry = 6 * q**2

        assert Y_arr.shape == (n_leaves, p**3, n_gauss_bdry)
        assert T_arr.shape == (n_leaves, n_gauss_bdry, n_gauss_bdry)
        assert v.shape == (n_leaves, p**3)
        assert h.shape == (n_leaves, n_gauss_bdry)
        jax.clear_caches()
