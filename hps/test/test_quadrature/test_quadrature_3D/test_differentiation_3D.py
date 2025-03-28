import numpy as np
import jax.numpy as jnp


from hps.src.quadrature.quadrature_utils import chebyshev_points
from hps.src.quadrature.quad_3D.grid_creation import (
    corners_to_cheby_points_lst,
)
from hps.src.quadrature.quad_3D.differentiation import (
    precompute_diff_operators,
)


class Test_precompute_diff_operators:
    def test_0(self) -> None:
        """Makes sure the outputs are the right shape."""
        p = 5
        half_side_len = 1 / 2
        du_dx, du_dy, du_dz, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz = (
            precompute_diff_operators(p, half_side_len)
        )

        assert du_dx.shape == (p**3, p**3)
        assert du_dy.shape == (p**3, p**3)
        assert du_dz.shape == (p**3, p**3)
        assert d_xx.shape == (p**3, p**3)
        assert d_yy.shape == (p**3, p**3)
        assert d_zz.shape == (p**3, p**3)
        assert d_xy.shape == (p**3, p**3)
        assert d_xz.shape == (p**3, p**3)
        assert d_yz.shape == (p**3, p**3)

    def test_1(self) -> None:
        """Tests du_dx, du_dy, du_dz work as expected with f(x, y, z) = x + y + z."""

        p = 5
        half_side_len = np.pi / 2
        corners = np.array(
            [
                [-np.pi / 2, -np.pi / 2, -np.pi / 2],
                [np.pi / 2, np.pi / 2, np.pi / 2],
            ]
        )
        pp = chebyshev_points(p)[0]
        cheby_pts = corners_to_cheby_points_lst(corners, pp)

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return x[:, 0] + x[:, 1] + x[:, 2]

        f_evals = f(cheby_pts)

        du_dx, du_dy, du_dz, _, _, _, _, _, _ = precompute_diff_operators(
            p, half_side_len
        )

        du_dx_f = du_dx @ f_evals
        du_dy_f = du_dy @ f_evals
        du_dz_f = du_dz @ f_evals

        assert jnp.allclose(du_dx_f, 1)
        assert jnp.allclose(du_dy_f, 1)
        assert jnp.allclose(du_dz_f, 1)

    def test_2(self) -> None:
        """Tests du_dx, du_dy, du_dz work as expected with f(x, y, z) = sin(x) + sin(y) + sin(z).
        Needs to have a relatively high p to fit the non-polynomial data."""

        p = 10
        half_side_len = 1 / 2
        corners = np.array([[0, 0, 0], [1, 1, 1]])
        pp = chebyshev_points(p)[0]
        cheby_pts = corners_to_cheby_points_lst(corners, pp)

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sin(x[:, 0]) + jnp.sin(x[:, 1]) + jnp.sin(x[:, 2])

        f_evals = f(cheby_pts)

        du_dx, du_dy, du_dz, _, _, _, _, _, _ = precompute_diff_operators(
            p, half_side_len
        )

        d_xx_f = du_dx @ f_evals
        d_yy_f = du_dy @ f_evals
        d_zz_f = du_dz @ f_evals

        assert jnp.allclose(d_xx_f, jnp.cos(cheby_pts[:, 0])), jnp.max(
            jnp.abs(d_xx_f - jnp.cos(cheby_pts[:, 0]))
        )
        assert jnp.allclose(d_yy_f, jnp.cos(cheby_pts[:, 1]))
        assert jnp.allclose(d_zz_f, jnp.cos(cheby_pts[:, 2]))

    def test_3(self) -> None:
        """Tests du_dxx, du_dyy, du_dzz work as expected with f(x, y, z) = x^2 + y^2 + z^2."""
        p = 5
        half_side_len = 1 / 2
        corners = np.array([[0, 0, 0], [1, 1, 1]])
        pp = chebyshev_points(p)[0]
        cheby_pts = corners_to_cheby_points_lst(corners, pp)

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2

        f_evals = f(cheby_pts)
        _, _, _, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz = (
            precompute_diff_operators(p, half_side_len)
        )

        d_xx_f = d_xx @ f_evals
        d_yy_f = d_yy @ f_evals
        d_zz_f = d_zz @ f_evals
        d_xy_f = d_xy @ f_evals
        d_xz_f = d_xz @ f_evals
        d_yz_f = d_yz @ f_evals

        assert jnp.allclose(d_xx_f, 2)
        assert jnp.allclose(d_yy_f, 2)
        assert jnp.allclose(d_zz_f, 2)
        assert jnp.allclose(d_xy_f, 0)
        assert jnp.allclose(d_xz_f, 0)
        assert jnp.allclose(d_yz_f, 0)

    def test_4(self) -> None:
        """Tests du_dxy, du_dxz, du_dyz work as expected with f(x, y, z) = xyz"""

        p = 5
        half_side_len = 1 / 2
        corners = np.array([[0, 0, 0], [1, 1, 1]])
        pp = chebyshev_points(p)[0]
        cheby_pts = corners_to_cheby_points_lst(corners, pp)

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return x[:, 0] * x[:, 1] * x[:, 2]

        f_evals = f(cheby_pts)
        _, _, _, _, _, _, d_xy, d_xz, d_yz = precompute_diff_operators(
            p, half_side_len
        )

        d_xy_f = d_xy @ f_evals
        d_xz_f = d_xz @ f_evals
        d_yz_f = d_yz @ f_evals

        assert jnp.allclose(d_xy_f, cheby_pts[:, 2])
        assert jnp.allclose(d_xz_f, cheby_pts[:, 1])
        assert jnp.allclose(d_yz_f, cheby_pts[:, 0])
