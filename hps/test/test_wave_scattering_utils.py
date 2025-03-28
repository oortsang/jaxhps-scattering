import pytest
import numpy as np
import jax.numpy as jnp
import jax
from hps.src.wave_scattering_utils import (
    get_uin_and_normals,
    get_uin,
)
from hps.src.quadrature.quad_2D.grid_creation import (
    get_all_boundary_gauss_legendre_points,
)
from hps.src.quadrature.trees import Node, add_uniform_levels

jax.config.update("jax_enable_x64", True)


class Test_get_uin:
    def test_0(self) -> None:
        n = 16
        n_src = 7
        k = 2.0
        gauss_bdry_pts = jnp.array(
            np.random.randn(n, 2).astype(np.float64), dtype=jnp.float64
        )

        source_directions = jnp.array(
            np.random.randn(n_src).astype(np.float64), dtype=jnp.float64
        )
        print("test_0: source_directions dtype: ", source_directions.dtype)

        uin = get_uin(k, gauss_bdry_pts, source_directions)

        assert uin.shape == (n, n_src)
        assert uin.dtype == jnp.complex128


class Test_get_uin_and_normals:
    def test_0(self) -> None:
        n = 16
        n_src = 7
        k = 2.0
        gauss_bdry_pts = jnp.array(
            np.random.randn(n, 2).astype(np.float64), dtype=jnp.float64
        )

        source_directions = jnp.array(
            np.random.randn(n_src).astype(np.float64), dtype=jnp.float64
        )

        uin, normals = get_uin_and_normals(
            k, gauss_bdry_pts, source_directions
        )

        assert uin.shape == (n, n_src)
        assert uin.dtype == jnp.complex128
        assert normals.shape == (n, n_src)
        assert normals.dtype == jnp.complex128

    def test_1(self) -> None:
        """Check correctness of normals"""
        q = 3
        l = 2
        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        add_uniform_levels(root=root, l=l, q=q)
        bdry_pts = get_all_boundary_gauss_legendre_points(q, root)

        n = bdry_pts.shape[0]
        src_theta = jnp.array([0.0, jnp.pi])
        k = 6.0
        uin, normals = get_uin_and_normals(k, bdry_pts, src_theta)
        s_0 = jnp.array([1.0, 0.0])
        # s_1 = jnp.array([0.0, 1.0])

        normals_0 = normals[:, 0]
        # normals_1 = normals[:, 1]

        # Compute expected normals 0
        expected_normals_0 = jnp.concatenate(
            [
                -1j * k * s_0[1] * uin[: n // 4, 0],
                1j * k * s_0[0] * uin[n // 4 : n // 2, 0],
                1j * k * s_0[1] * uin[n // 2 : 3 * n // 4, 0],
                -1j * k * s_0[0] * uin[3 * n // 4 :, 0],
            ]
        )
        print("test_1: expected_normals_0 shape: ", expected_normals_0.shape)
        print("test_1: normals_0 shape: ", normals_0.shape)
        assert jnp.allclose(normals_0, expected_normals_0)


if __name__ == "__main__":
    pytest.main()
