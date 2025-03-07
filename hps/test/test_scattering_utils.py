import pytest
import numpy as np
import jax.numpy as jnp

from hps.src.scattering_utils import (
    helmholtz_kernel,
    single_layer_potential,
    helmholtz_kernel_grad_y,
    double_layer_potential,
)


class Test_single_layer_potential:
    def test_0(self) -> None:
        n = 20
        from_pts = jnp.array(np.random.normal(size=(n, 2)))
        from_weights = jnp.array(np.random.normal(size=(n,)))
        m = 17
        to_pts = jnp.array(np.random.normal(size=(m, 2)))
        k = 1.0
        result = single_layer_potential(from_pts, from_weights, to_pts, k)
        assert result.shape == (m, n)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class Test_helmholtz_kernel:
    def test_0(self) -> None:
        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y = jnp.array(np.random.normal(size=(5, 2)))
        k = 1.0
        result = helmholtz_kernel(x, y, k)
        assert result.shape == (2, 5)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class Test_helmholtz_kernel_grad_y:
    def test_0(self) -> None:
        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y = jnp.array(np.random.normal(size=(5, 2)))
        k = 1.0
        result = helmholtz_kernel_grad_y(x, y, k)
        assert result.shape == (2, 5, 2)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class Test_double_layer_potential:
    def test_0(self) -> None:
        n = 20
        boundary_pts = jnp.array(np.random.normal(size=(n, 2)))
        k = 1.0
        result = double_layer_potential(boundary_pts, k)
        assert result.shape == (n, n)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


if __name__ == "__main__":
    pytest.main()
