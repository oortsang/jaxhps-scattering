import pytest
import numpy as np
import jax.numpy as jnp
import jax

from hps.src.poisson_boltzmann_eqn_helpers import (
    rho,
    permittivity,
    perm_2D,
    d_permittivity_d_x,
    d_permittivity_d_y,
    d_permittivity_d_x_2D,
    d_permittivity_d_y_2D,
    d_rho_d_x,
    d_rho_d_y,
)


class Test_rho:
    def test_0(self) -> None:
        """Make sure runs without error and return shape is correct."""
        key = jax.random.key(0)
        n_pts_0 = 27
        n_pts_1 = 4
        pts = jax.random.uniform(
            key, minval=-1.0, maxval=1.0, shape=(n_pts_0, n_pts_1, 3)
        )

        result = rho(pts)
        assert result.shape == (
            n_pts_0,
            n_pts_1,
        )
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


class Test_d_rho_d_x:
    def test_0(self) -> None:
        """Make sure runs without error and return shape is correct."""
        key = jax.random.key(0)
        n_pts_0 = 27
        n_pts_1 = 4
        pts = jax.random.uniform(
            key, minval=-1.0, maxval=1.0, shape=(n_pts_0, n_pts_1, 3)
        )

        result = d_rho_d_x(pts)
        assert result.shape == (
            n_pts_0,
            n_pts_1,
        )
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


class Test_d_rho_d_y:
    def test_0(self) -> None:
        """Make sure runs without error and return shape is correct."""
        key = jax.random.key(0)
        n_pts_0 = 27
        n_pts_1 = 4
        pts = jax.random.uniform(
            key, minval=-1.0, maxval=1.0, shape=(n_pts_0, n_pts_1, 3)
        )

        result = d_rho_d_y(pts)
        assert result.shape == (
            n_pts_0,
            n_pts_1,
        )
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


class Test_permittivity:
    def test_0(self) -> None:
        """Make sure runs without error and return shape is correct."""
        key = jax.random.key(0)
        n_pts_0 = 27
        n_pts_1 = 4
        pts = jax.random.uniform(
            key, minval=-1.0, maxval=1.0, shape=(n_pts_0, n_pts_1, 3)
        )

        result = permittivity(pts)
        assert result.shape == (
            n_pts_0,
            n_pts_1,
        )
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


class Test_perm_2D:
    def test_0(self) -> None:
        """Make sure runs without error and return shape is correct."""
        key = jax.random.key(0)
        n_pts_0 = 27
        n_pts_1 = 4
        pts = jax.random.uniform(
            key, minval=-1.0, maxval=1.0, shape=(n_pts_0, n_pts_1, 2)
        )

        result = perm_2D(pts)
        assert result.shape == (
            n_pts_0,
            n_pts_1,
        )
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


class Test_d_permittivity_d_x_2D:
    def test_0(self) -> None:
        """Make sure runs without error and return shape is correct."""
        key = jax.random.key(0)
        n_pts_0 = 27
        n_pts_1 = 4
        pts = jax.random.uniform(
            key, minval=-1.0, maxval=1.0, shape=(n_pts_0, n_pts_1, 2)
        )

        result = d_permittivity_d_x_2D(pts)
        assert result.shape == (
            n_pts_0,
            n_pts_1,
        )
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


class Test_d_permittivity_d_y_2D:
    def test_0(self) -> None:
        """Make sure runs without error and return shape is correct."""
        key = jax.random.key(0)
        n_pts_0 = 27
        n_pts_1 = 4
        pts = jax.random.uniform(
            key, minval=-1.0, maxval=1.0, shape=(n_pts_0, n_pts_1, 2)
        )

        result = d_permittivity_d_y_2D(pts)
        assert result.shape == (
            n_pts_0,
            n_pts_1,
        )
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))


if __name__ == "__main__":
    pytest.main()
