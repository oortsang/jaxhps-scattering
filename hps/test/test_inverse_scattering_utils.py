import pytest
import numpy as np
import jax.numpy as jnp

from hps.src.inverse_scattering_utils import (
    q_point_sources,
    source_locations_to_scattered_field,
)


class Test_q_point_sources:
    def test_0(self) -> None:

        a = 3
        b = 4
        n = 5

        x = jnp.array(np.random.normal(size=(a, b, 2)))
        source_locations = jnp.array(np.random.normal(size=(n, 2)))
        result = q_point_sources(x, source_locations)
        assert result.shape == (a, b)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


if __name__ == "__main__":
    pytest.main()
