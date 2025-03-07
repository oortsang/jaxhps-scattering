import pytest
import jax
import jax.numpy as jnp

from hps.src.config import (
    DEVICE_MEM_BYTES,
    get_n_levels_2D,
    get_fused_chunksize_2D,
    DEVICE_ARR,
    HOST_DEVICE,
    GPU_AVAILABLE,
)


class Test_get_fused_chunksize_2D:
    def test_0(self) -> None:
        p = 16
        dtype = jnp.float64
        n_leaves = 16
        max_chunksize = get_fused_chunksize_2D(p, dtype, n_leaves)
        print("test_0: max_chunksize: ", max_chunksize)
        print("test_0: DEVICE_MEM_BYTES: ", DEVICE_MEM_BYTES)

        assert len(max_chunksize) == 2
        assert type(max_chunksize[0]) == int


class Test_get_n_levels_2D:
    def test_0(self) -> None:
        for l in range(1, 5):
            input = 4**l
            output = get_n_levels_2D(input)
            assert output == l


if __name__ == "__main__":
    pytest.main()
