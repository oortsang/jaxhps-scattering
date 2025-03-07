import logging
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike
import numpy as np
from typing import Tuple

# Logging stuff
jax_logger = logging.getLogger("jax")
jax_logger.setLevel(logging.WARNING)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)

# Jax enable double precision
jax.config.update("jax_enable_x64", True)

# Figure out if GPU is available
GPU_AVAILABLE = any("NVIDIA" in device.device_kind for device in jax.devices())


# Device configuration
DEVICE_ARR = np.array(jax.devices()).flatten()
jax.config.update("jax_default_device", jax.devices("cpu")[0])

DEVICE_MESH = jax.sharding.Mesh(DEVICE_ARR, axis_names=("x",))
HOST_DEVICE = jax.devices("cpu")[0]
# HOST_DEVICE = DEVICE_ARR[0]

mem_bytes_lst = []
for d in DEVICE_ARR:
    if d.memory_stats() is not None:
        mem_bytes_lst.append(d.memory_stats()["bytes_limit"])
DEVICE_MEM_BYTES = np.array(mem_bytes_lst)


def nearest_factor_of_four(x: int) -> int:
    return (x // 4) * 4


def device_put_wrapper(x: ArrayLike, device: jax.Device) -> jax.numpy.array:
    """Put an array on a device."""
    x_devices = x.devices()

    if len(x_devices) == 1 and x_devices.pop() == device:
        return x
    else:
        return jax.device_put(x, device)


def get_n_levels_2D(n_leaves: int) -> int:
    # log_4(n_leaves) = ln(n_leaves) / ln(4)
    return int(jnp.log(n_leaves) / jnp.log(4))


def get_n_levels_3D(n_leaves: int) -> int:
    # log_8(n_leaves) = ln(n_leaves) / ln(8)
    return int(jnp.log(n_leaves) / jnp.log(8))


def get_fused_chunksize_2D(p: int, dtype: DTypeLike, n_leaves: int) -> Tuple[int, int]:
    """Empirically, we can fit 4**7 leaves into GPU memory without a problem
    when p = 16. When p > 16, we will try 4**6 leaves, and keep adding if
    statements as needed.
    """
    n_levels = get_n_levels_2D(n_leaves)

    if not GPU_AVAILABLE:
        return n_leaves, n_levels

    if p == 7:  # Test case on smaller problem instances
        return min(4**2, n_leaves), min(2, n_levels)

    if dtype == jnp.complex128:
        return min(4**6, n_leaves), min(6, n_levels)

    else:
        return min(4**7, n_leaves), min(7, n_levels)


def get_chunksize_3D(p: int, n_leaves: int) -> int:
    if p == 8:
        return 2_000
    elif p == 10:
        return 500
    elif p == 12:
        return 100  # bummer how small this must be
    elif p == 16:
        return 20
    else:
        return 5_000
