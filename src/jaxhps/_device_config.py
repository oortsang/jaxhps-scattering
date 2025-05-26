import logging
import jax
import numpy as np
import jax.numpy as jnp

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

#: Array of acceleration devices (GPUs) available for computation. If no GPUs are available, this will have a single entry for the CPU device.
DEVICE_ARR = np.array(jax.devices()).flatten()

DEVICE_MESH = jax.sharding.Mesh(DEVICE_ARR, axis_names=("x",))

#: The CPU device. Not used for computation but is often used for storing data that needs to be moved off the GPU.
HOST_DEVICE = jax.devices("cpu")[0]


def local_solve_chunksize_2D(p: int, dtype: jax.typing.DTypeLike) -> int:
    """
    Estimates the chunksize that can be used for the local solve stage in 2D probelms.

    Rounds to the nearest power of four that will fit on the device.

    Args:
        p (int): Chebyshev polynomial order.

        dtype (jax.Dtype): Datatype of the input data.

    Returns:

        int: The chunksize that can be used for the local solve stage.
    """

    if p == 7:
        return 4**2

    if dtype == jnp.complex128:
        return 4**6

    return 4**7


def local_solve_chunksize_3D(p: int, dtype: jax.typing.DTypeLike) -> int:
    """
    Estimates the chunksize that can be used for the local solve stage in 3D probelms.

    Args:
        p (int): Chebyshev polynomial order.

        dtype (jax.Dtype): Datatype of the input data.

    Returns:

        int: The chunksize that can be used for the local solve stage.
    """

    if p <= 8:
        return 2_000
    elif p <= 10:
        return 500
    elif p <= 12:
        return 100  # bummer how small this must be
    else:
        return 20
