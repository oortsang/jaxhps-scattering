import logging
import jax
import numpy as np

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
