# src/wave_scattering/scattering_utils.py
import h5py
import jax
import jax.numpy as jnp
from typing import Tuple

def load_SD_matrices(fp: str) -> Tuple[jnp.array, jnp.array]:
    """
    Load S and D matrices from a MATLAB v7.3 .mat file using h5py.

    Args:
        fp (str): File path to the .mat file

    Returns:
        Tuple[jnp.array, jnp.array]: A tuple containing (S, D) matrices

    Raises:
        ValueError: If the file doesn't contain required matrices
        FileNotFoundError: If the file doesn't exist
        RuntimeError: If there's an error reading the file
    """
    # try:

    # Open the file in read mode
    with h5py.File(fp, "r") as f:
        # Check if required matrices exist
        if "S" not in f or "D" not in f:
            raise ValueError("File must contain both 'S' and 'D' matrices")

        # Helper function to convert structured array to complex array
        def to_complex(structured_array):
            # Convert structured array to complex numpy array
            complex_array = (
                structured_array["real"] + 1j * structured_array["imag"]
            )
            # Convert to jax array
            return jnp.array(complex_array)

        # Load matrices and convert to complex numbers
        # Note: transpose the arrays as MATLAB stores them in column-major order
        S = to_complex(f["S"][:].T)
        D = to_complex(f["D"][:].T)

        return S, D


# @jax.jit
def get_DtN_from_ItI(R: jnp.array, eta: float) -> jnp.array:
    """
    Given an ItI matrix, generates the corresponding DtN matrix.

    equation 2.17 from the Gillman, Barnett, Martinsson paper.

    Implements the formula: T = -i eta (R - I)^{-1}(R + I)

    Args:
        R (jnp.array): Has shape (n, n)
        eta (float): Real number; parameter used in the ItI map creation.

    Returns:
        jnp.array: Has shape (n, n)
    """
    n = R.shape[0]
    I = jnp.eye(n)
    T = -1j * eta * jnp.linalg.solve(R - I, R + I)
    return T


# @jax.jit
def get_uin(
    k: float, pts: jnp.array, source_directions: jnp.array
) -> jnp.array:
    # source_vecs = jnp.array(
    #     [jnp.cos(source_directions), jnp.sin(source_directions)]
    # ).T
    # uin = jnp.exp(1j * k * jnp.dot(pts, source_vecs.T))
    source_vecs = jnp.stack(
        [jnp.cos(source_directions), jnp.sin(source_directions)],
        # Try flipping axes to match LS
        # [jnp.sin(-source_directions), jnp.cos(source_directions)],
        axis=0
    )
    uin = jnp.exp(1j * k * jnp.dot(pts, source_vecs))
    return uin


# @jax.jit
def get_uin_and_normals(
    k: float, bdry_pts: jnp.array, source_directions: jnp.array
) -> Tuple[jnp.array, jnp.array]:
    """
    Given the boundary points and the source directions, computes the incoming wave and the normal vectors.

    uin(x) = exp(i k <x,s>), where s = (cos(theta), sin(theta)) is the direction of the incoming plane wave.

    d uin(x) / dx = ik s_0 uin(x)
    d uin(x) / dy = ik s_1 uin(x)

    Args:
        k (float): Frequency of the incoming plane waves
        bdry_pts (jnp.array): Has shape (n, 2)
        source_directions (jnp.array): Has shape (n_sources,). Describes the direction of the incoming plane waves in radians.

    Returns:
        Tuple[jnp.array, jnp.array]: uin, normals. uin has shape (n,). normals has shape (n, 2).
    """
    n_per_side = bdry_pts.shape[0] // 4

    uin = get_uin(k, bdry_pts, source_directions)
    # print("get_uin_and_normals: uin shape: ", uin.shape)
    source_vecs = jnp.array(
        [jnp.cos(source_directions), jnp.sin(source_directions)]
    ).T
    # print("get_uin_and_normals: source_vecs shape: ", source_vecs.shape)

    normals = jnp.concatenate(
        [
            -1j
            * k
            * jnp.expand_dims(source_vecs[:, 1], axis=0)
            * uin[:n_per_side],  # -1 duin/dy
            1j
            * k
            * jnp.expand_dims(source_vecs[:, 0], axis=0)
            * uin[n_per_side : 2 * n_per_side],  # duin/dx
            1j
            * k
            * jnp.expand_dims(source_vecs[:, 1], axis=0)
            * uin[2 * n_per_side : 3 * n_per_side],  # duin/dy
            -1j
            * k
            * jnp.expand_dims(source_vecs[:, 0], axis=0)
            * uin[3 * n_per_side :],
        ]
    )

    return uin, normals
