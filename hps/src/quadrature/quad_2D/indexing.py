from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np


# This is only ever called with n=p**2, so compiling it with
# static_argnums=(0,) is not awful.
@partial(jax.jit, static_argnums=(0,))
def _rearrange_indices(n: int) -> jnp.ndarray:
    """This function gives the array indices to rearrange the 2D Cheby grid so that the
    4(p-1) boundary points are listed first, starting at the SW corner and going clockwise around the
    boundary. The interior points are listed after.
    """

    idxes = np.zeros(n**2, dtype=int)
    # S border
    for i, j in enumerate(range(n - 1, n**2, n)):
        idxes[i] = j
    # W border
    for i, j in enumerate(range(n**2 - 2, n**2 - n - 1, -1)):
        idxes[n + i] = j
    # N border
    for i, j in enumerate(range(n**2 - 2 * n, 0, -n)):
        idxes[2 * n - 1 + i] = j
    # S border
    for i, j in enumerate(range(1, n - 1)):
        idxes[3 * n - 2 + i] = j
    # Loop through the indices in column-rasterized form and fill in the ones from the interior.
    current_idx = 4 * n - 4
    nums = np.arange(n**2)
    for i in nums:
        if i not in idxes:
            idxes[current_idx] = i
            current_idx += 1
        else:
            continue

    return jnp.array(idxes)


def indexing_for_refinement_operator(p: int) -> jnp.array:
    """Returns row and column indexing to rearrange the refinement_operator matrix.

    Before reordering, that matrix has rows corresponding to a meshgrid of (cheby_pts_1d, cheby_pts_1d)
    and cols corresponding to a meshgrid of (cheby_pts_refined, cheby_pts_refined). After reordering,
    we want the rows to be ordered in the standard way, putting the exterior points first and then the
    interior points. We also want the columns to be ordered in the standard way, putting each of the
    four blocks of the meshgrid together and then ordering the points in each block so that the exterior
    points come first.

    Returns:
        jnp.array: r: row indices to rearrange the rows of the matrix
                   c: column indices to rearrange the columns of the matrix
    """
    col_idxes = _rearrange_indices(p)

    ii = jnp.arange(4 * p**2)

    # a is where x is in the first half and y is in the first half
    a_bools = jnp.logical_and(ii % (2 * p) >= p, (ii // (2 * p)) % (2 * p) < p)
    a_idxes = ii[a_bools]
    a_idxes = a_idxes[col_idxes]

    # b is where x is in the second half and y is in the first half
    b_bools = jnp.logical_and(ii % (2 * p) >= p, (ii // (2 * p)) % (2 * p) >= p)
    b_idxes = ii[b_bools]
    b_idxes = b_idxes[col_idxes]

    # c is where x is in the second half and y is in the second half
    c_bools = jnp.logical_and(ii % (2 * p) < p, (ii // (2 * p)) % (2 * p) >= p)
    c_idxes = ii[c_bools]
    c_idxes = c_idxes[col_idxes]

    # d is where x is in the first half and y is in the second half
    d_bools = jnp.logical_and(ii % (2 * p) < p, (ii // (2 * p)) % (2 * p) < p)
    d_idxes = ii[d_bools]
    d_idxes = d_idxes[col_idxes]

    row_idxes = jnp.concatenate(
        [
            a_idxes,
            b_idxes,
            c_idxes,
            d_idxes,
        ]
    )
    return row_idxes, col_idxes
