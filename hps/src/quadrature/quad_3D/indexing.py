from functools import partial
from typing import Tuple, List
import numpy as np
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0,))
def rearrange_indices_ext_int(p: int) -> jnp.ndarray:

    # out = np.zeros(p**3, dtype=int)
    idxes = np.arange(p**3)

    # The first p^2 points and last p^2 points are the points where x=x_min and x=x_max
    # respectively. This happens when (idx // p**2) == 0 or (idx // p**2) == p-1.
    left_face = idxes // p**2 == 0
    right_face = idxes // p**2 == p - 1

    mask = np.logical_or(left_face, right_face)

    # The top and bottom faces of the cube are the faces where y=y_min and y=y_max respectively.
    # This happens when (idx // p) % p == 0 or (idx // p) % p == p-1.
    bottom_face = np.logical_and((idxes // p) % p == 0, ~mask)
    top_face = np.logical_and((idxes // p) % p == p - 1, ~mask)

    mask = np.logical_or(mask, np.logical_or(bottom_face, top_face))

    # The front and back faces of the cube are the faces where z=z_min and z=z_max respectively.
    # This happens when idx % p == 0 or idx % p == p-1.
    front_face = np.logical_and(idxes % p == 0, ~mask)
    back_face = np.logical_and(idxes % p == p - 1, ~mask)

    mask = np.logical_or(mask, np.logical_or(front_face, back_face))

    out = jnp.concatenate(
        [
            idxes[left_face],
            idxes[right_face],
            idxes[bottom_face],
            idxes[top_face],
            idxes[front_face],
            idxes[back_face],
            idxes[~mask],
        ]
    )

    return out


############################################
# These next functions are designed to get the
# indices of certain faces of the p^3 Chebyshev grid.
# I am using numbers [1,...,6] for the faces; look at my notes for
# the definition of these numbers.


@partial(jax.jit, static_argnums=(0,))
def get_face_1_idxes(p: int) -> None:
    """Face lying paralel to the (y,z) plane farthest in the -x direction."""
    return jnp.arange(p**2)


@partial(jax.jit, static_argnums=(0,))
def get_face_2_idxes(p: int) -> None:
    """Face lying paralel to the (y,z) plane farthest in the +x direction."""
    return jnp.arange(p**2) + p**2


@partial(jax.jit, static_argnums=(0,))
def get_face_3_idxes(p: int) -> None:
    """Face lying paralel to the (x,z) plane farthest in the -y direction."""
    o = jnp.concatenate(
        [
            jnp.arange(p),
            jnp.arange(2 * p**2, 3 * p**2 - 2 * p),
            jnp.arange(p**2, p**2 + p),
        ]
    )
    return o


@partial(jax.jit, static_argnums=(0,))
def get_face_4_idxes(p: int) -> None:
    """Face lying paralel to the (x,z) plane farthest in the +y direction."""
    o = jnp.concatenate(
        [
            jnp.arange(p * (p - 1), p**2),
            jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p),
            jnp.arange(2 * p**2 - p, 2 * p**2),
        ]
    )
    return o


@partial(jax.jit, static_argnums=(0,))
def get_face_5_idxes(p: int) -> None:
    """Face lying paralel to the (x,y) plane farthest in the -z direction."""
    first_col = jnp.arange(0, p**2, p)
    last_col = jnp.arange(p**2, 2 * p**2, p)

    first_row = jnp.arange(2 * p**2, 3 * p**2 - 2 * p, p)
    last_row = jnp.arange(3 * p**2 - 2 * p, 4 * p**2 - 4 * p, p)

    int_idxes = jnp.arange(4 * p**2 - 4 * p, 4 * p**2 - 4 - p + (p - 2) ** 2)

    # The order we put these togethr in is first_col, then the interior is one from first_row,
    # then p-2 from int_idxes, then one from last_row. Repeat until out of elements in last_row.
    # Finally, add the last_col.
    out = jnp.zeros(p**2, dtype=int)
    out = out.at[:p].set(first_col)
    out = out.at[-p:].set(last_col)

    counter = p
    for i in range(p - 2):
        out = out.at[counter].set(first_row[i])
        counter += 1
        out = out.at[counter : counter + p - 2].set(
            int_idxes[i * (p - 2) : (i + 1) * (p - 2)]
        )
        counter += p - 2
        out = out.at[counter].set(last_row[i])
        counter += 1

    return out


@partial(jax.jit, static_argnums=(0,))
def get_face_6_idxes(p: int) -> None:
    """Face lying paralel to the (x,y) plane farthest in the +z direction."""
    first_col = jnp.arange(p - 1, p**2, p)
    last_col = jnp.arange(p**2 + p - 1, 2 * p**2, p)

    first_row = jnp.arange(2 * p**2 + p - 1, 3 * p**2 - 2 * p, p)
    last_row = jnp.arange(3 * p**2 - 2 * p + p - 1, 4 * p**2 - 4 * p, p)

    u = p**3 - (p - 2) ** 3
    l = u - (p - 2) ** 2
    int_idxes = jnp.arange(l, u)
    # The order we put these togethr in is first_col, then the interior is one from first_row,
    # then p-2 from int_idxes, then one from last_row. Repeat until out of elements in last_row.
    # Finally, add the last_col.
    out = jnp.zeros(p**2, dtype=int)
    out = out.at[:p].set(first_col)
    out = out.at[-p:].set(last_col)

    counter = p
    for i in range(p - 2):
        out = out.at[counter].set(first_row[i])
        counter += 1
        out = out.at[counter : counter + p - 2].set(
            int_idxes[i * (p - 2) : (i + 1) * (p - 2)]
        )
        counter += p - 2
        out = out.at[counter].set(last_row[i])
        counter += 1

    return out


##############################################
# These functions are for getting submatrices and subvectors
# of the DtN maps and v_prime arrays. These are used in the
# _oct_merge function.


@jax.jit
def into_column_first_order(
    q_idxes: jnp.array,
    idxes_a: jnp.array,
    idxes_b: jnp.array,
    idxes_c: jnp.array,
    idxes_d: jnp.array,
) -> jnp.array:
    """
    Supposes there are four regions, arranged like this:

    b d
    a c

    where the indices in each region tabulate the points in column-first order from ymin to ymax, then working from xmin to xmax.
    We want to merge all four index sets to be in the same column-first order. This function does that.
    """
    q = q_idxes.shape[0]
    idxes_ab = jnp.zeros(2 * idxes_a.shape[0], dtype=int)
    for i in range(q):
        idxes_ab = idxes_ab.at[i * 2 * q : i * q * 2 + q].set(
            idxes_a[i * q : (i + 1) * q]
        )
        idxes_ab = idxes_ab.at[i * q * 2 + q : i * q * 2 + 2 * q].set(
            idxes_b[i * q : (i + 1) * q]
        )

    idxes_cd = jnp.zeros(2 * idxes_c.shape[0], dtype=int)
    for i in range(q):
        idxes_cd = idxes_cd.at[i * 2 * q : i * q * 2 + q].set(
            idxes_c[i * q : (i + 1) * q]
        )
        idxes_cd = idxes_cd.at[i * q * 2 + q : i * q * 2 + 2 * q].set(
            idxes_d[i * q : (i + 1) * q]
        )

    return jnp.concatenate([idxes_ab, idxes_cd])


def indexing_for_refinement_operator(p: int) -> jnp.array:
    col_idxes = rearrange_indices_ext_int(p)

    ii = jnp.arange(8 * p**3)

    # part a is where x is in the first half, y is in the first half, and z is in the second half.
    a_bools = jnp.logical_and(
        ii % (2 * p) >= p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) < p, (ii // ((2 * p) ** 2)) % (2 * p) < p
        ),
    )
    a_idxes = ii[a_bools]
    a_idxes = a_idxes[col_idxes]

    # part b is where x is in the second half, y is in the first half, and z is in the second half.
    b_bools = jnp.logical_and(
        ii % (2 * p) >= p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) < p, (ii // ((2 * p) ** 2)) % (2 * p) >= p
        ),
    )
    b_idxes = ii[b_bools]
    b_idxes = b_idxes[col_idxes]

    # part c is where x is in the second half, y is in the second half, and z is in the second half.
    c_bools = jnp.logical_and(
        ii % (2 * p) >= p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) >= p, (ii // ((2 * p) ** 2)) % (2 * p) >= p
        ),
    )
    c_idxes = ii[c_bools]
    c_idxes = c_idxes[col_idxes]

    # part d is where x is in the first half, y is in the second half, and z is in the second half.
    d_bools = jnp.logical_and(
        ii % (2 * p) >= p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) >= p, (ii // ((2 * p) ** 2)) % (2 * p) < p
        ),
    )
    d_idxes = ii[d_bools]
    d_idxes = d_idxes[col_idxes]

    # part e is where x is in the first half, y is in the first half, and z is in the first half.
    e_bools = jnp.logical_and(
        ii % (2 * p) < p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) < p, (ii // ((2 * p) ** 2)) % (2 * p) < p
        ),
    )
    e_idxes = ii[e_bools]
    e_idxes = e_idxes[col_idxes]

    # part f is where x is in the second half, y is in the first half, and z is in the first half.
    f_bools = jnp.logical_and(
        ii % (2 * p) < p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) < p, (ii // ((2 * p) ** 2)) % (2 * p) >= p
        ),
    )
    f_idxes = ii[f_bools]
    f_idxes = f_idxes[col_idxes]

    # part g is where x is in the second half, y is in the second half, and z is in the first half.
    g_bools = jnp.logical_and(
        ii % (2 * p) < p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) >= p, (ii // ((2 * p) ** 2)) % (2 * p) >= p
        ),
    )
    g_idxes = ii[g_bools]
    g_idxes = g_idxes[col_idxes]

    # part h is where x is in the first half, y is in the second half, and z is in the first half.
    h_bools = jnp.logical_and(
        ii % (2 * p) < p,
        jnp.logical_and(
            (ii // (2 * p)) % (2 * p) >= p, (ii // ((2 * p) ** 2)) % (2 * p) < p
        ),
    )
    h_idxes = ii[h_bools]
    h_idxes = h_idxes[col_idxes]

    row_idxes = jnp.concatenate(
        [a_idxes, b_idxes, c_idxes, d_idxes, e_idxes, f_idxes, g_idxes, h_idxes]
    )

    return row_idxes, col_idxes
