import jax
import jax.numpy as jnp
from typing import List


def down_pass_uniform_2D_DtN(
    boundary_data: jax.Array,
    S_lst: List[jax.Array],
    g_tilde_lst: List[jax.Array],
    Y_arr: jax.Array,
    v_arr: jax.Array,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> jax.Array:
    """
    Computes the downward pass of the HPS algorithm. This function takes the Dirichlet data
    at the boundary of the domain and propagates it down the tree to the leaf nodes.


    If Y_arr is None, the function will exit early after doing all of the downward propagation operations.

    Parameters
    ----------

    boundary_data : jax.Array
        An array specifying Dirichlet data on the boundary of the domain.  Has shape (n_bdry,)

    S_lst : List[jax.Array]
        A list of propagation operators. The first element of the list are the propagation operators for the nodes just above the leaves, and the last element of the list is the propagation operator for the root of the quadtree.

    g_tilde_lst : List[jax.Array]
        A list of incoming particular solution data along the merge interfaces. The first element of the list corresponds to the nodes just above the leaves, and the last element of the list corresponds to the root of the quadtree.

    Y_arr : jax.Array
        Matrices mapping the solution to the interior of the leaf nodes. Has shape (n_leaf, p^2, 4q).

    v_arr : jax.Array
        Particular solution data at the interior of the leaves. Has shape (n_leaf, p^2).

    device : jax.Device
        Where to perform the computation. Defaults to jax.devices()[0].

    host_device : jax.Device
        Where to place the output. Defaults to jax.devices("cpu")[0].


    Returns
    -------

    solns : jax.Array
        Has shape (n_leaves, p^2). Interior solution on all of the leaf nodes.

    """

    boundary_data = jax.device_put(boundary_data, device)
    Y_arr = jax.device_put(Y_arr, device)
    v_arr = jax.device_put(v_arr, device)
    S_lst = [jax.device_put(S_arr, device) for S_arr in S_lst]
    g_tilde_lst = [jax.device_put(g_tilde, device) for g_tilde in g_tilde_lst]

    n_levels = len(S_lst)

    # Reshape to (1, n_bdry)
    if len(boundary_data.shape) == 1:
        bdry_data = jnp.expand_dims(boundary_data, axis=0)
    else:
        bdry_data = boundary_data

    # propagate the Dirichlet data down the tree using the S maps.
    for level in range(n_levels - 1, -1, -1):
        S_arr = S_lst[level]
        g_tilde = g_tilde_lst[level]

        bdry_data = vmapped_propagate_down_2D_DtN(S_arr, bdry_data, g_tilde)
        # Reshape from (-1, 4, n_bdry) to (-1, n_bdry)
        n_bdry = bdry_data.shape[-1]
        bdry_data = bdry_data.reshape((-1, n_bdry))

    root_dirichlet_data = bdry_data

    if Y_arr is None:
        return root_dirichlet_data

    # Batched matrix multiplication to compute homog solution on all leaves
    leaf_homog_solns = jnp.einsum("ijk,ik->ij", Y_arr, root_dirichlet_data)
    leaf_solns = leaf_homog_solns + v_arr
    leaf_solns = jax.device_put(leaf_solns, host_device)
    return leaf_solns


@jax.jit
def _propagate_down_2D_DtN(
    S_arr: jax.Array,
    bdry_data: jax.Array,
    g_tilde: jax.Array,
) -> jax.Array:
    """
    Given homogeneous data on the boundary, interface homogeneous solution operator S, and
    interface particular solution data, this function returns the solution on the boundaries
    of the four children.

    suppose n_child is the number of quadrature points on EACH SIDE of a child node.

    Args:
        S_arr (jax.Array): Has shape (4 * n_child, 8 * n_child)
        bdry_data (jax.Array): 8 * n_child
        g_tilde (jax.Array): 4 * n_child

    Returns:
        jax.Array: Has shape (4, 4 * n_child)
    """

    n_child = bdry_data.shape[0] // 8

    g_int = S_arr @ bdry_data + g_tilde

    # All of these slices of g_int are propogating from OUTSIDE to INSIDE
    g_int_5 = g_int[:n_child]
    g_int_6 = g_int[n_child : 2 * n_child]
    g_int_7 = g_int[2 * n_child : 3 * n_child]
    g_int_8 = g_int[3 * n_child :]

    g_a = jnp.concatenate(
        [
            bdry_data[:n_child],  # S edge
            g_int_5,  # E edge
            jnp.flipud(g_int_8),  # N edge
            bdry_data[7 * n_child :],  # W edge
        ]
    )

    g_b = jnp.concatenate(
        [
            bdry_data[n_child : 3 * n_child],  # S edge, E edge
            g_int_6,  # N edge
            jnp.flipud(g_int_5),  # W edge
        ]
    )

    g_c = jnp.concatenate(
        [
            jnp.flipud(g_int_6),  # S edge
            bdry_data[3 * n_child : 5 * n_child],  # E edge, N edge
            g_int_7,  # W edge
        ]
    )

    g_d = jnp.concatenate(
        [
            g_int_8,  # S edge
            jnp.flipud(g_int_7),  # E edge
            bdry_data[5 * n_child : 7 * n_child],  # N edge, W edge
        ]
    )
    return jnp.stack([g_a, g_b, g_c, g_d])


vmapped_propagate_down_2D_DtN = jax.vmap(
    _propagate_down_2D_DtN, in_axes=(0, 0, 0), out_axes=0
)
