import jax.numpy as jnp
import jax

key = jax.random.key(0)
N_ATOMS = 50
ATOM_CENTERS = jax.random.uniform(
    key, minval=-0.5, maxval=0.5, shape=(N_ATOMS, 3)
)
N_ATOMS_VDW = 1200
ATOM_CENTERS_VDW = jax.random.uniform(
    key, minval=-0.5, maxval=0.5, shape=(N_ATOMS_VDW, 3)
)
DELTA = 30.0 * 3 / 2
EPS_0 = 16.0
EPS_INF = 100.0
A = 10.0


@jax.jit
def perm_2D(x: jnp.array) -> jnp.array:
    # x has shape (..., 2)
    # Add zeros to the -1 axis of x
    print("perm_2D: x shape", x.shape)
    y = jnp.array([x[..., 0], x[..., 1], jnp.zeros_like(x[..., 0])])
    y = jnp.permute_dims(y, (1, 2, 0))
    print("perm_2D: y shape", y.shape)
    return permittivity(y)


@jax.jit
def rho_2D(x: jnp.array) -> jnp.array:
    # x has shape (..., 2)
    # Add zeros to the -1 axis of x
    y = jnp.array([x[..., 0], x[..., 1], jnp.zeros_like(x[..., 0])]).T
    in_shape = x.shape[:-1]
    y = y.reshape((-1, 3))
    return rho(y).reshape(in_shape)


@jax.jit
def rho(x: jnp.array) -> jnp.array:
    """
    rho is the charge density at point x

    x has shape [..., 3]
    return shape is [...,]

    inside the function, the input is flattened to shape (n, 3)
    """

    # sum over all atoms
    # exp(-delta * |x - x_i|^2)
    # want x to have shape (n, 1, 3)
    # want ATOM_CENTERS to have shape (1, n_atoms, 3)
    # thus diffs will have shape (n, n_atoms)
    in_shape = x.shape[:-1]

    x_expanded = jnp.reshape(x, (-1, 1, 3))
    atom_centers_expanded = jnp.expand_dims(ATOM_CENTERS, 0)
    diffs = x_expanded - atom_centers_expanded

    squared_diffs = diffs**2
    e = jnp.exp(-DELTA * jnp.sum(squared_diffs, axis=-1))
    e_summed = jnp.sum(e, axis=-1)
    return e_summed.reshape(in_shape)


@jax.jit
def d_rho_d_x(x: jnp.array) -> jnp.array:
    """
    This function is d/dx rho(x,y,z)

    = sum_i -2 delta (x_i - x) exp(-delta |x_i - x|^2)

    Args:
        x (jnp.array): Has shape (...,3)

    Returns:
        jnp.array: Has shape (...,)
    """

    in_shape = x.shape[:-1]

    x = jnp.reshape(x, (-1, 3))

    # These arrays should have shape (n, n_atoms, 3)
    diffs = jnp.expand_dims(x, 1) - jnp.expand_dims(ATOM_CENTERS, 0)
    squared_diffs = diffs**2

    # These arrays should have shape (n, n_atoms)
    e = jnp.exp(-DELTA * jnp.sum(squared_diffs, axis=-1))
    before_sum = -2 * DELTA * diffs[:, :, 0] * e

    return jnp.sum(before_sum, axis=-1).reshape(in_shape)


@jax.jit
def d_rho_d_y(x: jnp.array) -> jnp.array:
    """
    This function is d/dy rho(x,y,z)

    = sum_i -2 delta (y_i - y) exp(-delta |x_i - x|^2)

    Args:
        x (jnp.array): Has shape (...,3)

    Returns:
        jnp.array: Has shape (...,)
    """

    in_shape = x.shape[:-1]

    x = jnp.reshape(x, (-1, 3))

    # These arrays should have shape (n, n_atoms, 3)
    diffs = jnp.expand_dims(x, 1) - jnp.expand_dims(ATOM_CENTERS, 0)
    squared_diffs = diffs**2

    # These arrays should have shape (n, n_atoms)
    e = jnp.exp(-DELTA * jnp.sum(squared_diffs, axis=-1))
    before_sum = -2 * DELTA * diffs[:, :, 1] * e

    return jnp.sum(before_sum, axis=-1).reshape(in_shape)


@jax.jit
def d_rho_d_z(x: jnp.array) -> jnp.array:
    """
    This function is d/dz rho(x,y,z)

    = sum_i -2 delta (z_i - z) exp(-delta |x_i - x|^2)

    Args:
        x (jnp.array): Has shape (n,3)

    Returns:
        jnp.array: Has shape (n,)
    """
    in_shape = x.shape[:-1]

    x = jnp.reshape(x, (-1, 3))

    # These arrays should have shape (n, n_atoms, 3)
    diffs = jnp.expand_dims(x, 1) - jnp.expand_dims(ATOM_CENTERS, 0)
    squared_diffs = diffs**2

    # These arrays should have shape (n, n_atoms)
    e = jnp.exp(-DELTA * jnp.sum(squared_diffs, axis=-1))
    before_sum = -2 * DELTA * diffs[:, :, 2] * e

    return jnp.sum(before_sum, axis=-1).reshape(in_shape)


@jax.jit
def permittivity(x: jnp.array) -> jnp.array:
    """
    This is the permittivity at point x

    x has shape [n, 3]
    return shape is [n]
    """
    return EPS_0 + (EPS_INF - EPS_0) * jnp.exp(-A * rho(x))


@jax.jit
def d_permittivity_d_x(x: jnp.array) -> jnp.array:
    """
    This function is d /dx permittivity(x,y,z)

    Args:
        x (jnp.array): Has shape (n,3)

    Returns:
        jnp.array: Has shape (n,)
    """
    coeff = -A * (EPS_INF - EPS_0)
    return coeff * d_rho_d_x(x) * jnp.exp(-A * rho(x))


@jax.jit
def d_permittivity_d_y(x: jnp.array) -> jnp.array:
    """
    This function is d /dy permittivity(x,y,z)

    Args:
        x (jnp.array): Has shape (n,3)

    Returns:
        jnp.array: Has shape (n,)
    """
    coeff = -A * (EPS_INF - EPS_0)
    return coeff * d_rho_d_y(x) * jnp.exp(-A * rho(x))


@jax.jit
def d_permittivity_d_z(x: jnp.array) -> jnp.array:
    """
    This function is d /dz permittivity(x,y,z)

    Args:
        x (jnp.array): Has shape (n,3)

    Returns:
        jnp.array: Has shape (n,)
    """
    coeff = -A * (EPS_INF - EPS_0)
    return coeff * d_rho_d_z(x) * jnp.exp(-A * rho(x))


@jax.jit
def d_permittivity_d_x_2D(x: jnp.array) -> jnp.array:
    # x has shape (..., 2)
    y = jnp.array([x[..., 0], x[..., 1], jnp.zeros_like(x[..., 0])])
    y = jnp.permute_dims(y, (1, 2, 0))
    return d_permittivity_d_x(y)


@jax.jit
def d_permittivity_d_y_2D(x: jnp.array) -> jnp.array:
    # x has shape (..., 2)
    y = jnp.array([x[..., 0], x[..., 1], jnp.zeros_like(x[..., 0])])
    y = jnp.permute_dims(y, (1, 2, 0))
    return d_permittivity_d_y(y)


@jax.jit
def vdw_permittivity(x: jnp.array) -> jnp.array:
    """
    This is permittivity at point x
    x has shape (..., 3)
    return shape is (...)

    vdw_perm(x) = q(x) eps_0 + (1 - q(x))eps_infty
    """
    q_eval = q(x)
    return q_eval * EPS_0 + (1 - q_eval) * EPS_INF


@jax.jit
def q(x: jnp.array) -> jnp.array:
    """
    This is called rho_mol in the Nicholls paper.
    q(x) = 1 - prod(1 - exp(-delta || x - z_i||^2))
    """

    in_shape = x.shape[:-1]

    x = jnp.reshape(x, (-1, 3))

    # These arrays should have shape (n, n_atoms, 3)
    diffs = jnp.expand_dims(x, 1) - jnp.expand_dims(ATOM_CENTERS, 0)
    squared_diffs = diffs**2

    # These arrays should have shape (n, n_atoms)
    e = jnp.exp(-DELTA * jnp.sum(squared_diffs, axis=-1))
    q = 1 - jnp.prod(1 - e, axis=-1)

    return q.reshape(in_shape)


@jax.jit
def d_vdw_permittivity_d_x(x: jnp.array) -> jnp.array:
    """
    This function is d /dx vdw_permittivity(x,y,z)

    Args:
        x (jnp.array): Has shape (...,3)

    Returns:
        jnp.array: Has shape (...,)
    """
    # Create a tangent vector that's 1.0 for x and 0.0 for y and z
    # This needs to match the shape of inputs
    # batch_shape = x.shape[:-1]

    # Create a one-hot tangent vector where only the x-component is 1.0
    tangent = jnp.zeros_like(x)
    tangent = tangent.at[..., 0].set(1.0)

    # Compute the Jacobian-vector product
    _, derivative = jax.jvp(vdw_permittivity, (x,), (tangent,))

    return derivative


@jax.jit
def d_vdw_permittivity_d_y(x: jnp.array) -> jnp.array:
    """
    This function is d /dy vdw_permittivity(x,y,z)

    Args:
        x (jnp.array): Has shape (...,3)

    Returns:
        jnp.array: Has shape (...,)
    """
    # Create a tangent vector that's 1.0 for y and 0.0 for x and z
    # This needs to match the shape of inputs
    # batch_shape = x.shape[:-1]

    # Create a one-hot tangent vector where only the y-component is 1.0
    tangent = jnp.zeros_like(x)
    tangent = tangent.at[..., 1].set(1.0)

    # Compute the Jacobian-vector product
    _, derivative = jax.jvp(vdw_permittivity, (x,), (tangent,))

    return derivative


@jax.jit
def d_vdw_permittivity_d_z(x: jnp.array) -> jnp.array:
    """
    This function is d /dz vdw_permittivity(x,y,z)

    Args:
        x (jnp.array): Has shape (...,3)

    Returns:
        jnp.array: Has shape (...,)
    """
    # Create a tangent vector that's 1.0 for z and 0.0 for x and y
    # This needs to match the shape of inputs
    # batch_shape = x.shape[:-1]

    # Create a one-hot tangent vector where only the z-component is 1.0
    tangent = jnp.zeros_like(x)
    tangent = tangent.at[..., 2].set(1.0)

    # Compute the Jacobian-vector product
    _, derivative = jax.jvp(vdw_permittivity, (x,), (tangent,))

    return derivative
