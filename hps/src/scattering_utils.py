import jax.numpy as jnp
import scipy.special as sp


def _hankel1(x: jnp.array) -> jnp.array:
    pass


def _derivative_hankel1(x: jnp.array) -> jnp.array:
    pass


def helmholtz_kernel(x: jnp.array, y: jnp.array, k: float) -> jnp.array:
    """
    phi(x - y) = i / 4 * H_0^{(1)}(k |x - y|)

    Args:
        x (jnp.array): Has shape (n, 2)
        y (jnp.array): Has shape (m, 2)
        k (float): Wavenumber

    Returns:
        jnp.array: Has shape (n, m). The (i,j) entry is
        phi(x_i - y_j)
    """
    x = x[:, None, :]
    y = y[None, :, :]

    r = jnp.linalg.norm(x - y, axis=2)
    return jnp.nan_to_num(1j / 4 * sp.hankel1(0, k * r), nan=0.0)


def helmholtz_kernel_grad_y(x: jnp.array, y: jnp.array, k: float) -> jnp.array:
    """
    \nabla_y phi(||x - y||) = i / 4 * k * H_1^{(1)}(k ||x - y||) * (y - x) / ||x - y||

    Args:
        x (jnp.array): Has shape (n, 2)
        y (jnp.array): Has shape (m, 2)
        k (float): Wavenumber

    Returns:
        jnp.array: Has shape (n, m, 2). The (i,j) entry is \nabla_y phi(x_i - y_j)
    """
    x = x[:, None, :]
    y = y[None, :, :]

    r = jnp.linalg.norm(x - y, axis=2)

    coeff = 1j * k / 4
    h_evals = sp.hankel1(1, k * r)[:, :, None]
    diff = y - x
    r = r[:, :, None]
    # print("helmholtz_kernel_grad_y: diff.shape", diff.shape)
    # print("helmholtz_kernel_grad_y: r.shape", r.shape)
    # print("helmholtz_kernel_grad_y: h_evals.shape", h_evals.shape)
    grad = coeff * h_evals * diff / r
    return grad


def single_layer_potential(
    from_pts: jnp.array, from_weights: jnp.array, to_pts: jnp.array, k: float
) -> jnp.array:
    """
    Args:
        boundary_pts (jnp.array): _description_
        k (float): _description_
        eta (float): _description_

    Returns:
        jnp.array: _description_
    """

    S = helmholtz_kernel(to_pts, from_pts, k)

    # Normalize each row by the number of points
    S = S * from_weights
    return S


def double_layer_potential(boundary_pts: jnp.array, k: float) -> jnp.array:
    """
    Double-layer potential for the Helmholtz equation.

    d(x, y) = n(y) * grad_y \\phi(x - y)

    where n(y) is the outward normal to the boundary at y.
    and \\phi(x - y) is the Helmholtz kernel.

    This function returns a matrix D where
    D_{ij} = d(x_i, y_j)

    Args:
        boundary_pts (jnp.array): _description_
        k (float): _description_

    Returns:
        jnp.array: _description_
    """
    n = boundary_pts.shape[0]
    n_per_side = n // 4

    # Split the boundary into 4 sides
    s_points = boundary_pts[:n_per_side]
    e_points = boundary_pts[n_per_side : 2 * n_per_side]
    n_points = boundary_pts[2 * n_per_side : 3 * n_per_side]
    w_points = boundary_pts[3 * n_per_side :]

    s_normal = jnp.array([0.0, 1.0])
    e_normal = jnp.array([1.0, 0.0])
    n_normal = jnp.array([0.0, -1.0])
    w_normal = jnp.array([-1.0, 0.0])

    s_cols = helmholtz_kernel_grad_y(boundary_pts, s_points, k) @ s_normal
    e_cols = helmholtz_kernel_grad_y(boundary_pts, e_points, k) @ e_normal
    n_cols = helmholtz_kernel_grad_y(boundary_pts, n_points, k) @ n_normal
    w_cols = helmholtz_kernel_grad_y(boundary_pts, w_points, k) @ w_normal

    return jnp.nan_to_num(
        jnp.hstack([s_cols, e_cols, n_cols, w_cols]), nan=0.0
    )
