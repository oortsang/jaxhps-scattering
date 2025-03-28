import jax.numpy as jnp
import jax


jax.config.update("jax_enable_x64", True)


@jax.jit
def differentiation_matrix_1D(
    points: jnp.ndarray,
) -> jnp.ndarray:
    """
    Creates a 1-D Chebyshev differentiation matrix as described in [4]_ Ch 6.

    Expects Chebyshev points on the interval [-1, 1].

    Args:
        points (jnp.ndarray): Has shape (p,)

    Returns:
        jnp.ndarray: Has shape (p,p)
    """
    p = points.shape[0]
    # print(p)

    # Here's the code from the MATLAB recipe book
    # c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
    # X = repmat(x,1,N+1);
    # dX = X-X';
    # D = (c*(1./c)')./(dX+(eye(N+1))); % off-diagonal entries
    # D = D - diag(sum(D'));

    # Here's the jax version Owen wrote
    c = jnp.ones(p)
    c = c.at[0].set(2)
    c = c.at[-1].set(2)
    for i in range(1, p, 2):
        c = c.at[i].set(-1 * c[i])
    x = jnp.expand_dims(points, -1).repeat(p, axis=1)
    dx = x - jnp.transpose(x)
    coeff = jnp.outer(c, 1 / c)
    d = coeff / (dx + jnp.eye(p))
    dd = jnp.diag(jnp.sum(d, axis=1))

    d = d - dd

    return d
