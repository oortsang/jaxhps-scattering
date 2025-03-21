"""
This file contains utility functions for quadrature points and weights. It
defines Chebyshev and Gauss-Legendre quadrature points and weights.
It has utilites for interpolation maps between different quadratures, as well
as 1D differentiation matrices defined over 2D Chebyshev grids.
"""

from functools import partial

from typing import Tuple

import jax.numpy as jnp
import jax
import numpy as np


jax.config.update("jax_enable_x64", True)

EPS = jnp.finfo(jnp.float64).eps


@jax.jit
def differentiation_matrix_1d(
    points: jnp.ndarray,
) -> jnp.ndarray:
    """Creates a 1-D Chebyshev differentiation matrix as described in (T00) Ch 6. Expects points are Chebyshev points on [-1, 1].

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


def _chebyshev_points(n: int) -> jnp.ndarray:
    """Returns n Chebyshev points over the interval [-1, 1]

    out[i] = cos(pi * i / (n-1)) for i={0,...,n-1}

    Actually I return the reversed array so the smallest points come first.

    Args:
        n (int): number of Chebyshev points to return

    Returns:
        jnp.ndarray: The sampled points in [-1, 1] and the corresponding angles in [0, pi]
    """
    cos_args = jnp.arange(n, dtype=jnp.float64) / (n - 1)
    angles = jnp.flipud(jnp.pi * cos_args)
    pts = jnp.cos(angles)

    # Normalize by 1 / sqrt(1 - x^2)
    weights = jnp.sin(angles) ** 2 / (n - 1) * np.pi
    nrms = jnp.sqrt(1 - pts**2)
    weights = weights / nrms
    weights = weights.at[0].set(0.0)
    weights = weights.at[-1].set(0.0)
    return pts, weights


chebyshev_points = jax.jit(_chebyshev_points, static_argnums=(0,))


@partial(jax.jit, static_argnums=(0,))
def chebyshev_weights(n: int, bounds: jnp.array) -> jnp.array:
    """
    Generates weights for a Chebyshev quadrature rule with n points over the interval [a, b].

    Uses the Clenshaw-Curtis quadrature rule, specifically the version used in Chebfun:
    https://github.com/chebfun/chebfun/blob/f44234100d8af189d60e4bc533f5e98a0442a4c1/%40chebtech2/quadwts.m

    Args:
        n (int): Number of quadrature points
        bounds (jnp.array): Has shape (2,) and contains the interval endpoints [a, b]

    Returns:
        jnp.array: Has shape (n,) and contains the quadrature weights
    """
    a, b = bounds
    interval_len = b - a

    c = 2.0 / jnp.concatenate(
        [jnp.array([1.0]), 1.0 - jnp.arange(2, n, 2) ** 2]
    )

    if n % 2:
        # # Mirror for DCT via FFT
        start = n // 2
        c_slice = jnp.flip(c[1:start])
        c = jnp.concatenate([c, c_slice])

        w = jnp.fft.ifft(c).real
        w_out = jnp.concatenate([w, jnp.array([w[0] / 2])])
        w_out = w_out.at[0].set(w[0] / 2)
    else:
        c = 2.0 / jnp.concatenate(
            [jnp.array([1.0]), 1.0 - jnp.arange(2, n, 2) ** 2]
        )
        # Mirror for DCT via FFT
        start = n // 2 + 1
        c_slice = jnp.flip(c[1:start])
        c = jnp.concatenate([c, c_slice])

        # c = jnp.fft.ifftshift(c)
        w = jnp.fft.ifft(c).real
        w_out = jnp.concatenate([w, jnp.array([w[0] / 2])])
        w_out = w_out.at[0].set(w[0] / 2)

    # Scale by interval length
    w_out = w_out * interval_len / 2
    return w_out


@jax.jit
def barycentric_lagrange_interpolation_matrix(
    from_pts: jnp.ndarray, to_pts: jnp.ndarray
) -> jnp.ndarray:
    """
    Generates a Lagrange 1D polynomial interpolation matrix, which interpolates
    from the points in from_pts to the points in to_pts.

    This function uses the barycentric formula for Lagrange interpolation, from

    Berrut, J.-P., & Trefethen, L. N. (2004). Barycentric Lagrange Interpolation.

    Args:
        from_pts (jnp.ndarray): Has shape (p,)
        to_pts (jnp.ndarray): Has shape (n,)

    Returns:
        jnp.ndarray: Has shape (n,p)
    """
    p = from_pts.shape[0]
    n = to_pts.shape[0]

    # Compute the inverses of the Barycentric weights
    w = jnp.ones(p, dtype=jnp.float64)
    for j in range(p):
        for k in range(p):
            if j != k:
                w = w.at[j].mul(from_pts[j] - from_pts[k])

    # print("barycentric_lagrange_interpolation_matrix: w", w)

    # Normalizing factor is sum_j w_j / (x - x_j)
    norm_factors = jnp.zeros(n, dtype=jnp.float64)
    for i in range(p):
        norm_factors += 1 / (w[i] * (to_pts - from_pts[i]))

        # print("barycentric_lagrange_interpolation_matrix: norm_factors", norm_factors)

    # Compute the matrix
    matrix = jnp.zeros((n, p), dtype=jnp.float64)
    for i in range(n):
        for j in range(p):
            matrix = matrix.at[i, j].set(
                1 / ((to_pts[i] - from_pts[j]) * w[j] * norm_factors[i])
            )

    # Check if any of the source and target points overlap
    # This code is semantically the same as what comes after.
    # The code below is vectorized and is able to be compiled because it does not
    # use the conditionals on to_pts and from_pts.

    # for i in range(n):
    #     for j in range(p):
    #         if to_pts[i] == from_pts[j]:
    #             matrix = matrix.at[i, :].set(0)
    #             matrix = matrix.at[i, j].set(1)

    # Create a boolean mask for matching points
    matches = to_pts[:, None] == from_pts[None, :]  # Shape: (n, p)

    # Create row masks for any matching points
    has_match = matches.any(axis=1)  # Shape: (n,)

    # Update the matrix
    matrix = jnp.where(
        has_match[:, None],  # Broadcasting to shape (n, p)
        jnp.where(
            matches,
            1.0,  # Where points match
            0.0,  # Where points don't match but row has a match
        ),
        matrix,  # Keep original values where row has no matches
    )

    return matrix


@jax.jit
def barycentric_lagrange_2d_interpolation_matrix(
    from_pts_x: jnp.ndarray,
    from_pts_y: jnp.ndarray,
    to_pts_x: jnp.ndarray,
    to_pts_y: jnp.ndarray,
) -> jnp.ndarray:
    n = from_pts_x.shape[0]
    p = to_pts_x.shape[0]
    # print("barycentric_lagrange_2d_interpolation_matrix: n, p", n, p)

    # Compute the inverses of the barycentric weights for x and y dimensions.
    # w_x[j] = \prod_{k != j} (from_pts_x[j] - from_pts_x[k])
    w_x = jnp.ones(n, dtype=jnp.float64)
    w_y = jnp.ones(n, dtype=jnp.float64)
    for j in range(n):
        for k in range(n):
            if j != k:
                w_x = w_x.at[j].mul(from_pts_x[j] - from_pts_x[k])
                w_y = w_y.at[j].mul(from_pts_y[j] - from_pts_y[k])

    # Compute matrix of distances between x and y points.
    xdist = to_pts_x[None, :] - from_pts_x[:, None]
    ydist = to_pts_y[None, :] - from_pts_y[:, None]
    # print("barycentric_lagrange_2d_interpolation_matrix: xdist", xdist.shape)

    # Replace exact 0's with EPS. This is to avoid division by zero.
    # This is a bit of a hack and the proper way to do this is identifying which
    # rows/cols of the matrix need to be amended using 1D interpolation maps.
    # But this is a good quick fix.
    xdist = jnp.where(xdist == 0, EPS, xdist)
    ydist = jnp.where(ydist == 0, EPS, ydist)

    # Compute the normalization factors for x and y dimensions.
    norm_factors_x = jnp.sum(1 / (w_x[:, None] * (xdist)), axis=0)
    norm_factors_y = jnp.sum(1 / (w_y[:, None] * (ydist)), axis=0)

    # Compute the matrix, iterating over the y_pts first.
    i, j, k, l = jnp.indices((p, p, n, n))
    matrix = 1 / (
        xdist[k, i]
        * ydist[l, j]
        * w_x[k]
        * w_y[l]
        * norm_factors_x[i]
        * norm_factors_y[j]
    )

    matrix = matrix.reshape(p * p, n * n)

    return matrix


@jax.jit
def barycentric_lagrange_3d_interpolation_matrix(
    from_pts_x: jnp.ndarray,
    from_pts_y: jnp.ndarray,
    from_pts_z: jnp.ndarray,
    to_pts_x: jnp.ndarray,
    to_pts_y: jnp.ndarray,
    to_pts_z: jnp.ndarray,
) -> jnp.ndarray:
    """_summary_

    Args:
        from_pts_x (jnp.ndarray): Has shape (n,)
        from_pts_y (jnp.ndarray): Has shape (n,)
        from_pts_z (jnp.ndarray): Has shape (n,)
        to_pts_x (jnp.ndarray): Has shape (p,)
        to_pts_y (jnp.ndarray): Has shape (p,)
        to_pts_z (jnp.ndarray): Has shape (p,)

    Returns:
        jnp.ndarray: Has shape (p**3, n**3)
    """

    n = from_pts_x.shape[0]
    p = to_pts_x.shape[0]

    # Compute the inverses of the barycentric weights for x, y, and z dimensions.
    w_x = jnp.ones(n, dtype=jnp.float64)
    w_y = jnp.ones(n, dtype=jnp.float64)
    w_z = jnp.ones(n, dtype=jnp.float64)
    for j in range(n):
        for k in range(n):
            if j != k:
                w_x = w_x.at[j].mul(from_pts_x[j] - from_pts_x[k])
                w_y = w_y.at[j].mul(from_pts_y[j] - from_pts_y[k])
                w_z = w_z.at[j].mul(from_pts_z[j] - from_pts_z[k])

    # Compute the normalization factors for x, y, and z dimensions.
    xdist = to_pts_x[None, :] - from_pts_x[:, None]
    ydist = to_pts_y[None, :] - from_pts_y[:, None]
    zdist = to_pts_z[None, :] - from_pts_z[:, None]

    # Replace 0's in the denominator with EPS to avoid division by zero.
    xdist = jnp.where(xdist == 0, EPS, xdist)
    ydist = jnp.where(ydist == 0, EPS, ydist)
    zdist = jnp.where(zdist == 0, EPS, zdist)

    norm_factors_x = jnp.sum(1 / (w_x[:, None] * xdist), axis=0)
    norm_factors_y = jnp.sum(1 / (w_y[:, None] * ydist), axis=0)
    norm_factors_z = jnp.sum(1 / (w_z[:, None] * zdist), axis=0)

    # Compute the matrix, iterating over the z_pts first.
    i, j, k, l, m, o = jnp.indices((p, p, p, n, n, n))

    matrix = 1 / (
        xdist[l, i]
        * ydist[m, j]
        * zdist[o, k]
        * w_x[l]
        * w_y[m]
        * w_z[o]
        * norm_factors_x[i]
        * norm_factors_y[j]
        * norm_factors_z[k]
    )

    mat_shape = (p**3, n**3)
    matrix = matrix.reshape(mat_shape)
    return matrix


@jax.jit
def affine_transform(pts: jnp.ndarray, ab: jnp.ndarray) -> jnp.ndarray:
    """Affine transforms the points pts, which are assumed to be
    in the interval [-1, 1], to the interval [a, b].

    Args:
        pts (jnp.ndarray): Has shape (n,)
        ab (jnp.ndarray): Has shape (2,)
    Returns:
        jnp.ndarray: Has shape (n,)
    """
    a, b = ab
    return 0.5 * (b - a) * pts + 0.5 * (a + b)


def check_current_discretization(
    f_evals: jnp.ndarray,
    f_evals_refined: jnp.ndarray,
    refinement_op: jnp.ndarray,
    tol: float,
) -> bool:
    """Implements the check performed in section 3.1 of the Geldermans Gillman
    2019 paper.

    Takes the f_evals, interpolates them to the refined grid, and compares the
    interpolated values to the f_evals_refined. If the difference is less than
    tol, returns True. Else, returns False.


    Args:
        f_evals (jnp.ndarray): Has shape (n,)
        f_evals_refined (jnp.ndarray): Has shape (4n,) or in the 3D case (8n,)
        refinement_op (jnp.ndarray): Has shape (4n, n) or in the 3D case (8n, n)
        tol (float): Tolerance for the check

    Returns:
        bool: True if the current discretization is good, False otherwise
    """
    f_interp = refinement_op @ f_evals
    f_evals_nrm = jnp.linalg.norm(f_evals_refined)
    diff_nrm = jnp.linalg.norm(f_interp - f_evals_refined)
    err = diff_nrm / f_evals_nrm
    return err < tol


@jax.jit
def check_current_discretization_global_linf_norm(
    f_evals: jnp.ndarray,
    f_evals_refined: jnp.ndarray,
    refinement_op: jnp.ndarray,
    tol: float,
    global_linf_norm: float,
) -> Tuple[bool, float]:
    """Takes the f_evals, interpolates them to the refined grid, and computes the
    L_inf norm of the difference.

    Args:
        f_evals (jnp.ndarray): Has shape (n,)
        f_evals_refined (jnp.ndarray): Has shape (4n,) or in the 3D case (8n,)
        refinement_op (jnp.ndarray): Has shape (4n, n) or in the 3D case (8n, n)
        tol (float): Tolerance for the check
        global_linf_norm (float): The global L_inf norm of the function

    Returns:
        Tuple[bool, float]: True if the current discretization is good, False otherwise,
        and max(global_linf_norm, L_inf(f_evals_refined))
    """

    f_interp = refinement_op @ f_evals
    ref_max = jnp.max(jnp.abs(f_evals_refined))
    diff_nrm = jnp.max(jnp.abs(f_interp - f_evals_refined))
    nrm_const = jnp.max(jnp.array([global_linf_norm, ref_max]))
    err = diff_nrm / nrm_const
    return err < tol, nrm_const
