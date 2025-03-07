import jax.numpy as jnp

from hps.accuracy_checks.test_cases_3D import (
    K_DIRICHLET,
    K_DU_DX,
    K_DU_DY,
    K_DU_DZ,
    K_XX_COEFF,
    K_YY_COEFF,
    K_ZZ_COEFF,
    K_SOURCE,
    K_PART_SOLN,
    K_PART_SOLN_DUDX,
    K_PART_SOLN_DUDY,
    K_PART_SOLN_DUDZ,
)


def polynomial_dirichlet_data_0(x):
    # Assumes x has shape (n, 2).
    # f(x, y) = x^2 - y^2
    return jnp.square(x[..., 0]) - jnp.square(x[..., 1])


def dudx_polynomial_dirichlet_data_0(x):
    # Assumes x has shape (n, 2).
    # df/dx = 2x
    return 2 * x[..., 0]


def dudy_polynomial_dirichlet_data_0(x):
    # Assumes x has shape (n, 2).
    # df/dy = -2y
    return -2 * x[..., 1]


def polynomial_dirichlet_data_1(x):
    # Assumes x has shape (..., 2)
    # f(x,y) = 3 x^2 - 3 y^2
    return 3 * jnp.square(x[..., 0]) - 3 * jnp.square(x[..., 1])


def dudx_polynomial_dirichlet_data_1(x):
    # Assumes x has shape (..., 2)
    # df/dx = 6x
    return 6 * x[..., 0]


def dudy_polynomial_dirichlet_data_1(x):
    # Assumes x has shape (..., 2)
    # df/dy = -6y
    return -6 * x[..., 1]


def source_polynomial_dirichlet_data_1(x):
    # Assumes x has shape (..., 2)
    # f(x, y) = 6x^2 - 18y
    return 6 * jnp.square(x[..., 0]) - 18 * x[..., 1]


def coeff_fn_dxx_polynomial_dirichlet_data_1(x):
    # Assumes x has shape (..., 2)
    # f(x,y) = x^2
    return jnp.square(x[..., 0])


def coeff_fn_dyy_polynomial_dirichlet_data_1(x):
    # Assumes x has shape (..., 2)
    # f(x,y) = 3y
    return 3 * x[..., 1]


def nonpoly_dirichlet_data_0(x):
    # Assumes x has shape (n, 2).
    # f(x, y) = e^x sin(y)
    return jnp.exp(x[..., 0]) * jnp.sin(x[..., 1])


def nonpoly_dirichlet_data_1(x):
    # f(x,y) = sin(x) e^y
    return jnp.sin(x[..., 0]) * jnp.exp(x[..., 1])


def dudx_nonpoly_dirichlet_data_0(x):
    # Assumes x has shape (n, 2).
    # df/dx = e^x sin(y)
    return jnp.exp(x[:, 0]) * jnp.sin(x[:, 1])


def dudx_nonpoly_dirichlet_data_1(x):
    # df/dx = cos(x) e^y
    return jnp.cos(x[..., 0]) * jnp.exp(x[..., 1])


def dudy_nonpoly_dirichlet_data_0(x):
    # Assumes x has shape (n, 2).
    # df/dy = e^x cos(y)
    return jnp.exp(x[:, 0]) * jnp.cos(x[:, 1])


def dudy_nonpoly_dirichlet_data_1(x):
    # df/dy = sin(x) e^y
    return jnp.sin(x[..., 0]) * jnp.exp(x[..., 1])


def default_lap_coeffs(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.ones_like(x[..., 0])


def default_zero_source(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x[..., 0])


TEST_CASE_NONCONSTANT_COEFF_POLY = {
    "dirichlet_data_fn": polynomial_dirichlet_data_1,
    "d_xx_coeff_fn": coeff_fn_dxx_polynomial_dirichlet_data_1,
    "d_yy_coeff_fn": coeff_fn_dyy_polynomial_dirichlet_data_1,
    "source_fn": source_polynomial_dirichlet_data_1,
    "du_dx_fn": dudx_polynomial_dirichlet_data_1,
    "du_dy_fn": dudy_polynomial_dirichlet_data_1,
}


TEST_CASE_POISSON_POLY = {
    K_DIRICHLET: polynomial_dirichlet_data_0,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_SOURCE: default_zero_source,
    K_DU_DX: dudx_polynomial_dirichlet_data_0,
    K_DU_DY: dudy_polynomial_dirichlet_data_0,
}

TEST_CASE_POISSON_NONPOLY = {
    "dirichlet_data_fn": nonpoly_dirichlet_data_0,
    "d_xx_coeff_fn": default_lap_coeffs,
    "d_yy_coeff_fn": default_lap_coeffs,
    "source_fn": default_zero_source,
    "du_dx_fn": dudx_nonpoly_dirichlet_data_0,
    "du_dy_fn": dudy_nonpoly_dirichlet_data_0,
    "part_soln_fn": default_zero_source,
}

TEST_CASE_POISSON_NONPOLY_1 = {
    "dirichlet_data_fn": nonpoly_dirichlet_data_1,
    "d_xx_coeff_fn": default_lap_coeffs,
    "d_yy_coeff_fn": default_lap_coeffs,
    "source_fn": default_zero_source,
    "du_dx_fn": dudx_nonpoly_dirichlet_data_1,
    "du_dy_fn": dudy_nonpoly_dirichlet_data_1,
}


def part_soln_fn(x: jnp.ndarray) -> jnp.ndarray:
    # v(x,y) = y(x^2 - (pi/2)^2)(y^2 - (pi/2)^2)
    term_1 = x[..., 1] * (jnp.square(x[..., 1]) - (jnp.pi / 2) ** 2)
    term_2 = jnp.square(x[..., 0]) - (jnp.pi / 2) ** 2
    return term_1 * term_2


def source_fn(x: jnp.ndarray) -> jnp.ndarray:
    # f(x, y) = 6y(x^2 - (pi/2)^2) + 2y(y^2 - (pi/2)^2)
    term_1 = 2 * x[..., 1] * (jnp.square(x[..., 1]) - (jnp.pi / 2) ** 2)
    term_2 = 6 * x[..., 1] * (jnp.square(x[..., 0]) - (jnp.pi / 2) ** 2)
    return term_1 + term_2


def d_dx_part_soln_fn(x: jnp.ndarray) -> jnp.ndarray:
    # dv/dx = 2xy(y^2 - (pi/2)^2)
    term_1 = jnp.square(x[..., 1]) - (jnp.pi / 2) ** 2
    return 2 * x[..., 0] * x[..., 1] * term_1


def d_dy_part_soln_fn(x: jnp.ndarray) -> jnp.ndarray:
    # dv/dy = (3y^2 - (pi/2)^2)(x^2 - (pi/2)^2)
    term_1 = jnp.square(x[..., 0]) - (jnp.pi / 2) ** 2
    term_2 = 3 * jnp.square(x[..., 1]) - (jnp.pi / 2) ** 2
    return term_1 * term_2


TEST_CASE_POLY_PART_HOMOG = {
    "dirichlet_data_fn": polynomial_dirichlet_data_0,
    "d_xx_coeff_fn": default_lap_coeffs,
    "d_yy_coeff_fn": default_lap_coeffs,
    "particular_soln_fn": part_soln_fn,
    "homogeneous_soln_fn": polynomial_dirichlet_data_0,
    "particular_dx_fn": d_dx_part_soln_fn,
    "particular_dy_fn": d_dy_part_soln_fn,
    "source_fn": source_fn,
    "du_dx_fn": dudx_polynomial_dirichlet_data_0,
    "du_dy_fn": dudy_polynomial_dirichlet_data_0,
}


def polynomial_dirichlet_data_3(x: jnp.array) -> jnp.array:
    # u(x,y) = x^2 - 3y
    return jnp.square(x[..., 0]) - 3 * x[..., 1]


def dudx_polynomial_dirichlet_data_3(x: jnp.array) -> jnp.array:
    # du/dx = 2x
    return 2 * x[..., 0]


def dudy_polynomial_dirichlet_data_3(x: jnp.array) -> jnp.array:
    # du/dy = -3
    return -3 * jnp.ones_like(x[..., 1])


def source_polynomial_dirichlet_data_3(x: jnp.array) -> jnp.array:
    # f(x, y) = 2
    return 2 * jnp.ones_like(x[..., 0])


TEST_CASE_POLY_ITI = {
    "dirichlet_data_fn": polynomial_dirichlet_data_3,
    "d_xx_coeff_fn": default_lap_coeffs,
    "d_yy_coeff_fn": default_lap_coeffs,
    "source_fn": source_polynomial_dirichlet_data_3,
    "du_dx_fn": dudx_polynomial_dirichlet_data_3,
    "du_dy_fn": dudy_polynomial_dirichlet_data_3,
}


def nonpoly_dirichlet_data_3(x: jnp.array) -> jnp.array:
    # u(x,y) = sin(3x) + cos(3y)
    return jnp.sin(3 * x[..., 0]) + jnp.cos(3 * x[..., 1])


def dudx_nonpoly_dirichlet_data_3(x: jnp.array) -> jnp.array:
    # du/dx = 3cos(3x)
    return 3 * jnp.cos(3 * x[..., 0])


def dudy_nonpoly_dirichlet_data_3(x: jnp.array) -> jnp.array:
    # du/dy = -3sin(3y)
    return -3 * jnp.sin(3 * x[..., 1])


def source_nonpoly_dirichlet_data_3(x: jnp.array) -> jnp.array:
    # f(x, y) = -9(sin(3x) + cos(3y))
    return -9 * (jnp.sin(3 * x[..., 0]) + jnp.cos(3 * x[..., 1]))


TEST_CASE_NONPOLY_ITI = {
    "dirichlet_data_fn": nonpoly_dirichlet_data_3,
    "d_xx_coeff_fn": default_lap_coeffs,
    "d_yy_coeff_fn": default_lap_coeffs,
    "source_fn": source_nonpoly_dirichlet_data_3,
    "du_dx_fn": dudx_nonpoly_dirichlet_data_3,
    "du_dy_fn": dudy_nonpoly_dirichlet_data_3,
}

# If we set K1 higher than 5.0, then we need to increase the polynomial orders in
# single_leaf_check_11, single_leaf_check_12, single_merge_check_9, single_merge_check_10
# to start to see the h-p convergence.
K1: float = 2.0
# K2: float = 3.0
# ETA: float = jnp.pi
# K1: float = 1.0
ETA: float = jnp.pi


def helmholtz_dirichlet_data_0(x: jnp.array, k: float = K1) -> jnp.array:
    # u(x,y) = e^{ik(x+y)}
    return jnp.exp(1j * k * (x[..., 0] + x[..., 1]))


def dudx_helmholtz_dirichlet_data_0(x: jnp.array, k: float = K1) -> jnp.array:
    # du/dx = ike^{ik(x+y)}
    return 1j * k * jnp.exp(1j * k * (x[..., 0] + x[..., 1]))


def dudy_helmholtz_dirichlet_data_0(x: jnp.array, k: float = K1) -> jnp.array:
    # du/dy = ike^{ik(x+y)}
    return 1j * k * jnp.exp(1j * k * (x[..., 0] + x[..., 1]))


def helmholtz_I_coeff_fn(x: jnp.array, k: float = K1) -> jnp.array:
    return k**2 * 2 * jnp.ones_like(x[..., 0])


"""
Lu = f in Omega
u = g on the boundary of Omega

This test case is the following:
L = Delta + 2 * k_1^2
f = 0
g = e^{ik_1(x+y)}

The homogeneous solution is w(x, y) = e^{ik_1(x+y)}.
The particular solution is v(x,y) = 0.
The total solution is u(x,y) = e^{ik_1(x+y)}.
"""
TEST_CASE_HELMHOLTZ_0 = {
    "dirichlet_data_fn": helmholtz_dirichlet_data_0,
    "d_xx_coeff_fn": default_lap_coeffs,
    "d_yy_coeff_fn": default_lap_coeffs,
    "I_coeff_fn": helmholtz_I_coeff_fn,
    "du_dx_fn": dudx_helmholtz_dirichlet_data_0,
    "du_dy_fn": dudy_helmholtz_dirichlet_data_0,
    "source_fn": default_zero_source,
    "part_soln_fn": default_zero_source,
    "part_du_dx_fn": default_zero_source,
    "part_du_dy_fn": default_zero_source,
    "eta": ETA,
}


def helmholtz_case_1_particular_soln(
    x: jnp.array, k: float = K1, eta: float = ETA
) -> jnp.array:
    # Expect x to have shape [..., 2]
    # v(x,y) = exp{-i (x^2 + y^2)}
    return jnp.exp(-1j * (x[..., 0] ** 2 + x[..., 1] ** 2))


def helmholtz_case_1_part_du_dx_fn(
    x: jnp.array, k: float = K1, eta: float = ETA
) -> jnp.array:
    # Expect x to have shape [..., 2]
    # v(x,y) = exp{-i (x^2 + y^2)}
    # dvdx(x,y) = -i 2x exp{-i (x + y)}
    return -1j * helmholtz_case_1_particular_soln(x, k, eta) * 2 * x[..., 0]


def helmholtz_case_1_part_du_dy_fn(
    x: jnp.array, k: float = K1, eta: float = ETA
) -> jnp.array:
    # Expect x to have shape [..., 2]
    # v(x,y) = exp{-i (x^2 + y^2)}
    # dvdy(x,y) = -i 2y exp{-i (x + y)}
    return -1j * helmholtz_case_1_particular_soln(x, k, eta) * 2 * x[..., 1]


def helmholtz_case_1_source(x: jnp.array, k: float = K1, eta: float = ETA) -> jnp.array:
    # Expect x to have shape [..., 2]
    # f(x, y) = (- 4 x^2 -  4 y^2 - 4i + 2k^2) v(x,y)
    term1 = -4 * x[..., 0] ** 2
    term2 = -4 * x[..., 1] ** 2
    term3 = -4j
    term4 = 2 * k**2
    return (term1 + term2 + term3 + term4) * helmholtz_case_1_particular_soln(x, k, eta)


"""
Lu = f in Omega
u = g on the boundary of Omega

This test case is the following:
L = Delta + 2 * k_1^2
f = (- 4 x^2 -  4 y^2 - 4i + 2k^2) e^{-i (x^2 + y^2)}
g = e^{ik_1(x+y)}

The homogeneous solution is w(x, y) = e^{ik_1(x+y)}.
The particular solution is v(x,y) = e^{-i(x^2 + y^2)}.
The total solution is u(x,y) = e^{ik_1(x+y)}.
"""
TEST_CASE_HELMHOLTZ_1 = {
    # Test 2D.ItI.e in my notes
    "dirichlet_data_fn": helmholtz_dirichlet_data_0,
    "d_xx_coeff_fn": default_lap_coeffs,
    "d_yy_coeff_fn": default_lap_coeffs,
    "I_coeff_fn": helmholtz_I_coeff_fn,
    "du_dx_fn": dudx_helmholtz_dirichlet_data_0,
    "du_dy_fn": dudy_helmholtz_dirichlet_data_0,
    "part_soln_fn": helmholtz_case_1_particular_soln,
    "source_fn": helmholtz_case_1_source,
    "part_du_dx_fn": helmholtz_case_1_part_du_dx_fn,
    "part_du_dy_fn": helmholtz_case_1_part_du_dy_fn,
    "eta": ETA,
}


def nonpoly_dirichlet_data_2(x: jnp.array, n: float = 10.0):
    # f(r, theta) = r^n cos(n theta)
    r = jnp.linalg.norm(x, axis=-1)
    theta = jnp.arctan2(x[..., 1], x[..., 0])
    return jnp.power(r, n) * jnp.cos(n * theta)


TEST_CASE_POISSON_NONPOLY_2 = {
    K_DIRICHLET: nonpoly_dirichlet_data_2,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_SOURCE: default_zero_source,
}


K_1 = 10.0


def nonpoly_dirichlet_data_3(x: jnp.array, k: float = K_1):
    # u(x,y) = cos(kx) + sin(ky)
    return jnp.cos(k * x[..., 0]) + jnp.sin(k * x[..., 1])


def nonpoly_source_3(x: jnp.array, k: float = K_1):
    # f(x,y) = -k^2(cos(kx) + sin(ky))
    coeff = -(k**2)
    return coeff * (jnp.cos(k * x[..., 0]) + jnp.sin(k * x[..., 1]))


"""
Lu = f in the domain
u = g on the boundary
L = Delta
f = -k^2(cos(kx) + sin(ky))
g = cos(kx) + sin(ky)

Homogeneous solution is w(x,y) = cos(kx) + sin(ky)
Particular solution is v(x,y) = 0
Solution is u(x,y) = w(x,y) + v(x,y) = cos(kx) + sin(ky)
"""

TEST_CASE_POISSON_NONPOLY_3 = {
    K_DIRICHLET: nonpoly_dirichlet_data_3,
    K_SOURCE: nonpoly_source_3,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    "part_soln_fn": default_zero_source,
}


def adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    # x has shape [..., 2]
    # u(x,y) = \tan^{-1}\left( 50 \sqrt{(x + 0.05)^2 + (y + 0.05)^2 }- 0.7\right)
    y = x[..., 1]
    x = x[..., 0]
    return jnp.arctan(10 * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2) - 0.7))


def d_x_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    # x has shape [..., 2]
    y = x[..., 1]
    x = x[..., 0]
    out = (
        10
        * (x + 0.05)
        / (
            jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2)
            * (100 * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2) - 0.7) ** 2 + 1)
        )
    )
    return out


def d_y_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    # x has shape [..., 2]
    y = x[..., 1]
    x = x[..., 0]
    out = (
        10
        * (y + 0.05)
        / (
            jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2)
            * (100 * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2) - 0.7) ** 2 + 1)
        )
    )
    return out


def d_xx_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    # x has shape [..., 2]
    y = x[..., 0]
    x = x[..., 1]

    # This output is from sympy
    out = (
        10
        * (
            -200
            * (x + 0.05) ** 2
            * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2) - 0.7)
            / (
                ((x + 0.05) ** 2 + (y + 0.05) ** 2)
                * (100 * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2) - 0.7) ** 2 + 1)
            )
            - (x + 0.05) ** 2 / ((x + 0.05) ** 2 + (y + 0.05) ** 2) ** (3 / 2)
            + 1 / jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2)
        )
        / (100 * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2) - 0.7) ** 2 + 1)
    )

    return out


def d_yy_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    # x has shape [..., 2]
    y = x[..., 0]
    x = x[..., 1]

    # This output is from sympy
    out = (
        10
        * (
            -200
            * (y + 0.05) ** 2
            * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2) - 0.7)
            / (
                ((x + 0.05) ** 2 + (y + 0.05) ** 2)
                * (100 * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2) - 0.7) ** 2 + 1)
            )
            - (y + 0.05) ** 2 / ((x + 0.05) ** 2 + (y + 0.05) ** 2) ** (3 / 2)
            + 1 / jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2)
        )
        / (100 * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2) - 0.7) ** 2 + 1)
    )

    return out
