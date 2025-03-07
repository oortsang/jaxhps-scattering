import jax.numpy as jnp
import jax

K_DIRICHLET = "dirichlet_data_fn"
K_XX_COEFF = "d_xx_coeff_fn"
K_YY_COEFF = "d_yy_coeff_fn"
K_ZZ_COEFF = "d_zz_coeff_fn"
K_SOURCE = "source_fn"
K_DU_DX = "du_dx_fn"
K_DU_DY = "du_dy_fn"
K_DU_DZ = "du_dz_fn"
K_PART_SOLN = "part_soln_fn"
K_PART_SOLN_DUDX = "d_dx_part_soln_fn"
K_PART_SOLN_DUDY = "d_dy_part_soln_fn"
K_PART_SOLN_DUDZ = "d_dz_part_soln_fn"


def default_lap_coeffs(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.ones_like(x[..., 0])


def default_zero_source(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x[..., 0])


def poisson_poly_dirichlet_data(x: jnp.ndarray) -> jnp.ndarray:
    # f(x,y,z) = x**2 + y**2 - 2z**2
    return (x[..., 0] ** 2) + (x[..., 1] ** 2) - 2 * (x[..., 2] ** 2)


def poisson_poly_du_dx(x: jnp.ndarray) -> jnp.ndarray:
    # df/dx = 2x
    return 2 * x[..., 0]


def poisson_poly_du_dy(x: jnp.ndarray) -> jnp.ndarray:
    # df/dy = 2y
    return 2 * x[..., 1]


def poisson_poly_du_dz(x: jnp.ndarray) -> jnp.ndarray:
    # df/dz = -4z
    return -4 * x[..., 2]


TEST_CASE_POISSON_POLY = {
    K_DIRICHLET: poisson_poly_dirichlet_data,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_ZZ_COEFF: default_lap_coeffs,
    K_SOURCE: default_zero_source,
    K_DU_DX: poisson_poly_du_dx,
    K_DU_DY: poisson_poly_du_dy,
    K_DU_DZ: poisson_poly_du_dz,
}


def poisson_nonpoly_dirichlet_data(x: jnp.ndarray) -> jnp.ndarray:
    # f(x,y,z) = exp(x) * sin(sqrt{2} y) * exp(z)
    return jnp.exp(x[..., 0]) * jnp.sin(jnp.sqrt(2) * x[..., 1]) * jnp.exp(x[..., 2])


def poission_nonpoly_du_dx(x: jnp.ndarray) -> jnp.ndarray:
    # df/dx = exp(x) * sin(sqrt{2} y) * exp(z)
    return jnp.exp(x[..., 0]) * jnp.sin(jnp.sqrt(2) * x[..., 1]) * jnp.exp(x[..., 2])


def poissson_nonpoly_du_dy(x: jnp.ndarray) -> jnp.ndarray:
    # df/dy = sqrt{2} exp(x) * cos(sqrt{2} y) * exp(z)
    return (
        jnp.sqrt(2)
        * jnp.exp(x[..., 0])
        * jnp.cos(jnp.sqrt(2) * x[..., 1])
        * jnp.exp(x[..., 2])
    )


def poisson_nonpoly_du_dz(x: jnp.ndarray) -> jnp.ndarray:
    # df/dz = exp(x) * sin(sqrt{2} y) * exp(z)
    return jnp.exp(x[..., 0]) * jnp.sin(jnp.sqrt(2) * x[..., 1]) * jnp.exp(x[..., 2])


TEST_CASE_POISSON_NONPOLY = {
    K_DIRICHLET: poisson_nonpoly_dirichlet_data,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_ZZ_COEFF: default_lap_coeffs,
    K_SOURCE: default_zero_source,
    K_DU_DX: poission_nonpoly_du_dx,
    K_DU_DY: poissson_nonpoly_du_dy,
    K_DU_DZ: poisson_nonpoly_du_dz,
}


def poisson_nonzero_source_part_soln(x: jnp.ndarray) -> jnp.ndarray:
    # v(x,y,z) = y * (x^2 - pi/2^2)(y^2 - pi/2^2)(z^2 - pi/2^2)
    return (
        x[..., 1]
        * (x[..., 0] ** 2 - (jnp.pi / 2) ** 2)
        * (x[..., 1] ** 2 - (jnp.pi / 2) ** 2)
        * (x[..., 2] ** 2 - (jnp.pi / 2) ** 2)
    )


def poisson_nonzero_source_source(x: jnp.ndarray) -> jnp.ndarray:
    # f(x,y,z) = 2y(y^2 - pi/2^2)(z^2 - pi/2^2)
    #           + 6y(x^2 - pi/2^2)(z^2 - pi/2^2)
    #           + 2y(x^2 - pi/2^2)(y^2 - pi/2^2)
    return (
        2
        * x[..., 1]
        * (x[..., 1] ** 2 - (jnp.pi / 2) ** 2)
        * (x[..., 2] ** 2 - (jnp.pi / 2) ** 2)
        + 6
        * x[..., 1]
        * (x[..., 0] ** 2 - (jnp.pi / 2) ** 2)
        * (x[..., 2] ** 2 - (jnp.pi / 2) ** 2)
        + 2
        * x[..., 1]
        * (x[..., 0] ** 2 - (jnp.pi / 2) ** 2)
        * (x[..., 1] ** 2 - (jnp.pi / 2) ** 2)
    )


def poisson_nonzero_source_part_soln_dudx(x: jnp.ndarray) -> jnp.ndarray:
    # dv/dx = 2xy(y^2 - pi/2^2)(z^2 - pi/2^2)
    return (
        2
        * x[..., 0]
        * x[..., 1]
        * (x[..., 1] ** 2 - (jnp.pi / 2) ** 2)
        * (x[..., 2] ** 2 - (jnp.pi / 2) ** 2)
    )


def poisson_nonzero_source_part_soln_dudy(x: jnp.ndarray) -> jnp.ndarray:
    # dv/dy = (x^2 - pi/2^2)(3y^2 - pi/2^2)(z^2 - pi/2^2)
    return (
        (x[..., 0] ** 2 - (jnp.pi / 2) ** 2)
        * (3 * (x[..., 1] ** 2) - (jnp.pi / 2) ** 2)
        * (x[..., 2] ** 2 - (jnp.pi / 2) ** 2)
    )


def poisson_nonzero_source_part_soln_dudz(x: jnp.ndarray) -> jnp.ndarray:
    # dv/dz = 2yz(x^2 - pi/2^2)(y^2 - pi/2^2)
    return (
        2
        * x[..., 1]
        * x[..., 2]
        * (x[..., 0] ** 2 - (jnp.pi / 2) ** 2)
        * (x[..., 1] ** 2 - (jnp.pi / 2) ** 2)
    )


TEST_CASE_HOMOG_PART_POLY = {
    K_DIRICHLET: poisson_poly_dirichlet_data,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_ZZ_COEFF: default_lap_coeffs,
    K_SOURCE: poisson_nonzero_source_source,
    K_PART_SOLN: poisson_nonzero_source_part_soln,
    K_PART_SOLN_DUDX: poisson_nonzero_source_part_soln_dudx,
    K_PART_SOLN_DUDY: poisson_nonzero_source_part_soln_dudy,
    K_PART_SOLN_DUDZ: poisson_nonzero_source_part_soln_dudz,
}


def poisson_nonpoly_dirichlet_data_2(x: jnp.array, k: float = 10.0) -> jnp.array:
    # f(x,y,z) = cos(kx) + sin(ky) + sin(kz)
    term_1 = jnp.cos(k * x[..., 0])
    term_2 = jnp.sin(k * x[..., 1])
    term_3 = jnp.sin(k * x[..., 2])
    return term_1 + term_2 + term_3


def poisson_nonpoly_source_2(x: jnp.array, k: float = 10.0) -> jnp.array:
    # f(x,y,z) = -k^2(cos(kx) + sin(ky) + sin(kz))
    term_1 = jnp.cos(k * x[..., 0])
    term_2 = jnp.sin(k * x[..., 1])
    term_3 = jnp.sin(k * x[..., 2])
    coeff = -1 * (k**2)
    return coeff * (term_1 + term_2 + term_3)


TEST_CASE_POISSON_NONPOLY_2 = {
    K_DIRICHLET: poisson_nonpoly_dirichlet_data_2,
    K_SOURCE: poisson_nonpoly_source_2,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_ZZ_COEFF: default_lap_coeffs,
}


@jax.jit
def adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    # x has shape (..., 3)
    # u(x,y,z) = = \tan^{-1}\left( 50 \sqrt{(x + 0.05)^2 + (y + 0.05)^2 + (z + 0.05)^2 }- 0.7\right)
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    return jnp.arctan(
        10 * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7)
    )


@jax.jit
def d_x_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    out = (
        10
        * (x + 0.05)
        / (
            (
                100
                * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7)
                ** 2
                + 1
            )
            * jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
        )
    )
    return out


@jax.jit
def d_y_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    out = (
        10
        * (y + 0.05)
        / (
            (
                100
                * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7)
                ** 2
                + 1
            )
            * jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
        )
    )
    return out


@jax.jit
def d_z_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    out = (
        10
        * (z + 0.05)
        / (
            (
                100
                * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7)
                ** 2
                + 1
            )
            * jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
        )
    )
    return out


@jax.jit
def d_xx_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    out = (
        10
        * (
            -((x + 0.05) ** 2)
            / ((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) ** (3 / 2)
            - 200
            * (x + 0.05) ** 2
            * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7)
            / (
                (
                    100
                    * (
                        jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
                        - 0.7
                    )
                    ** 2
                    + 1
                )
                * ((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
            )
            + 1 / jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
        )
        / (
            100
            * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7) ** 2
            + 1
        )
    )
    return out


@jax.jit
def d_yy_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    out = (
        10
        * (
            -((y + 0.05) ** 2)
            / ((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) ** (3 / 2)
            - 200
            * (y + 0.05) ** 2
            * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7)
            / (
                (
                    100
                    * (
                        jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
                        - 0.7
                    )
                    ** 2
                    + 1
                )
                * ((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
            )
            + 1 / jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
        )
        / (
            100
            * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7) ** 2
            + 1
        )
    )
    return out


@jax.jit
def d_zz_adaptive_meshing_data_fn(x: jnp.array) -> jnp.array:
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    out = (
        10
        * (
            -((z + 0.05) ** 2)
            / ((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) ** (3 / 2)
            - 200
            * (z + 0.05) ** 2
            * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7)
            / (
                (
                    100
                    * (
                        jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
                        - 0.7
                    )
                    ** 2
                    + 1
                )
                * ((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
            )
            + 1 / jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
        )
        / (
            100
            * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7) ** 2
            + 1
        )
    )
    return out
