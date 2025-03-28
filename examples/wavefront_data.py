import jax
import jax.numpy as jnp


@jax.jit
def default_lap_coeffs(x: jax.Array) -> jax.Array:
    return jnp.ones_like(x[..., 0])


@jax.jit
def wavefront_soln(x: jax.Array) -> jax.Array:
    # x has shape (..., 3)
    # u(x,y,z) = = \tan^{-1}\left( 50 \sqrt{(x + 0.05)^2 + (y + 0.05)^2 + (z + 0.05)^2 }- 0.7\right)
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    return jnp.arctan(
        10
        * (jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2) - 0.7)
    )


@jax.jit
def d_x_wavefront_soln(x: jax.Array) -> jax.Array:
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    out = (
        10
        * (x + 0.05)
        / (
            (
                100
                * (
                    jnp.sqrt(
                        (x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2
                    )
                    - 0.7
                )
                ** 2
                + 1
            )
            * jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
        )
    )
    return out


@jax.jit
def d_y_wavefront_soln(x: jax.Array) -> jax.Array:
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    out = (
        10
        * (y + 0.05)
        / (
            (
                100
                * (
                    jnp.sqrt(
                        (x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2
                    )
                    - 0.7
                )
                ** 2
                + 1
            )
            * jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
        )
    )
    return out


@jax.jit
def d_z_wavefront_soln(x: jax.Array) -> jax.Array:
    y = x[..., 1]
    z = x[..., 2]
    x = x[..., 0]
    out = (
        10
        * (z + 0.05)
        / (
            (
                100
                * (
                    jnp.sqrt(
                        (x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2
                    )
                    - 0.7
                )
                ** 2
                + 1
            )
            * jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
        )
    )
    return out


@jax.jit
def d_xx_wavefront_soln(x: jax.Array) -> jax.Array:
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
            * (
                jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
                - 0.7
            )
            / (
                (
                    100
                    * (
                        jnp.sqrt(
                            (x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2
                        )
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
            * (
                jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
                - 0.7
            )
            ** 2
            + 1
        )
    )
    return out


@jax.jit
def d_yy_wavefront_soln(x: jax.Array) -> jax.Array:
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
            * (
                jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
                - 0.7
            )
            / (
                (
                    100
                    * (
                        jnp.sqrt(
                            (x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2
                        )
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
            * (
                jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
                - 0.7
            )
            ** 2
            + 1
        )
    )
    return out


@jax.jit
def d_zz_wavefront_soln(x: jax.Array) -> jax.Array:
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
            * (
                jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
                - 0.7
            )
            / (
                (
                    100
                    * (
                        jnp.sqrt(
                            (x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2
                        )
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
            * (
                jnp.sqrt((x + 0.05) ** 2 + (y + 0.05) ** 2 + (z + 0.05) ** 2)
                - 0.7
            )
            ** 2
            + 1
        )
    )
    return out
