# src/wave_scattering/exterior_solver.py
# Contains the code to compute the solution
# to the scattering problem at exterior points

import jax.numpy as jnp
import jax
from typing import Tuple
import logging

from src.jaxhps import Domain, PDEProblem, DiscretizationNode2D, build_solver

from src.wave_scattering.ScatteringProblem import ScatteringProblem
from src.wave_scattering.gen_SD_exterior import (
    gen_D_exterior,
    gen_S_exterior,
)

from src.wave_scattering.scattering_utils import (
    get_DtN_from_ItI,
    get_uin_and_normals,
    get_uin,
)

def setup_scattering_lin_system(
    S: jnp.array,
    D: jnp.array,
    T_int: jnp.array,
    uin: jax.Array,
    uin_dn: jax.Array,
) -> Tuple[jnp.array, jnp.array]:
    """
    Sets up the BIE system in eqn (3.4) of the Gillman, Barnett, Martinsson paper.

    Args:
        S (jnp.array): Single-layer potential matrix. Has shape (n,n)
        D (jnp.array): Double-layer potential matrix. Has shape (n,n)
        T_int (jnp.array): Dirichlet-to-Neumann matrix. Has shape (n,n)
        gauss_bdry_pts (jnp.array): Has shape (n, 2)
        k (float): Frequency of the incoming plane waves
        source_directions (jnp.array): Has shape (n_sources,). Describes the direction of the incoming plane waves in radians.

    Returns:
        Tuple[jnp.array, jnp.array]: A, which has shape (n, n) and b, which has shape (n, n_sources).
    """
    n_bdry_pts = S.shape[0]

    A = 0.5 * jnp.eye(n_bdry_pts) - D + S @ T_int
    b = S @ (uin_dn - T_int @ uin)

    return A, b

def get_uscat_and_dn(
    S: jnp.array,
    D: jnp.array,
    T: jnp.array,
    uin: jax.Array,
    uin_dn: jax.Array,
) -> jnp.array:
    """Get uscat on the boundary and its normal derivatives
    For use computing uscat outside the computational domain
    """
    A, b = setup_scattering_lin_system(
        S=S, D=D, T_int=T, uin=uin, uin_dn=uin_dn
    )

    # logging.info(f"Solving the system... (A.shape={A.shape}; b.shape={b.shape})")
    # Solve the lin system to get uscat on the boundary
    uscat = jnp.linalg.solve(A, b)

    # Eqn 1.12 from the Gillman, Barnett, Martinsson paper
    uscat_dn = T @ (uscat + uin) - uin_dn

    return uscat, uscat_dn


def forward_model_exterior(
    scattering_problem: ScatteringProblem,
    q: jax.Array,
) -> jax.Array:
    """
    Implements the foward scattering model by finding uscat on the boundary of the computational domain and
    evaluating uscat at exterior sensor points.

    If scattering_problem does not have incident waves specified, this function will construct incident plane waves using
    scattering_problem.source_dirs and k=pde_problem.eta
    """
    # logging.info(f"Starting exterior solve...")
    pde_problem = scattering_problem.pde_problem

    # Set up the parts of the PDE
    # i_term = pde_problem.eta**2 * (jnp.ones_like(q) + q)
    i_term = pde_problem.eta**2 * (q+1)

    pde_problem.update_coefficients(I_coefficients=i_term)

    # Build the solver
    # logging.info(f"Building the solver...")
    T_ItI = build_solver(
        pde_problem=pde_problem,
        return_top_T=True,
        compute_device=jax.devices()[0],
        host_device=jax.devices()[0],
    )
    # logging.info(f"Computing DtN from ItI...")
    T_DtN = get_DtN_from_ItI(T_ItI, pde_problem.eta)

    # If the incident waves are not specified, use plane waves with freq k and
    # directions indicated by scattering_problem.source_dirs
    # logging.info(f"Computing uin and uin_dn on the boundary")
    if scattering_problem.uin_bdry is None:
        uin_bdry, uin_dn_bdry = get_uin_and_normals(
            k=pde_problem.eta,
            bdry_pts=pde_problem.domain.boundary_points,
            source_directions=scattering_problem.source_dirs,
        )
    else:
        uin_bdry = scattering_problem.uin_bdry
        uin_dn_bdry = scattering_problem.uin_dn_bdry

    # Scattered field and its outward normal derivative on the boundary
    # logging.info(f"Computing uscat values and normal derivatives on the boundary...")
    uscat_homog, uscat_dn_homog = get_uscat_and_dn(
        S=scattering_problem.S_int,
        D=scattering_problem.D_int,
        T=T_DtN,
        uin=uin_bdry,
        uin_dn=uin_dn_bdry,
    )

    # Now we need to compute the scattered field at the exterior points
    # logging.info(f"Mapping to exterior points...")
    out = (
        scattering_problem.D_ext @ uscat_homog
        - scattering_problem.S_ext @ uscat_dn_homog
    )
    # logging.info(f"Finished and returning!")
    return out
