"""Utility functions for implementing the wave scattering solver presented in the
Gillman, Barnett, Martinsson paper."""

from typing import Tuple, Callable
import logging
from timeit import default_timer

import jax.numpy as jnp
import jax
import h5py


from hahps import (
    DiscretizationNode2D,
    Domain,
    PDEProblem,
    upward_pass_subtree,
    downward_pass_subtree,
    local_solve_chunksize_2D,
)
from hahps.local_solve import local_solve_stage_uniform_2D_ItI
from hahps.merge import merge_stage_uniform_2D_ItI
from hahps.down_pass import down_pass_uniform_2D_ItI


# Disable logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)


@jax.jit
def get_DtN_from_ItI(R: jnp.array, eta: float) -> jnp.array:
    """
    Given an ItI matrix, generates the corresponding DtN matrix.

    equation 2.17 from the Gillman, Barnett, Martinsson paper.

    Implements the formula: T = -i eta (R - I)^{-1}(R + I)

    Args:
        R (jnp.array): Has shape (n, n)
        eta (float): Real number; parameter used in the ItI map creation.

    Returns:
        jnp.array: Has shape (n, n)
    """
    n = R.shape[0]
    I = jnp.eye(n)
    T = -1j * eta * jnp.linalg.solve(R - I, R + I)
    return T


def load_SD_matrices(fp: str) -> Tuple[jnp.array, jnp.array]:
    """
    Load S and D matrices from a MATLAB v7.3 .mat file using h5py.

    Args:
        fp (str): File path to the .mat file

    Returns:
        Tuple[jnp.array, jnp.array]: A tuple containing (S, D) matrices

    Raises:
        ValueError: If the file doesn't contain required matrices
        FileNotFoundError: If the file doesn't exist
        RuntimeError: If there's an error reading the file
    """
    # try:

    # Open the file in read mode
    with h5py.File(fp, "r") as f:
        # Check if required matrices exist
        if "S" not in f or "D" not in f:
            raise ValueError("File must contain both 'S' and 'D' matrices")

        # Helper function to convert structured array to complex array
        def to_complex(structured_array):
            # Convert structured array to complex numpy array
            complex_array = (
                structured_array["real"] + 1j * structured_array["imag"]
            )
            # Convert to jax array
            return jnp.array(complex_array)

        # Load matrices and convert to complex numbers
        # Note: transpose the arrays as MATLAB stores them in column-major order
        S = to_complex(f["S"][:].T)
        D = to_complex(f["D"][:].T)

        return S, D

    # except:
    #     raise RuntimeError(f"Error reading MATLAB file: {fp}")


@jax.jit
def setup_scattering_lin_system(
    S: jnp.array,
    D: jnp.array,
    T_int: jnp.array,
    gauss_bdry_pts: jnp.array,
    k: float,
    source_directions: jnp.array,
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
    n_bdry_pts = gauss_bdry_pts.shape[0]

    # T_int = get_DtN_from_ItI(R, eta)

    uin, normals = get_uin_and_normals(k, gauss_bdry_pts, source_directions)

    A = 0.5 * jnp.eye(n_bdry_pts) - D + S @ T_int
    b = S @ (normals - T_int @ uin)

    return A, b


@jax.jit
def get_uin_and_normals(
    k: float, bdry_pts: jnp.array, source_directions: jnp.array
) -> Tuple[jnp.array, jnp.array]:
    """
    Given the boundary points and the source directions, computes the incoming wave and the normal vectors.

    uin(x) = exp(i k <x,s>), where s = (cos(theta), sin(theta)) is the direction of the incoming plane wave.

    d uin(x) / dx = ik s_0 uin(x)
    d uin(x) / dy = ik s_1 uin(x)

    Args:
        k (float): Frequency of the incoming plane waves
        bdry_pts (jnp.array): Has shape (n, 2)
        source_directions (jnp.array): Has shape (n_sources,). Describes the direction of the incoming plane waves in radians.

    Returns:
        Tuple[jnp.array, jnp.array]: uin, normals. uin has shape (n,). normals has shape (n, 2).
    """
    n_per_side = bdry_pts.shape[0] // 4

    uin = get_uin(k, bdry_pts, source_directions)
    # print("get_uin_and_normals: uin shape: ", uin.shape)
    source_vecs = jnp.array(
        [jnp.cos(source_directions), jnp.sin(source_directions)]
    ).T
    # print("get_uin_and_normals: source_vecs shape: ", source_vecs.shape)

    normals = jnp.concatenate(
        [
            -1j
            * k
            * jnp.expand_dims(source_vecs[:, 1], axis=0)
            * uin[:n_per_side],  # -1 duin/dy
            1j
            * k
            * jnp.expand_dims(source_vecs[:, 0], axis=0)
            * uin[n_per_side : 2 * n_per_side],  # duin/dx
            1j
            * k
            * jnp.expand_dims(source_vecs[:, 1], axis=0)
            * uin[2 * n_per_side : 3 * n_per_side],  # duin/dy
            -1j
            * k
            * jnp.expand_dims(source_vecs[:, 0], axis=0)
            * uin[3 * n_per_side :],
        ]
    )

    # Plot the normals
    # plt.plot(normals.real, "-o", label="real")
    # plt.plot(normals.imag, "-o", label="imag")
    # plt.legend()
    # plt.show()

    return uin, normals


@jax.jit
def get_uin(
    k: float, pts: jnp.array, source_directions: jnp.array
) -> jnp.array:
    source_vecs = jnp.array(
        [jnp.cos(source_directions), jnp.sin(source_directions)]
    ).T
    # jax.debug.print("get_uin: source_vecs: {x}", x=source_vecs)
    uin = jnp.exp(1j * k * jnp.dot(pts, source_vecs.T))
    return uin


@jax.jit
def get_scattering_uscat_impedance(
    S: jnp.array,
    D: jnp.array,
    T: jnp.array,
    source_dirs: jnp.array,
    bdry_pts: jnp.array,
    k: float,
    eta: float,
) -> jnp.array:
    A, b = setup_scattering_lin_system(
        S=S,
        D=D,
        T_int=T,
        gauss_bdry_pts=bdry_pts,
        k=k,
        source_directions=source_dirs,
    )
    uin, uin_dn = get_uin_and_normals(k, bdry_pts, source_dirs)

    # logging.debug(
    #     "get_scattering_uscat_impedance: A shape: %s, b shape: %s, uin shape: %s, uin_dn shape: %s",
    #     A.shape,
    #     b.shape,
    #     uin.shape,
    #     uin_dn.shape,
    # )

    # Solve the lin system to get uscat on the boundary
    uscat = jnp.linalg.solve(A, b)

    # Eqn 1.12 from the Gillman, Barnett, Martinsson paper
    uscat_dn = T @ (uscat + uin) - uin_dn

    # Assemble incoming impedance for the scattered field
    imp = uscat_dn + 1j * eta * uscat

    return imp


# Not optimized.
INTERP_BATCH_SIZE = 20


def solve_scattering_problem(
    l: int,
    p: int,
    n: int,
    k: float,
    q_fn: Callable[[jnp.array], jnp.array],
    domain_bounds: jnp.array,
    source_dirs: jnp.array,
    S: jax.Array,
    D: jax.Array,
) -> Tuple[jnp.array, jnp.array, float]:
    """
    Does the following:

    1. Sets up the HPS quadrature for l levels and polynomial order p.
    2. Evaluates the coefficients and source term for the wave scattering problem.
    3. Performs local solves and merges to get a top-level ItI operator.
    4. Uses top-level ItI operator along with loaded S and D matrices to solve the exterior problem.
    5. Propagates the resulting impedance data down to the leaves.
    6. Interpolates the solution onto a regular grid with n points per dimension.


    Args:
        l (int): _description_
        p (int): _description_
        n (int): _description_
        k (float): _description_
        q_fn (Callable[[jnp.array], jnp.array]): _description_
        domain_bounds (jnp.array): _description_
        source_dirs (jnp.array): _description_
        S (jax.Array): _description_
        D (jax.Array): _description_

    Returns:
        Tuple[jnp.array, jnp.array, float]: _description_
    """

    # These things are fast to precompute

    # Set up the HPS quadrature for l levels and polynomial order p
    logging.info("solve_scattering_problem: Creating tree...")
    xmin, xmax, ymin, ymax = domain_bounds
    root = DiscretizationNode2D(
        xmin=float(xmin), xmax=float(xmax), ymin=float(ymin), ymax=float(ymax)
    )

    domain = Domain(p=p, q=p - 2, root=root, L=l)

    # Evaluate the coefficients and source term for the wave scattering problem
    d_xx_coeffs = jnp.ones_like(domain.interior_points[:, :, 0])
    d_yy_coeffs = jnp.ones_like(domain.interior_points[:, :, 0])
    i_term = k**2 * (1 + q_fn(domain.interior_points))
    logging.debug("solve_scattering_problem: i_term shape: %s", i_term.shape)

    uin_evals = get_uin(k, domain.interior_points, source_dirs)

    source_term = (
        -1 * (k**2) * q_fn(domain.interior_points)[..., None] * uin_evals
    )
    logging.debug(
        "solve_scattering_problem: source_term shape: %s", source_term.shape
    )

    logging.debug("solve_scattering_problem: S device: %s", S.devices())

    t = PDEProblem(
        domain=domain,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        I_coefficients=i_term,
        source=source_term,
        use_ItI=True,
        eta=k,
    )
    t_0 = default_timer()

    # Determine whether we need to use fused functions or can fit everything on the
    n_leaves = domain.n_leaves
    chunksize = local_solve_chunksize_2D(p, jnp.complex128)

    bool_use_recomp = chunksize < n_leaves

    if bool_use_recomp:
        T_ItI = upward_pass_subtree(
            pde_problem=t,
            subtree_height=6,
            compute_device=jax.devices()[0],
            host_device=jax.devices()[0],
        )
    else:
        Y_arr, T_arr, v_arr, h_arr = local_solve_stage_uniform_2D_ItI(
            pde_problem=t,
            host_device=jax.devices()[0],
            device=jax.devices()[0],
        )
        S_arr_lst, g_tilde_lst, T_ItI = merge_stage_uniform_2D_ItI(
            T_arr=T_arr,
            h_arr=h_arr,
            l=domain.L,
            device=jax.devices()[0],
            host_device=jax.devices()[0],
            return_T=True,
        )

    T_DtN = get_DtN_from_ItI(T_ItI, t.eta)

    logging.info(
        "solve_scattering_problem: Solving boundary integral equation..."
    )

    if jax.devices()[0] not in S.devices():
        S = jax.device_put(S, jax.devices()[0])
        D = jax.device_put(D, jax.devices()[0])
        bool_delete_SD = True
    else:
        bool_delete_SD = False
    incoming_imp_data = get_scattering_uscat_impedance(
        S=S,
        D=D,
        T=T_DtN,
        source_dirs=source_dirs,
        bdry_pts=t.domain.boundary_points,
        k=k,
        eta=k,
    )

    # Delete exterior matrices we no longer need
    T_ItI.delete()
    T_DtN.delete()
    if bool_delete_SD:
        S.delete()
        D.delete()

    # Propagate the resulting impedance data down to the leaves
    if bool_use_recomp:
        uscat_soln = downward_pass_subtree(
            pde_problem=t,
            boundary_data=incoming_imp_data,
            subtree_height=6,
            compute_device=jax.devices()[0],
            host_device=jax.devices()[0],
        )
    else:
        uscat_soln = down_pass_uniform_2D_ItI(
            boundary_imp_data=incoming_imp_data,
            S_lst=S_arr_lst,
            g_tilde_lst=g_tilde_lst,
            Y_arr=Y_arr,
            v_arr=v_arr,
            device=jax.devices()[0],
            host_device=jax.devices()[0],
        )

    # Interpolate the solution onto a regular grid with n points per dimension

    logging.info(
        "solve_scattering_problem: Interpolating solution onto regular grid..."
    )
    xvals_reg = jnp.linspace(xmin, xmax, n)
    yvals_reg = jnp.linspace(ymin, ymax, n)

    n_src = source_dirs.shape[0]

    uscat_regular = jnp.zeros(
        (n, n, n_src), dtype=jnp.complex128, device=jax.devices("cpu")[0]
    )

    # Do the interpolation from HPS to regular grid in batches of size INTERP_BATCH_SIZE
    # along the source dimension
    for i in range(0, n_src, INTERP_BATCH_SIZE):
        chunk_start = i
        chunk_end = min((i + INTERP_BATCH_SIZE), n_src)
        logging.debug(
            "solve_scattering_problem: Interpolating chunk i=%s, %s:%s",
            i,
            chunk_start,
            chunk_end,
        )
        uscat_i = uscat_soln[..., chunk_start:chunk_end]
        logging.debug(
            "solve_scattering_problem: uscat_i.devices()=%s", uscat_i.devices()
        )
        chunk_i, target_pts = domain.interp_from_interior_points(
            samples=uscat_i, eval_points_x=xvals_reg, eval_points_y=yvals_reg
        )

        chunk_i = jax.device_put(chunk_i, jax.devices("cpu")[0])
        uscat_regular = uscat_regular.at[..., chunk_start:chunk_end].set(
            chunk_i
        )

    uscat_regular.block_until_ready()

    t_1 = default_timer() - t_0
    return uscat_regular, target_pts, t_1
