import jax.numpy as jnp
import jax

from wave_scattering_utils import (
    get_uin,
    load_SD_matrices,
    get_scattering_uscat_impedance,
    get_DtN_from_ItI,
)

from hahps import (
    DiscretizationNode2D,
    Domain,
    PDEProblem,
)
from hahps.local_solve import local_solve_stage_uniform_2D_ItI
from hahps.merge import merge_stage_uniform_2D_ItI
from hahps.down_pass import down_pass_uniform_2D_ItI


L = 3
K = 20
P = 16
# XMIN = -jnp.pi
# XMAX = jnp.pi
# YMIN = -jnp.pi
# YMAX = jnp.pi
XMIN = -1
XMAX = 1
YMIN = -1
YMAX = 1

SIGMA = 0.15
SOURCE_DIRS = jnp.array([0.0])
SD_MATRIX_FP = (
    f"data/wave_scattering/SD_matrices/SD_k{K}_n{P - 2}_nside{2**L}_dom1.mat"
)
S, D = load_SD_matrices(SD_MATRIX_FP)

S = jax.device_put(S, jax.devices()[0])
D = jax.device_put(D, jax.devices()[0])
SAMPLE_ROOT = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
SAMPLE_DOMAIN = Domain(p=P, q=P - 2, root=SAMPLE_ROOT, L=L)
SAMPLE_DOMAIN.interior_points = jax.device_put(
    SAMPLE_DOMAIN.interior_points, jax.devices()[0]
)
AMPLITUDE = 0.5
MIN_TOL = 1e-15

# find the nearest x-coordinate in SAMPLE_DOMAIN.interior_points to 0.875
nearest_x_coord_idx = jnp.argmin(
    jnp.abs(SAMPLE_DOMAIN.interior_points[..., 0].flatten() - 0.875)
)
nearest_x_coord = SAMPLE_DOMAIN.interior_points[..., 0].flatten()[
    nearest_x_coord_idx
]

observation_bools_x = jnp.logical_and(
    jnp.abs(SAMPLE_DOMAIN.interior_points[..., 0]) == nearest_x_coord,
    jnp.abs(SAMPLE_DOMAIN.interior_points[..., 1]) <= nearest_x_coord,
)
observation_bools_y = jnp.logical_and(
    jnp.abs(SAMPLE_DOMAIN.interior_points[..., 1]) == nearest_x_coord,
    jnp.abs(SAMPLE_DOMAIN.interior_points[..., 0]) <= nearest_x_coord,
)
OBSERVATION_BOOLS = jnp.logical_or(observation_bools_x, observation_bools_y)


# Set up the PDEProblem object
uin_evals = get_uin(K, SAMPLE_DOMAIN.interior_points, SOURCE_DIRS)[:, :, 0]

d_xx_coeffs = jnp.ones_like(SAMPLE_DOMAIN.interior_points[:, :, 0])
d_yy_coeffs = jnp.ones_like(SAMPLE_DOMAIN.interior_points[:, :, 0])
SAMPLE_PDEPROBLEM = PDEProblem(
    domain=SAMPLE_DOMAIN,
    D_xx_coefficients=d_xx_coeffs,
    D_yy_coefficients=d_yy_coeffs,
    source=d_xx_coeffs,  # Dummy for now
    use_ItI=True,
    eta=K,
)


@jax.jit
def q_point_sources(x: jnp.array, source_locations: jnp.array) -> jnp.array:
    """
    Source locations gives a list of point source locations z_i

    This function returns the function q(x) = sum_i exp{ -|| x - z_i ||^2 / sigma^2}

    Args:
        x (jnp.array): Evaluation points. Has shape (..., 2)
        source_locations (jnp.array): Has shape (N_s,2). Will be reshaped to (N, 2)

    Returns:
        jnp.array: Has shape (...)
    """
    *batch_dims, coord_dim = x.shape
    source_locations = jnp.reshape(source_locations, (-1, 2))
    n = source_locations.shape[0]

    x_expanded = jnp.expand_dims(x, axis=-2)  # shape (..., 1, 2)
    broadcast_shape = [1] * len(batch_dims) + [n, 2]
    source_locations_expanded = jnp.reshape(
        source_locations, broadcast_shape
    )  # shape (..., n, 2)

    diff = x_expanded - source_locations_expanded  # shape (..., n, 2)
    radii = jnp.linalg.norm(diff, axis=-1) ** 2  # shape (..., n)
    return AMPLITUDE * jnp.sum(jnp.exp(-radii / SIGMA**2), axis=-1)


def source_locations_to_scattered_field(
    source_locations: jnp.array,
) -> jnp.array:
    # corners = jnp.array([[XMIN, YMIN], [XMAX, YMIN], [XMAX, YMAX], [XMIN, YMAX]])
    # root = Node(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
    # tree = create_solver_obj_2D(
    #     p=P, q=P - 2, root=root, uniform_levels=L, use_ItI=True, eta=K
    # )
    # Put the leaf Chebyshev points on the GPU so that
    # q evals and d_xx_coeffs are initialized on the same device
    source_locations = source_locations.reshape(-1, 2)
    q_evals = q_point_sources(SAMPLE_DOMAIN.interior_points, source_locations)

    i_term = K**2 * (jnp.ones_like(q_evals) + q_evals)

    source_term = source_term = -1 * (K**2) * q_evals * uin_evals

    SAMPLE_PDEPROBLEM.update_coefficients(
        source=source_term, I_coefficients=i_term
    )

    # Compute the local solve and build stages
    # local_solve_stage(
    #     tree,
    #     source_term=source_term,
    #     D_xx_coeffs=d_xx_coeffs,
    #     D_yy_coeffs=d_yy_coeffs,
    #     I_coeffs=i_term,
    #     device=jax.devices()[0],
    #     host_device=jax.devices()[0],
    # )
    # build_stage(
    #     tree,
    #     device=jax.devices()[0],
    #     host_device=jax.devices()[0],
    # )
    Y_arr, T_arr, v_arr, h_arr = local_solve_stage_uniform_2D_ItI(
        pde_problem=SAMPLE_PDEPROBLEM,
        host_device=jax.devices()[0],
        device=jax.devices()[0],
    )
    S_arr_lst, g_tilde_lst, T_ItI = merge_stage_uniform_2D_ItI(
        T_arr=T_arr,
        h_arr=h_arr,
        l=SAMPLE_DOMAIN.L,
        device=jax.devices()[0],
        host_device=jax.devices()[0],
        return_T=True,
    )

    T_DtN = get_DtN_from_ItI(T_ItI, SAMPLE_PDEPROBLEM.eta)

    incoming_imp_data = get_scattering_uscat_impedance(
        S=S,
        D=D,
        T=T_DtN,
        source_dirs=SOURCE_DIRS,
        bdry_pts=SAMPLE_DOMAIN.boundary_points,
        k=K,
        eta=K,
    )

    # Propagate the resulting impedance data down to the leaves
    interior_solns = down_pass_uniform_2D_ItI(
        boundary_imp_data=incoming_imp_data,
        S_lst=S_arr_lst,
        g_tilde_lst=g_tilde_lst,
        Y_arr=Y_arr,
        v_arr=v_arr,
    )

    uscat_evals = interior_solns

    return uscat_evals, SAMPLE_DOMAIN.interior_points


def forward_model(source_locations: jnp.array) -> jnp.array:
    """
    Args:
        source_locations (jnp.array): Has shape (N_s,) which we should interpret as a list of point source locations z_i
    """
    # cheby_pts = chebyshev_points(P - 2)[0]
    # x_pts = jnp.linspace(-1.0, 1.0, 10)
    # interp = barycentric_lagrange_interpolation_matrix(cheby_pts, x_pts)

    uscat, _ = source_locations_to_scattered_field(source_locations)

    # Take a subset of the HPS grid which is a box with radius
    uscat_obs = uscat[OBSERVATION_BOOLS].flatten()
    return uscat_obs
    # return uscat.flatten()
