import jax.numpy as jnp
import jax
from hps.src.solver_obj import create_solver_obj_2D

# from hps.src.up_down_passes import local_solve_stage, build_stage, down_pass
# from hps.src.methods.fused_methods import _fused_local_solve_and_build_2D_ItI, _down_pass_from_fused_ItI
from hps.src.methods.local_solve_stage import _local_solve_stage_2D_ItI
from hps.src.methods.uniform_build_stage import _uniform_build_stage_2D_ItI
from hps.src.methods.uniform_down_pass import _uniform_down_pass_2D_ItI
from hps.src.wave_scattering_utils import (
    get_uin,
    load_SD_matrices,
    get_scattering_uscat_impedance,
    get_DtN_from_ItI,
)
from hps.src.config import DEVICE_ARR
from hps.src.quadrature.trees import Node

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
SD_MATRIX_FP = f"data/wave_scattering/SD_matrices/SD_k{K}_n{P-2}_nside{2**L}_dom1.mat"
S, D = load_SD_matrices(SD_MATRIX_FP)

S = jax.device_put(S, DEVICE_ARR[0])
D = jax.device_put(D, DEVICE_ARR[0])
SAMPLE_ROOT = Node(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
SAMPLE_TREE = create_solver_obj_2D(
    p=P,
    q=P - 2,
    root=SAMPLE_ROOT,
    uniform_levels=L,
    use_ItI=True,
    eta=K,
)
SAMPLE_TREE.leaf_cheby_points = jax.device_put(
    SAMPLE_TREE.leaf_cheby_points, DEVICE_ARR[0]
)
AMPLITUDE = 0.5
MIN_TOL = 1e-15

# find the nearest x-coordinate in SAMPLE_TREE.leaf_cheby_points to 0.875
nearest_x_coord_idx = jnp.argmin(
    jnp.abs(SAMPLE_TREE.leaf_cheby_points[..., 0].flatten() - 0.875)
)
nearest_x_coord = SAMPLE_TREE.leaf_cheby_points[..., 0].flatten()[nearest_x_coord_idx]

observation_bools_x = jnp.logical_and(
    jnp.abs(SAMPLE_TREE.leaf_cheby_points[..., 0]) == nearest_x_coord,
    jnp.abs(SAMPLE_TREE.leaf_cheby_points[..., 1]) <= nearest_x_coord,
)
observation_bools_y = jnp.logical_and(
    jnp.abs(SAMPLE_TREE.leaf_cheby_points[..., 1]) == nearest_x_coord,
    jnp.abs(SAMPLE_TREE.leaf_cheby_points[..., 0]) <= nearest_x_coord,
)
OBSERVATION_BOOLS = jnp.logical_or(observation_bools_x, observation_bools_y)


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


def source_locations_to_scattered_field(source_locations: jnp.array) -> jnp.array:

    # corners = jnp.array([[XMIN, YMIN], [XMAX, YMIN], [XMAX, YMAX], [XMIN, YMAX]])
    # root = Node(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
    # tree = create_solver_obj_2D(
    #     p=P, q=P - 2, root=root, uniform_levels=L, use_ItI=True, eta=K
    # )
    SAMPLE_TREE.reset()
    tree = SAMPLE_TREE
    # Put the leaf Chebyshev points on the GPU so that
    # q evals and d_xx_coeffs are initialized on the same device
    tree.leaf_cheby_points = jax.device_put(tree.leaf_cheby_points, DEVICE_ARR[0])
    source_locations = source_locations.reshape(-1, 2)
    q_evals = q_point_sources(tree.leaf_cheby_points, source_locations)

    uin_evals = get_uin(K, tree.leaf_cheby_points, SOURCE_DIRS)[:, :, 0]

    d_xx_coeffs = jnp.ones_like(tree.leaf_cheby_points[:, :, 0])
    d_yy_coeffs = jnp.ones_like(tree.leaf_cheby_points[:, :, 0])
    i_term = K**2 * (jnp.ones_like(q_evals) + q_evals)

    source_term = source_term = -1 * (K**2) * q_evals * uin_evals

    # Compute the local solve and build stages
    # local_solve_stage(
    #     tree,
    #     source_term=source_term,
    #     D_xx_coeffs=d_xx_coeffs,
    #     D_yy_coeffs=d_yy_coeffs,
    #     I_coeffs=i_term,
    #     device=DEVICE_ARR[0],
    #     host_device=DEVICE_ARR[0],
    # )
    # build_stage(
    #     tree,
    #     device=DEVICE_ARR[0],
    #     host_device=DEVICE_ARR[0],
    # )
    T_arr, Y_arr, h_arr, v_arr = _local_solve_stage_2D_ItI(
        D_xx=tree.D_xx,
        D_xy=tree.D_xy,
        D_yy=tree.D_yy,
        D_x=tree.D_x,
        D_y=tree.D_y,
        I_P_0=tree.I_P_0,
        Q_I=tree.Q_I,
        F=tree.F,
        G=tree.G,
        p=tree.p,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
        I_coeffs=i_term,
        source_term=source_term,
        host_device=DEVICE_ARR[0],
    )
    S_arr_lst, f_arr_lst, R = _uniform_build_stage_2D_ItI(
        R_maps=T_arr, h_arr=h_arr, l=tree.l, host_device=DEVICE_ARR[0], return_ItI=True
    )
    # R is the top-level ItI matrix

    T = get_DtN_from_ItI(R, tree.eta)

    incoming_imp_data = get_scattering_uscat_impedance(
        S=S,
        D=D,
        T=T,
        source_dirs=SOURCE_DIRS,
        bdry_pts=tree.root_boundary_points,
        k=K,
        eta=K,
    )
    # n_per_side = T.shape[0] // 4
    # # Break incoming_imp_data into 4 sides
    # incoming_imp_data = [
    #     incoming_imp_data[i * n_per_side : (i + 1) * n_per_side] for i in range(4)
    # ]

    # Propagate the resulting impedance data down to the leaves
    # interior_solns = _down_pass_from_fused_ItI(
    #     bdry_data=incoming_imp_data,
    #     S_arr_lst=S_arr_lst,
    #     f_lst=f_arr_lst,
    #     D_xx=tree.D_xx,
    #     D_xy=tree.D_xy,
    #     D_yy=tree.D_yy,
    #     D_x=tree.D_x,
    #     D_y=tree.D_y,
    #     I_P_0=tree.I_P_0,
    #     Q_I=tree.Q_I,
    #     F=tree.F,
    #     G=tree.G,
    #     p=tree.p,
    #     l=tree.l,
    #     source_term=source_term,
    #     D_xx_coeffs=d_xx_coeffs,
    #     D_yy_coeffs=d_yy_coeffs,
    #     I_coeffs=i_term,
    # )
    interior_solns = _uniform_down_pass_2D_ItI(
        boundary_imp_data=incoming_imp_data,
        S_maps_lst=S_arr_lst,
        f_lst=f_arr_lst,
        leaf_Y_maps=Y_arr,
        v_array=v_arr,
        host_device=DEVICE_ARR[0],
    )

    uscat_evals = interior_solns

    return uscat_evals, tree.leaf_cheby_points


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
