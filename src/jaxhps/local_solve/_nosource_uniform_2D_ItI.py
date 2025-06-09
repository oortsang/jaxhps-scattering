import jax.numpy as jnp
import jax

from .._pdeproblem import PDEProblem
from ._uniform_2D_DtN import _gather_coeffs_2D, vmapped_assemble_diff_operator
from typing import Tuple


def nosource_local_solve_stage_uniform_2D_ItI(
    pde_problem: PDEProblem,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    This function performs the local solve stage for 2D problems with a uniform quadtree, creating ItI matrices.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.
    device : jax.Device
        Where to perform the computation. Defaults to ``jax.devices()[0]``.
    host_device : jax.Device
        Where to place the output. Defaults to ``jax.devices("cpu")[0]``.

    Returns
    -------
    Y : jax.Array
        Solution operators mapping from Impedance boundary data to homogeneous solutions on the leaf interiors. Has shape (n_leaves, p^2, 4q)
    T : jax.Array
        Impedance-to-Impedance matrices for each leaf. Has shape (n_leaves, 4q, 4q)
    Phi : jax.Array
        Particular solution operators mapping from the source term evaluated on the interior Cheby nodes to the particular solution on all of the Cheby nodes. Has shape (n_leaves, p^2, (p-1)^2).
    """

    # Gather the coefficients into a single array.
    coeffs_gathered, which_coeffs = _gather_coeffs_2D(
        D_xx_coeffs=pde_problem.D_xx_coefficients,
        D_xy_coeffs=pde_problem.D_xy_coefficients,
        D_yy_coeffs=pde_problem.D_yy_coefficients,
        D_x_coeffs=pde_problem.D_x_coefficients,
        D_y_coeffs=pde_problem.D_y_coefficients,
        I_coeffs=pde_problem.I_coefficients,
    )
    # stack the precomputed differential operators into a single array
    diff_ops = jnp.stack(
        [
            pde_problem.D_xx,
            pde_problem.D_xy,
            pde_problem.D_yy,
            pde_problem.D_x,
            pde_problem.D_y,
            jnp.eye(pde_problem.D_xx.shape[0], dtype=jnp.float64),
        ]
    )

    coeffs_gathered = jax.device_put(
        coeffs_gathered,
        device,
    )

    all_diff_operators = vmapped_assemble_diff_operator(
        coeffs_gathered, which_coeffs, diff_ops
    )

    R_arr, Y_arr, Phi_arr = vmapped_get_ItI_nosource(
        all_diff_operators,
        pde_problem.P,
        pde_problem.QH,
        pde_problem.G,
    )

    R_arr_host = jax.device_put(R_arr, host_device)
    Y_arr_host = jax.device_put(Y_arr, host_device)
    Phi_arr_host = jax.device_put(Phi_arr, host_device)

    return Y_arr_host, R_arr_host, Phi_arr_host


@jax.jit
def get_ItI_nosource(
    diff_operator: jax.Array,
    P: jax.Array,
    QH: jax.Array,
    G: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Given the coefficients specifying a partial differential operator on a leaf, this function
    computes the particular solution, particular solution boundary fluxes, the
    impedance to impedance map, and the impedance to solution map.

    Args:
        coeffs_arr (jax.Array): Has shape (?, p**2). Specifies the PDE coefficients.
        source_term (jax.Array): Has shape (p**2, n_sources). Specifies the RHS of the PDE.
        diff_ops (jax.Array): Has shape (5, p**2, p**2). Contains the precomputed differential operators. In 3D,
                                this has shape (9, p**3, p**3).
        which_coeffs (jax.Array): Has shape (5,) and specifies which coefficients are not None.
        P (jax.Array): Has shape (4(p-1), 4q). Maps data on the Gauss boundary nodes to data on the Cheby boundary nodes.
                            Is formed by taking the kronecker product of I and P_0, which is the standard
                            Gauss -> Cheby 1D interp matrix missing the last row.
        QH (jax.Array): Has shape (4q, p**2). Maps a function on the Chebyshev nodes to the function's outgoing impedance
                    on the boundary Gauss nodes.
        G (jax.Array): Has shape (4(p-1), p**2). Maps a function on the Chebyshev nodes to the function's incoming
                    impedance on the boundary Cheby nodes, counting corners once.

    Returns:
        Tuple[jax.Array]:
            T (jax.Array): Has shape (4q, 4q). This is the "ItI" operator, which maps incoming impedance data on the
                boundary Gauss nodes to the outgoing impedance data on the boundary Gauss nodes.
            Y (jax.Array): Has shape (p**2, 4q). This is the interior solution operator, which maps from incoming impedance
                data on the boundary Gauss nodes to the resulting homogeneous solution on the Chebyshev nodes.
            Phi (jax.Array): Has shape (p**2, (p-1)**2) and maps from the source term evaluated on the interior Cheby nodes to the particular solution on all of the Cheby nodes.

    """
    # print("get_ItI: I_P_0 shape: ", I_P_0.shape)
    # print("get_ItI: Q_I shape: ", Q_I.shape)
    n_cheby_pts = diff_operator.shape[-1]
    n_cheby_bdry_pts = P.shape[0]
    A = diff_operator

    # B has shape (n_cheby_pts, n_cheby_pts). Its top rows are F and its bottom rows are the
    # bottom rows of A.
    B = jnp.zeros((n_cheby_pts, n_cheby_pts), dtype=jnp.complex128)
    B = B.at[:n_cheby_bdry_pts].set(G)
    B = B.at[n_cheby_bdry_pts:].set(A[n_cheby_bdry_pts:])
    B_inv = jnp.linalg.inv(B)

    # Y has shape (n_cheby_pts, n_cheby_bdry_pts). It maps from
    # incoming impedance data on the boundary G-L nodes to the
    # homogeneous solution on all of the Cheby nodes.
    Y = B_inv[:, :n_cheby_bdry_pts] @ P

    # Phi has shape (n_cheby_pts, n_cheby_interior_pts). It maps from the source
    # term evaluated on the interior Cheby nodes to the particular soln on all of
    # the Cheby nodes.
    Phi = B_inv[:, n_cheby_bdry_pts:]

    # Interpolate to Gauss nodes
    T = QH @ Y
    return (T, Y, Phi)


vmapped_get_ItI_nosource = jax.vmap(
    get_ItI_nosource,
    in_axes=(0, None, None, None),
    out_axes=(0, 0, 0),
)
