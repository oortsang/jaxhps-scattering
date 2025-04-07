import logging
import jax
import jax.numpy as jnp

from ._pdeproblem import PDEProblem, _get_PDEProblem_chunk

from ._device_config import (
    HOST_DEVICE,
    DEVICE_ARR,
    local_solve_chunksize_2D,
    local_solve_chunksize_3D,
)
from ._discretization_tree import DiscretizationNode2D


from .local_solve._adaptive_2D_DtN import local_solve_stage_adaptive_2D_DtN
from .local_solve._adaptive_3D_DtN import local_solve_stage_adaptive_3D_DtN
from .local_solve._uniform_2D_DtN import local_solve_stage_uniform_2D_DtN
from .local_solve._uniform_2D_ItI import local_solve_stage_uniform_2D_ItI
from .local_solve._uniform_3D_DtN import local_solve_stage_uniform_3D_DtN
from .local_solve._nosource_uniform_2D_ItI import (
    nosource_local_solve_stage_uniform_2D_ItI,
)


from .merge._adaptive_2D_DtN import merge_stage_adaptive_2D_DtN
from .merge._adaptive_3D_DtN import merge_stage_adaptive_3D_DtN
from .merge._uniform_2D_DtN import merge_stage_uniform_2D_DtN
from .merge._uniform_2D_ItI import merge_stage_uniform_2D_ItI
from .merge._uniform_3D_DtN import merge_stage_uniform_3D_DtN
from .merge._nosource_uniform_2D_ItI import nosource_merge_stage_uniform_2D_ItI

from ._discretization_tree import get_all_leaves


def build_solver(
    pde_problem: PDEProblem,
    return_top_T: bool = False,
    compute_device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> None | jax.Array:
    """
    This function builds all of the matrices for the fast direct solver. This comprises of
    performing a local solve stage on each leaf, and merging information from the leaves to
    the root of the domain.

    This function performs the computation on compute_device and then transfers the data to
    host_device.

    The function will save the solution operators in the PDEProblem object.

    The function can optionally return the top-level Poincare--Steklov operator T. This is
    useful for problems, such as wave scattering, where one wants to couple the solver in
    the computational domain with a boundary integral equation defined on the domain's boundary.

    To compute solutions of the PDE, one must call the :func:`hahps.solve` after this one.

    Args:
        pde_problem (PDEProblem): Specifies the differential operator, source, domain, and precomputed interpolation and differentiation matrices.

        return_top_T (bool, optional): If set to True, the function will return the computed top-level Poincare--Steklov matrix. Defaults to False.

        compute_device (jax.Device, optional): Where the computation should happen. Defaults to jax.devices()[0].

        host_device (jax.Device, optional): Where the solution operators should be stored. Defaults to jax.devices("cpu")[0].

    Returns:
        None | jax.Array: If return_top_T is set to True, the function will return the computed top-level Poincare--Steklov matrix. Otherwise, it returns None.
    """

    # Special code path for 2D ItI problems, for which we are implementing
    # Upward and Downward passes.
    if pde_problem.source is None:
        if (
            not pde_problem.domain.bool_uniform
            or not isinstance(pde_problem.domain.root, DiscretizationNode2D)
            or not pde_problem.use_ItI
        ):
            raise ValueError(
                "Build stage for problems without source terms is only implemented for 2D uniform ItI problems."
            )

        else:
            return _nosource_build_solver(
                pde_problem=pde_problem,
                return_top_T=return_top_T,
                compute_device=compute_device,
                host_device=host_device,
            )

    # If it's a uniform problem, use this function. Otherwise,
    # use _adaptive_build_solver.
    if not pde_problem.domain.bool_uniform:
        return _adaptive_build_solver(
            pde_problem=pde_problem,
            return_top_T=return_top_T,
            compute_device=compute_device,
            host_device=host_device,
        )
    if pde_problem.use_ItI:
        local_solve_fn = local_solve_stage_uniform_2D_ItI
        merge_fn = merge_stage_uniform_2D_ItI
        chunksize_fn = local_solve_chunksize_2D

    elif pde_problem.domain.bool_uniform and pde_problem.domain.bool_2D:
        local_solve_fn = local_solve_stage_uniform_2D_DtN
        merge_fn = merge_stage_uniform_2D_DtN
        chunksize_fn = local_solve_chunksize_2D
    elif pde_problem.domain.bool_uniform and not pde_problem.domain.bool_2D:
        local_solve_fn = local_solve_stage_uniform_3D_DtN
        merge_fn = merge_stage_uniform_3D_DtN
        chunksize_fn = local_solve_chunksize_3D

    # Determine if batching is necessary.
    chunksize = chunksize_fn(pde_problem.domain.p, pde_problem.source.dtype)
    if chunksize < pde_problem.domain.n_leaves:
        # Do the local solve stage in batches.
        Y_arr_lst = []
        T_arr_lst = []
        v_lst = []
        h_lst = []

        for start_idx in range(0, pde_problem.domain.n_leaves, chunksize):
            end_idx = min(start_idx + chunksize, pde_problem.domain.n_leaves)

            # Get a chunk of the PDEProblem
            chunk_i = _get_PDEProblem_chunk(
                pde_problem=pde_problem, start_idx=start_idx, end_idx=end_idx
            )
            logging.debug(
                "build_solver: chunk_i.source.shape = %s", chunk_i.source.shape
            )
            # Perform the local solve stage on the chunk
            Y_arr_chunk, T_arr_chunk, v_chunk, h_chunk = local_solve_fn(
                pde_problem=chunk_i,
                device=compute_device,
                host_device=host_device,
            )
            Y_arr_lst.append(Y_arr_chunk)
            T_arr_lst.append(T_arr_chunk)
            v_lst.append(v_chunk)
            h_lst.append(h_chunk)

        # Concatenate the results from all chunks
        Y_arr_host = jnp.concatenate(Y_arr_lst, axis=0)
        T_arr_host = jnp.concatenate(T_arr_lst, axis=0)
        v_host = jnp.concatenate(v_lst, axis=0)
        h_host = jnp.concatenate(h_lst, axis=0)
    else:
        # Perform the local solve stage all at once for smaller problem sizes
        Y_arr_host, T_arr_host, v_host, h_host = local_solve_fn(
            pde_problem=pde_problem,
            device=compute_device,
            host_device=host_device,
        )
    pde_problem.Y = Y_arr_host
    pde_problem.v = v_host

    # Perform the merge stage
    merge_out = merge_fn(
        T_arr_host,
        h_host,
        l=pde_problem.domain.L,
        device=compute_device,
        host_device=host_device,
        return_T=return_top_T,
    )
    pde_problem.S_lst = merge_out[0]
    pde_problem.g_tilde_lst = merge_out[1]

    if return_top_T:
        return merge_out[2]


def _adaptive_build_solver(
    pde_problem: PDEProblem,
    return_top_T: bool = False,
    compute_device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> None:
    if pde_problem.domain.bool_2D:
        Y_arr_host, T_arr_host, v_host, h_host = (
            local_solve_stage_adaptive_2D_DtN(
                pde_problem=pde_problem,
                device=compute_device,
                host_device=host_device,
            )
        )
        pde_problem.Y = Y_arr_host
        pde_problem.v = v_host

        # Need to set all Y, T, v, h attributes in the leaves
        leaves = get_all_leaves(pde_problem.domain.root)
        for i, leaf in enumerate(leaves):
            leaf.data.Y = Y_arr_host[i]
            leaf.data.T = T_arr_host[i]
            leaf.data.v = v_host[i]
            leaf.data.h = h_host[i]

        merge_stage_adaptive_2D_DtN(
            pde_problem=pde_problem,
        )

    else:
        Y_arr_host, T_arr_host, v_host, h_host = (
            local_solve_stage_adaptive_3D_DtN(
                pde_problem=pde_problem,
                device=compute_device,
                host_device=host_device,
            )
        )
        pde_problem.Y = Y_arr_host
        pde_problem.v = v_host

        # Need to set all Y, T, v, h attributes in the leaves
        leaves = get_all_leaves(pde_problem.domain.root)
        for i, leaf in enumerate(leaves):
            leaf.data.Y = Y_arr_host[i]
            leaf.data.T = T_arr_host[i]
            leaf.data.v = v_host[i]
            leaf.data.h = h_host[i]

        merge_stage_adaptive_3D_DtN(
            pde_problem=pde_problem,
            T_arr=T_arr_host,
            h_arr=h_host,
            device=compute_device,
            host_device=host_device,
        )

    if return_top_T:
        return pde_problem.domain.root.data.T


def _nosource_build_solver(
    pde_problem: PDEProblem,
    return_top_T: bool = False,
    compute_device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> None | jax.Array:
    # Determine if batching is necessary.
    chunksize = local_solve_chunksize_2D(pde_problem.domain.p, jnp.complex128)
    if chunksize < pde_problem.domain.n_leaves:
        # Do the local solve stage in batches.
        Y_arr_lst = []
        T_arr_lst = []

        for start_idx in range(0, pde_problem.domain.n_leaves, chunksize):
            end_idx = min(start_idx + chunksize, pde_problem.domain.n_leaves)

            # Get a chunk of the PDEProblem
            chunk_i = _get_PDEProblem_chunk(
                pde_problem=pde_problem, start_idx=start_idx, end_idx=end_idx
            )
            logging.debug(
                "build_solver: chunk_i.source.shape = %s", chunk_i.source.shape
            )
            # Perform the local solve stage on the chunk
            Y_arr_chunk, T_arr_chunk = (
                nosource_local_solve_stage_uniform_2D_ItI(
                    pde_problem=chunk_i,
                    device=compute_device,
                    host_device=host_device,
                )
            )
            Y_arr_lst.append(Y_arr_chunk)
            T_arr_lst.append(T_arr_chunk)

        # Concatenate the results from all chunks
        Y_arr_host = jnp.concatenate(Y_arr_lst, axis=0)
        T_arr_host = jnp.concatenate(T_arr_lst, axis=0)
        # v_host = jnp.concatenate(v_lst, axis=0)
        # h_host = jnp.concatenate(h_lst, axis=0)
    else:
        # Perform the local solve stage all at once for smaller problem sizes
        Y_arr_host, T_arr_host = nosource_local_solve_stage_uniform_2D_ItI(
            pde_problem=pde_problem,
            device=compute_device,
            host_device=host_device,
        )
    pde_problem.Y = Y_arr_host
    # pde_problem.v = v_host

    # Perform the merge stage
    merge_out = nosource_merge_stage_uniform_2D_ItI(
        T_arr_host,
        l=pde_problem.domain.L,
        device=compute_device,
        host_device=host_device,
        return_T=return_top_T,
    )
    pde_problem.S_lst = merge_out[0]
    pde_problem.D_inv_lst = merge_out[1]
    pde_problem.BD_inv_lst = merge_out[2]
    if return_top_T:
        return merge_out[3]
    else:
        return None
