import jax
import jax.numpy as jnp
from typing import List, Callable
from ._pdeproblem import PDEProblem, _get_PDEProblem_chunk
import logging
from ._device_config import (
    local_solve_chunksize_2D,
)
from .local_solve._uniform_2D_DtN import local_solve_stage_uniform_2D_DtN
from .local_solve._uniform_2D_ItI import local_solve_stage_uniform_2D_ItI


from .merge._uniform_2D_DtN import merge_stage_uniform_2D_DtN
from .merge._uniform_2D_ItI import merge_stage_uniform_2D_ItI

from .down_pass._uniform_2D_DtN import down_pass_uniform_2D_DtN
from .down_pass._uniform_2D_ItI import down_pass_uniform_2D_ItI


def solve_subtree(
    pde_problem: PDEProblem,
    boundary_data: jax.Array | List,
    subtree_height: int = 7,
    compute_device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> jax.Array:
    """
    This function solves the PDE using the novel subtree recomputation strategy.
    This algorithm is only supported for 2D problems with uniform quadtrees.
    The algorithm proceeds by splitting the problem into a set of subtrees, computing the outgoing data (T and h)
    for the root of the subtrees, and then performing the highest level merges. This allows us to
    greatly reduce the number of data movements between the CPU and GPU, at the cost of more floating point operations.

    Unlike the :func:`jaxhps.build_solver` method, this function does not save any of the solution matrices computed during the upward pass.
    Thus, it is most appropriate when we want to solve one instance of the problem very quickly.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    boundary_data : jax.Array | List
        Specifies the boundary data on the boundary discretization points. Can be a list or a jax.Array.

    subtree_height : int, optional
        Height of the subtrees used in our recomputation algorithm. The default is what we found to be optimal for DtN merges using fp64 on an NVIDIA H100 GPU.

    compute_device : jax.Device, optional
        Where the computation should be performed. This is typically a GPU device. Defaults to ``jax.devices()[0]``.

    host_device : jax.Device, optional
        Device where the returned data lives. This is typically a CPU device. Defaults to ``jax.devices("cpu")[0]``.

    Returns
    -------
    solns : jax.Array
        Solutions to the boundary value problem on the HPS grid.

    """
    if not pde_problem.domain.bool_2D:
        raise ValueError(
            "Subtree recomputation is only supported for 2D problems."
        )
    if not pde_problem.domain.bool_uniform:
        raise ValueError(
            "Subtree recomputation is only supported for uniform quadtrees."
        )

    if isinstance(boundary_data, list):
        # If the boundary data is a list, we need to concatenate it
        # into a single array.
        boundary_data = jnp.concatenate(boundary_data)

    if pde_problem.use_ItI:
        # Check whether we need to use recomputation.
        if (
            local_solve_chunksize_2D(pde_problem.domain.p, jnp.complex128)
            >= pde_problem.domain.n_leaves
        ):
            return _all_together_ItI(
                pde_problem,
                boundary_data,
                compute_device=compute_device,
                host_device=host_device,
            )
        else:
            # This is the branch which uses subtree recomputation.
            S_lst, g_tilde_lst = _local_solve_and_build(
                pde_problem=pde_problem,
                boundary_data=None,
                subtree_height=subtree_height,
                compute_device=compute_device,
                host_device=host_device,
                local_solve_fn=local_solve_stage_uniform_2D_ItI,
                merge_fn=merge_stage_uniform_2D_ItI,
                down_pass_fn=down_pass_uniform_2D_ItI,
            )
            # Perform a partial down pass
            bdry_data = down_pass_uniform_2D_ItI(
                boundary_data,
                S_lst,
                g_tilde_lst,
                device=compute_device,
                host_device=host_device,
            )

            # Final local solve + build + down pass
            return _local_solve_and_build(
                pde_problem=pde_problem,
                boundary_data=bdry_data,
                subtree_height=subtree_height,
                compute_device=compute_device,
                host_device=host_device,
                local_solve_fn=local_solve_stage_uniform_2D_ItI,
                merge_fn=merge_stage_uniform_2D_ItI,
                down_pass_fn=down_pass_uniform_2D_ItI,
            )

    else:
        # Check whether we need to use recomputation.
        if (
            local_solve_chunksize_2D(pde_problem.domain.p, jnp.float64)
            >= pde_problem.domain.n_leaves
        ):
            return _all_together_DtN(
                pde_problem,
                boundary_data,
                compute_device=compute_device,
                host_device=host_device,
            )

        else:
            # This is the branch which uses subtree recomputation.
            S_lst, g_tilde_lst = _local_solve_and_build(
                pde_problem=pde_problem,
                boundary_data=None,
                subtree_height=subtree_height,
                compute_device=compute_device,
                host_device=host_device,
                local_solve_fn=local_solve_stage_uniform_2D_DtN,
                merge_fn=merge_stage_uniform_2D_DtN,
                down_pass_fn=down_pass_uniform_2D_DtN,
            )
            # Perform a partial down pass
            bdry_data = down_pass_uniform_2D_DtN(
                boundary_data,
                S_lst,
                g_tilde_lst,
                device=compute_device,
                host_device=host_device,
            )

            # Final local solve + build + down pass
            return _local_solve_and_build(
                pde_problem=pde_problem,
                boundary_data=bdry_data,
                subtree_height=subtree_height,
                compute_device=compute_device,
                host_device=host_device,
                local_solve_fn=local_solve_stage_uniform_2D_DtN,
                merge_fn=merge_stage_uniform_2D_DtN,
                down_pass_fn=down_pass_uniform_2D_DtN,
            )


def _all_together_ItI(
    pde_problem: PDEProblem,
    boundary_data: jax.Array,
    compute_device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> jax.Array:
    # Perform the local solve stage on the whole problem
    Y, T, v, h = local_solve_stage_uniform_2D_ItI(
        pde_problem=pde_problem,
        device=compute_device,
        host_device=compute_device,
    )
    # Perform the merge stage
    S_arr, g_tilde_arr = merge_stage_uniform_2D_ItI(
        T,
        h,
        l=pde_problem.domain.L,
        device=compute_device,
        host_device=compute_device,
    )
    # Perform the down pass
    solns = down_pass_uniform_2D_ItI(
        boundary_data,
        S_arr,
        g_tilde_arr,
        Y,
        v,
        device=compute_device,
        host_device=host_device,
    )
    return solns


def _all_together_DtN(
    pde_problem: PDEProblem,
    boundary_data: jax.Array,
    compute_device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> jax.Array:
    # Perform the local solve stage on the whole problem
    Y, T, v, h = local_solve_stage_uniform_2D_DtN(
        pde_problem=pde_problem,
        device=compute_device,
        host_device=compute_device,
    )
    # Perform the merge stage
    S_arr, g_tilde_arr = merge_stage_uniform_2D_DtN(
        T,
        h,
        l=pde_problem.domain.L,
        device=compute_device,
        host_device=compute_device,
    )
    # Perform the down pass
    solns = down_pass_uniform_2D_DtN(
        boundary_data,
        S_arr,
        g_tilde_arr,
        Y,
        v,
        device=compute_device,
        host_device=host_device,
    )
    return solns


def _local_solve_and_build(
    pde_problem: PDEProblem,
    boundary_data: jax.Array | None,
    subtree_height: int,
    compute_device: jax.Device,
    host_device: jax.Device,
    local_solve_fn: Callable,
    merge_fn: Callable,
    down_pass_fn: Callable,
    return_top_T: bool = False,
) -> jax.Array:
    """
    Performs the meat of the upward and downward pass of the subtree recomputation algorithm.

    If boundary_data is specified, performs the downward pass. Otherwise, performs the upward
    pass.

    Args:
        pde_problem (PDEProblem): _description_
        boundary_data (jax.Array | None): _description_
        subtree_height (int): _description_
        compute_device (jax.Device): _description_
        host_device (jax.Device): _description_
        local_solve_fn (Callable): _description_
        merge_fn (Callable): _description_
        down_pass_fn (Callable): _description_
        return_top_T (bool, optional): _description_. Defaults to False.

    Returns:
        jax.Array: _description_
    """
    n_leaves = pde_problem.domain.n_leaves
    # Get the chunk size
    chunk_size = 4**subtree_height

    n_chunks = n_leaves // chunk_size

    # This function is used in the upward and downward passes of the algorithm.
    # If the boundary data is None, we are in the upward pass.
    upward_pass = boundary_data is None

    logging.debug(
        "_local_solve_and_build: n_leaves=%s, chunk_size=%s, n_chunks=%s, subtree_height=%s",
        n_leaves,
        chunk_size,
        n_chunks,
        subtree_height,
    )
    # If the boundary data is not None, we are in the downward pass.
    if not upward_pass:
        # # Do some downward passes to get the boundary data propagated to the
        # # roots of the subtrees
        # boundary_data = down_pass_fn(
        #     boundary_data,
        #     pde_problem.S_lst,
        #     pde_problem.g_tilde_lst,
        #     None,
        #     None,
        #     device=compute_device,
        #     host_device=compute_device,
        # )
        logging.debug(
            "_local_solve_and_build: boundary_data shape: %s",
            boundary_data.shape,
        )
        # logging.debug(
        #     "_local_solve_and_build: S_lst[-1].shape: %s", boundary_data.shape
        # )

        # Now that it's been reshaped, compute the chunksize
        bdry_data_chunksize = boundary_data.shape[0] // n_chunks

    # For storing the data at the top of the subtrees
    T_lst = []
    h_lst = []
    solns_lst = []

    # Iterate over the chunks
    for i, start_idx in enumerate(range(0, n_leaves, chunk_size)):
        end_idx = min(start_idx + chunk_size, n_leaves)
        pde_problem_i = _get_PDEProblem_chunk(
            pde_problem=pde_problem, start_idx=start_idx, end_idx=end_idx
        )

        # Perform the local solve stage on the chunk
        Y_arr_chunk, T_arr_chunk, v_chunk, h_chunk = local_solve_fn(
            pde_problem=pde_problem_i,
            device=compute_device,
            host_device=compute_device,
        )

        # Merge the stuff together
        merge_out = merge_fn(
            T_arr=T_arr_chunk,
            h_arr=h_chunk,
            l=subtree_height,
            device=compute_device,
            host_device=compute_device,
            subtree_recomp=upward_pass,
        )

        if upward_pass:
            T_last, h_last = merge_out

            Y_arr_chunk.delete()
            T_arr_chunk.delete()

            T_lst.append(T_last)
            h_lst.append(h_last)

        else:
            S_lst, g_tilde_lst = merge_out
            logging.debug(
                "_local_solve_and_build: S_lst[-1].shape = %s", S_lst[-1].shape
            )

            T_arr_chunk.delete()
            h_chunk.delete()

            bdry_data_i = boundary_data[
                i * bdry_data_chunksize : (i + 1) * bdry_data_chunksize
            ]
            bdry_data_i = jax.device_put(bdry_data_i, compute_device)
            # Perform the down pass
            solns = down_pass_fn(
                bdry_data_i,
                S_lst,
                g_tilde_lst,
                Y_arr_chunk,
                v_chunk,
                device=compute_device,
                host_device=compute_device,
            )

            # Store the solution
            solns_lst.append(solns)

    # At the end, we either concatenate the solutions and return them, or we concatenate the
    # T and h arrays and do the final upward merges
    if upward_pass:
        T_arr = jnp.concatenate(T_lst, axis=0)
        for T in T_lst:
            T.delete()
        h_arr = jnp.concatenate(h_lst, axis=0)
        for h in h_lst:
            h.delete()

        # Perform the final upward merges
        final_merge_out = merge_fn(
            T_arr=T_arr,
            h_arr=h_arr,
            l=pde_problem.domain.L - subtree_height,
            device=compute_device,
            host_device=compute_device,
            subtree_recomp=False,
            return_T=return_top_T,
        )

        return final_merge_out
    else:
        return jax.device_put(jnp.concatenate(solns_lst, axis=0), host_device)


def upward_pass_subtree(
    pde_problem: PDEProblem,
    subtree_height: int = 7,
    compute_device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> jax.Array:
    """
     Does the upward pass of the subtree recomputation algorithm, returns the top-level Poincare--Steklov matrix, and
     stores the high-level :math:`S` and :math:`\\tilde{g}` data. This is meant to be used in conjunction with
     :func:`jaxhps.downward_pass_subtree` for large problems where the boundary data must be specified after the upward pass,
     such as a wave scattering context, where the boundary impedance values can not be computed without the top-level ItI matrix.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    subtree_height : int, optional
        Height of the subtrees used in our recomputation algorithm. The default is what we found to be optimal for DtN merges using fp64 on an NVIDIA H100 GPU.

    compute_device : jax.Device, optional
        Where the computation should be performed. This is typically a GPU device. Defaults to ``jax.devices()[0]``.

    host_device : jax.Device, optional
        Device where the returned data lives. This is typically a CPU device. Defaults to ``jax.devices("cpu")[0]``.

    Returns
    -------
    T_last : jax.Array
        Top-level Poincare--Steklov matrix for the whole domain.

    """
    if not pde_problem.use_ItI:
        merge_out = _local_solve_and_build(
            pde_problem=pde_problem,
            boundary_data=None,
            subtree_height=subtree_height,
            compute_device=compute_device,
            host_device=host_device,
            local_solve_fn=local_solve_stage_uniform_2D_DtN,
            merge_fn=merge_stage_uniform_2D_DtN,
            down_pass_fn=down_pass_uniform_2D_DtN,
            return_top_T=True,
        )

    else:
        merge_out = _local_solve_and_build(
            pde_problem=pde_problem,
            boundary_data=None,
            subtree_height=subtree_height,
            compute_device=compute_device,
            host_device=host_device,
            local_solve_fn=local_solve_stage_uniform_2D_ItI,
            merge_fn=merge_stage_uniform_2D_ItI,
            down_pass_fn=down_pass_uniform_2D_ItI,
            return_top_T=True,
        )

    pde_problem.S_lst = merge_out[0]
    pde_problem.g_tilde_lst = merge_out[1]

    return merge_out[2]


def downward_pass_subtree(
    pde_problem: PDEProblem,
    boundary_data: jax.Array,
    subtree_height: int = 7,
    compute_device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> jax.Array:
    """
     Does the downward pass of the subtree recomputation algorithm. This is meant to be used in conjunction with
     func:`jaxhps.upward_pass_subtree` for large problems where the boundary data must be specified after the upward pass,
     such as a wave scattering context, where the boundary impedance values can not be computed without the top-level ItI matrix.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    boundary_data : jax.Array | List
        Specifies the boundary data on the boundary discretization points. Can be a list or a jax.Array.

    subtree_height : int, optional
        Height of the subtrees used in our recomputation algorithm. Must be the same as used in the upward pass.

    compute_device : jax.Device, optional
        Where the computation should be performed. This is typically a GPU device. Defaults to ``jax.devices()[0]``.

    host_device : jax.Device, optional
        Device where the returned data lives. This is typically a CPU device. Defaults to ``jax.devices("cpu")[0]``.

    Returns
    -------
    solns : jax.Array
        Solutions to the boundary value problem on the HPS grid. Has shape (n_leaves, p^2)

    """
    if isinstance(boundary_data, list):
        # If the boundary data is a list, we need to concatenate it
        # into a single array.
        boundary_data = jnp.concatenate(boundary_data)
    if not pde_problem.use_ItI:
        # Perform a partial down pass
        bdry_data = down_pass_uniform_2D_DtN(
            boundary_data,
            pde_problem.S_lst,
            pde_problem.g_tilde_lst,
            v_arr=None,
            Y_arr=None,
            device=compute_device,
            host_device=host_device,
        )

        return _local_solve_and_build(
            pde_problem=pde_problem,
            boundary_data=bdry_data,
            subtree_height=subtree_height,
            compute_device=compute_device,
            host_device=host_device,
            local_solve_fn=local_solve_stage_uniform_2D_DtN,
            merge_fn=merge_stage_uniform_2D_DtN,
            down_pass_fn=down_pass_uniform_2D_DtN,
        )

    else:
        # Perform a partial down pass
        bdry_data = down_pass_uniform_2D_ItI(
            boundary_data,
            pde_problem.S_lst,
            pde_problem.g_tilde_lst,
            v_arr=None,
            Y_arr=None,
            device=compute_device,
            host_device=host_device,
        )
        return _local_solve_and_build(
            pde_problem=pde_problem,
            boundary_data=bdry_data,
            subtree_height=subtree_height,
            compute_device=compute_device,
            host_device=host_device,
            local_solve_fn=local_solve_stage_uniform_2D_ItI,
            merge_fn=merge_stage_uniform_2D_ItI,
            down_pass_fn=down_pass_uniform_2D_ItI,
        )
