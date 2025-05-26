import logging
from typing import Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp


from ._discretization_tree import (
    DiscretizationNode3D,
    get_all_leaves,
)

from ._discretization_tree_operations_3D import add_eight_children
from ._grid_creation_3D import (
    compute_interior_Chebyshev_points_adaptive_3D,
    compute_interior_Chebyshev_points_uniform_3D,
    bounds_for_oct_subdivision,
    rearrange_indices_ext_int,
)
from .quadrature import (
    chebyshev_weights,
)
from ._precompute_operators_3D import precompute_L_8f1
from ._device_config import DEVICE_ARR, HOST_DEVICE


@jax.jit
def node_to_bounds(node: DiscretizationNode3D) -> jax.Array:
    return jnp.array(
        [node.xmin, node.xmax, node.ymin, node.ymax, node.zmin, node.zmax]
    )


def generate_adaptive_mesh_level_restriction(
    root: DiscretizationNode3D,
    f_fn: Callable[[jax.Array], jax.Array],
    tol: float,
    p: int,
    q: int,
    restrict_bool: bool = True,
    l2_norm: bool = False,
) -> None:
    L_8f1 = precompute_L_8f1(p)
    if l2_norm:
        # Get a rough estimate of the L2 norm of the function
        add_eight_children(root, root=root, q=q)
        pts = compute_interior_Chebyshev_points_adaptive_3D(root, p)
        f_evals = f_fn(pts)
        # Estimate the squared L2 norm of the function on each child
        patch_nrms = jnp.array(
            [
                get_squared_l2_norm_single_voxel(
                    f_evals[i], node_to_bounds(node), p
                )
                for i, node in enumerate(get_all_leaves(root))
            ]
        )
        # global_nrm holds an estimate of the squared L2 norm of the function over the entire domain
        global_nrm = jnp.sum(patch_nrms)

        # We are computing squared L2 norms all over the place so need to square the
        # tolerance to make sure the original || f||_2 < tol is satisfied
        tol = tol**2
    else:
        # Get a rough estimate of the L_infinity norm of the function.
        # This will be refined as we go.
        points_1 = compute_interior_Chebyshev_points_uniform_3D(
            root, L=1, p=p
        ).reshape(-1, 3)
        global_nrm = jnp.max(f_fn(points_1))

    if l2_norm:

        def check_single_node_l2(
            bounds: jax.Array, global_nrm: float
        ) -> Tuple[bool, float]:
            """
            Input has shape (2,3). Given the corners of the panel, check
            the L2 refinement criterion and update the
            estimate of the global l2 norm.

            External variables are:
            f_fn, L_8f1, tol, global_nrm, p
            """
            node = DiscretizationNode3D(
                xmin=bounds[0],
                xmax=bounds[1],
                ymin=bounds[2],
                ymax=bounds[3],
                zmin=bounds[4],
                zmax=bounds[5],
            )
            points_0 = compute_interior_Chebyshev_points_uniform_3D(
                node, L=0, p=p
            ).reshape(-1, 3)
            points_1 = compute_interior_Chebyshev_points_uniform_3D(
                node, L=1, p=p
            ).reshape(-1, 3)

            f_evals = f_fn(points_0)
            f_evals_refined = f_fn(points_1)
            f_interp = L_8f1 @ f_evals
            err = get_squared_l2_norm_eight_voxels(
                f_interp - f_evals_refined, bounds, p
            )
            return err / global_nrm < tol, 0.0

        vmapped_check_queue = jax.vmap(
            check_single_node_l2, in_axes=(0, None), out_axes=(0, 0)
        )

    else:

        def check_single_node_linf(
            bounds: jax.Array, global_nrm: float
        ) -> Tuple[bool, float]:
            """
            Input has shape (2,3). Given the corners of the panel, check
            the L_infinitiy refinement criterion and update the
            estimate of the global l infinity norm.
            """
            node = DiscretizationNode3D(
                xmin=bounds[0],
                xmax=bounds[1],
                ymin=bounds[2],
                ymax=bounds[3],
                zmin=bounds[4],
                zmax=bounds[5],
            )
            points_0 = compute_interior_Chebyshev_points_uniform_3D(
                node, L=0, p=p
            ).reshape(-1, 3)
            points_1 = compute_interior_Chebyshev_points_uniform_3D(
                node, L=1, p=p
            ).reshape(-1, 3)

            f_evals = f_fn(points_0)
            f_evals_refined = f_fn(points_1)
            return check_current_discretization_global_linf_norm(
                f_evals, f_evals_refined, L_8f1, tol, global_nrm
            )

        vmapped_check_queue = jax.vmap(
            check_single_node_linf, in_axes=(0, None), out_axes=(0, 0)
        )

    refinement_check_queue = list(get_all_leaves(root))
    refinement_check_bounds = jnp.array(
        [node_to_bounds(node) for node in refinement_check_queue]
    )

    # Loop through the queue and refine nodes as necessary.
    while len(refinement_check_queue):
        logging.debug(
            "generate_adaptive_mesh_level_restriction: Queue length: %i",
            len(refinement_check_queue),
        )
        refinement_check_bounds = jax.device_put(
            refinement_check_bounds, DEVICE_ARR[0]
        )
        global_nrm = jax.device_put(global_nrm, DEVICE_ARR[0])
        checks_bool, linf_nrms_arr = vmapped_check_queue(
            refinement_check_bounds, global_nrm
        )
        checks_bool = jax.device_put(checks_bool, HOST_DEVICE)

        if not l2_norm:
            # Update l_infinity norm
            global_nrm = jnp.max(linf_nrms_arr)

        # Loop through the nodes we just tested. If we have to refine a node, we will need to add its children
        # to a new queue.
        new_refinement_check_queue = []
        for i, node in enumerate(refinement_check_queue):
            if not checks_bool[i]:
                add_eight_children(node, root=root, q=q)
                new_refinement_check_queue.extend(node.children)

                if restrict_bool:
                    # If we are enforcing the level restriction criterion, we need to check whether
                    # the neighbors of the newly refined node need to be refined.
                    level_restriction_check_queue = [
                        node,
                    ]

                    # Loop through the nodes that need to be checked for the level restriction criterion.
                    # If we refine a node, we'll need to add it to the queue.
                    while len(level_restriction_check_queue):
                        # Pop one of the nodes off the queue
                        for_check = level_restriction_check_queue.pop()
                        # Compute the neighbors of the node
                        bounds = volumes_to_check(for_check, root)
                        for vol in bounds:
                            # logging.debug(
                            #     "generate_adaptive_mesh_level_restriction: Calling find_or_add_child"
                            # )
                            n = find_or_add_child(root, root, q, *vol)
                            # logging.debug(
                            #     "generate_adaptive_mesh_level_restriction: find_or_add_child returned"
                            # )

                            if n is not None:
                                level_restriction_check_queue.append(n)
                                new_refinement_check_queue.extend(n.children)

        # Update the queue and do it all again
        refinement_check_queue = new_refinement_check_queue
        refinement_check_bounds = jnp.array(
            [node_to_bounds(node) for node in refinement_check_queue]
        )


def volumes_to_check(
    newly_refined: DiscretizationNode3D, root: DiscretizationNode3D
) -> jax.Array:
    """
    Given a newly refined node, find the volume bounds that need to be checked for the level restriction criterion.
    For instance, if we refine a node with bounds
    [[xmin, ymin, zmin], [xmax, ymax, zmax]] == [[0, 0, 0], [0.5, 0.5, 0.5]],

    and the root bounds are
    [[xmin, ymin, zmin], [xmax, ymax, zmax]] == [[0, 0, 0], [1, 1, 1]],

    then the volumes to check are volumes with sidelength 0.5 that are adjacent to the newly refined node:
    [[xmin, ymin, zmin], [xmax, ymax, zmax]] == [[0.5, 0, 0], [1, 0.5, 0.5]],
    [[xmin, ymin, zmin], [xmax, ymax, zmax]] == [[0, 0.5, 0], [0.5, 1, 0.5]],
    [[xmin, ymin, zmin], [xmax, ymax, zmax]] == [[0, 0, 0.5], [0.5, 0.5, 1]],

    Args:
        newly_refined (Node): _description_
        root (Node): _description_

    Returns:
        jax.Array: Has shape [???, 6] and lists the 3D volume bounds that need to be checked. Second axis lists
        [xmin, xmax, ymin, ymax, zmin, zmax]
    """

    node_sidelen = newly_refined.xmax - newly_refined.xmin
    new_xmin = newly_refined.xmin - node_sidelen
    new_xmax = newly_refined.xmax + node_sidelen
    new_ymin = newly_refined.ymin - node_sidelen
    new_ymax = newly_refined.ymax + node_sidelen
    new_zmin = newly_refined.zmin - node_sidelen
    new_zmax = newly_refined.zmax + node_sidelen

    volumes = []
    if new_xmin >= root.xmin:
        volumes.append(
            [
                new_xmin,
                newly_refined.xmin,
                newly_refined.ymin,
                newly_refined.ymax,
                newly_refined.zmin,
                newly_refined.zmax,
            ]
        )
    if new_xmax <= root.xmax:
        volumes.append(
            [
                newly_refined.xmax,
                new_xmax,
                newly_refined.ymin,
                newly_refined.ymax,
                newly_refined.zmin,
                newly_refined.zmax,
            ]
        )
    if new_ymin >= root.ymin:
        volumes.append(
            [
                newly_refined.xmin,
                newly_refined.xmax,
                new_ymin,
                newly_refined.ymin,
                newly_refined.zmin,
                newly_refined.zmax,
            ]
        )
    if new_ymax <= root.ymax:
        volumes.append(
            [
                newly_refined.xmin,
                newly_refined.xmax,
                newly_refined.ymax,
                new_ymax,
                newly_refined.zmin,
                newly_refined.zmax,
            ]
        )
    if new_zmin >= root.zmin:
        volumes.append(
            [
                newly_refined.xmin,
                newly_refined.xmax,
                newly_refined.ymin,
                newly_refined.ymax,
                new_zmin,
                newly_refined.zmin,
            ]
        )
    if new_zmax <= root.zmax:
        volumes.append(
            [
                newly_refined.xmin,
                newly_refined.xmax,
                newly_refined.ymin,
                newly_refined.ymax,
                newly_refined.zmax,
                new_zmax,
            ]
        )

    return jnp.array(volumes)


def find_or_add_child(
    node: DiscretizationNode3D,
    root: DiscretizationNode3D,
    q: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> DiscretizationNode3D | None:
    """
    Given a node and volume bounds, find the node with those bounds. If it doesn't exist, add it.

    Args:
        node (Node): The node to search from
        xmin (float): _description_
        xmax (float): _description_
        ymin (float): _description_
        ymax (float): _description_
        zmin (float): _description_
        zmax (float): _description_

    Returns:
        Node | None: If a node needs to be created, it is returned. Otherwise, None is returned.
    """

    node_sidelen = node.xmax - node.xmin
    requested_sidelen = xmax - xmin

    # if requested_sidelen * 2 == node_sidelen, we can make sure
    # the node has children and early exit
    if jnp.allclose(requested_sidelen * 2, node_sidelen):
        if len(node.children) == 0:
            # logging.debug(
            #     "find_or_add_child: Refining a node due to level restriction: %s", node
            # )
            add_eight_children(node, root=root, q=q)
            return node
        else:
            return None

    node_xmid = (node.xmax + node.xmin) / 2
    node_ymid = (node.ymax + node.ymin) / 2
    node_zmid = (node.zmax + node.zmin) / 2

    # Find which part of the octree to look in
    if xmin < node_xmid:
        # It could be in children (a,d,e,h) which are idxes (0,3,4,7)
        if ymin < node_ymid:
            # It could be in children (a,e) which are idxes (0,4)
            if zmin < node_zmid:
                # Child e
                child = node.children[4]
            else:
                # Child a
                child = node.children[0]
        else:
            # It could be in children (d,h) which are idxes (3,7)
            if zmin < node_zmid:
                # Child h
                child = node.children[7]
            else:
                # Child d
                child = node.children[3]
    else:
        # It could be children (b,c,f,g) which are idxes (1,2,5,6)
        if ymin < node_ymid:
            # It could be in children (b,f) which are idxes (1,5)
            if zmin < node_zmid:
                # Child f
                child = node.children[5]
            else:
                # Child b
                child = node.children[1]
        else:
            # It could be in children (c,g) which are idxes (2,6)
            if zmin < node_zmid:
                # Child g
                child = node.children[6]
            else:
                # Child c
                child = node.children[2]

    # If node_sidelen == 4 * requested_sidelen, then a grandchild of
    # the current node (aka a child of the child) needs to exist
    if jnp.allclose(node_sidelen, 4 * requested_sidelen):
        if len(child.children) == 0:
            # logging.debug(
            #     "find_or_add_child: Refining a node due to level restriction: %s", child
            # )

            add_eight_children(child, root=root, q=q)
            return child
        else:
            return None

    elif node_sidelen < 2 * requested_sidelen:
        raise ValueError("Requested volume is too large for the current node")

    else:
        return find_or_add_child(
            child, root, q, xmin, xmax, ymin, ymax, zmin, zmax
        )


@partial(jax.jit, static_argnums=(2,))
def get_squared_l2_norm_eight_voxels(
    f_evals: jax.Array, bounds: jax.Array, p: int
) -> float:
    # Split the corners into eight children
    bounds_lst = bounds_for_oct_subdivision(bounds)
    n_per_voxel = p**3

    # Call get_squared_l2_norm_single_voxel on each child
    out_lst = []
    for i, b in enumerate(bounds_lst):
        f_i = f_evals[i * n_per_voxel : (i + 1) * n_per_voxel]
        out_lst.append(get_squared_l2_norm_single_voxel(f_i, b, p))

    return jnp.sum(jnp.array(out_lst))


@partial(jax.jit, static_argnums=(2,))
def get_squared_l2_norm_single_voxel(
    f_evals: jax.Array, bounds: jax.Array, p: int
) -> float:
    """
    Given the function evaluations at the Chebyshev points of a single voxel, compute the squared L2 norm of the function
    over the voxel.



    Args:
        f_evals (jax.Array): Has shape [p**3]
        bounds (jax.Array): Has shape 6 and lists the bounds of the voxel [xmin, xmax, ymin, ymax, zmin, zmax]
        p (int): Chebyshev parameter.

    Returns:
        float: Squared L2 norm.
    """

    # Get the quadrature weights
    cheby_weights_x = chebyshev_weights(p, bounds[:2])
    cheby_weights_y = chebyshev_weights(p, bounds[2:4])
    cheby_weights_z = chebyshev_weights(p, bounds[4:])

    # Outer product of all three weight vectors then flatten it then
    # rearrange it
    cheby_weights = jnp.outer(cheby_weights_x, cheby_weights_y).flatten()
    cheby_weights = jnp.outer(cheby_weights, cheby_weights_z).flatten()
    r = rearrange_indices_ext_int(p)
    cheby_weights = cheby_weights[r]

    # Compute the squared L2 norm
    return jnp.sum(cheby_weights * f_evals**2)


@jax.jit
def check_current_discretization_global_linf_norm(
    f_evals: jnp.ndarray,
    f_evals_refined: jnp.ndarray,
    refinement_op: jnp.ndarray,
    tol: float,
    global_linf_norm: float,
) -> Tuple[bool, float]:
    """Takes the f_evals, interpolates them to the refined grid, and computes the
    L_inf norm of the difference.

    Args:
        f_evals (jnp.ndarray): Has shape (n,)
        f_evals_refined (jnp.ndarray): Has shape (4n,) or in the 3D case (8n,)
        refinement_op (jnp.ndarray): Has shape (4n, n) or in the 3D case (8n, n)
        tol (float): Tolerance for the check
        global_linf_norm (float): The global L_inf norm of the function

    Returns:
        Tuple[bool, float]: True if the current discretization is good, False otherwise,
        and max(global_linf_norm, L_inf(f_evals_refined))
    """

    f_interp = refinement_op @ f_evals
    ref_max = jnp.max(jnp.abs(f_evals_refined))
    diff_nrm = jnp.max(jnp.abs(f_interp - f_evals_refined))
    nrm_const = jnp.max(jnp.array([global_linf_norm, ref_max]))
    err = diff_nrm / nrm_const
    return err < tol, nrm_const
