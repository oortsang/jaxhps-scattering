import logging
from typing import Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp


from ._discretization_tree import (
    DiscretizationNode2D,
    get_all_leaves,
)

from ._discretization_tree_operations_2D import add_four_children
from ._grid_creation_2D import (
    compute_interior_Chebyshev_points_adaptive_2D,
    compute_interior_Chebyshev_points_uniform_2D,
    bounds_for_quad_subdivision,
    rearrange_indices_ext_int,
)
from .quadrature import (
    chebyshev_weights,
)
from ._precompute_operators_2D import precompute_L_4f1
from ._device_config import DEVICE_ARR, HOST_DEVICE
from ._adaptive_discretization_3D import (
    check_current_discretization_global_linf_norm,
)


@jax.jit
def node_to_bounds(node: DiscretizationNode2D) -> jax.Array:
    return jnp.array([node.xmin, node.xmax, node.ymin, node.ymax])


def generate_adaptive_mesh_level_restriction_2D(
    root: DiscretizationNode2D,
    f_fn: Callable[[jax.Array], jax.Array],
    tol: float,
    p: int,
    q: int,
    restrict_bool: bool = True,
    l2_norm: bool = False,
) -> None:
    L_4f1 = precompute_L_4f1(p)
    if l2_norm:
        # Get a rough estimate of the L2 norm of the function
        add_four_children(root, root=root, q=q)
        pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)
        f_evals = f_fn(pts)
        # Estimate the squared L2 norm of the function on each child
        patch_nrms = jnp.array(
            [
                get_squared_l2_norm_single_panel(
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
        points_1 = compute_interior_Chebyshev_points_uniform_2D(
            root, L=1, p=p
        ).reshape(-1, 2)
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
            node = DiscretizationNode2D(
                xmin=bounds[0],
                xmax=bounds[1],
                ymin=bounds[2],
                ymax=bounds[3],
            )
            points_0 = compute_interior_Chebyshev_points_uniform_2D(
                node, L=0, p=p
            ).reshape(-1, 2)
            points_1 = compute_interior_Chebyshev_points_uniform_2D(
                node, L=1, p=p
            ).reshape(-1, 2)

            f_evals = f_fn(points_0)
            f_evals_refined = f_fn(points_1)
            f_interp = L_4f1 @ f_evals
            err = get_squared_l2_norm_four_panels(
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
            node = DiscretizationNode2D(
                xmin=bounds[0],
                xmax=bounds[1],
                ymin=bounds[2],
                ymax=bounds[3],
            )
            points_0 = compute_interior_Chebyshev_points_uniform_2D(
                node, L=0, p=p
            ).reshape(-1, 2)
            points_1 = compute_interior_Chebyshev_points_uniform_2D(
                node, L=1, p=p
            ).reshape(-1, 2)

            f_evals = f_fn(points_0)
            f_evals_refined = f_fn(points_1)
            return check_current_discretization_global_linf_norm(
                f_evals, f_evals_refined, L_4f1, tol, global_nrm
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
                # logging.debug(
                #     "generate_adaptive_mesh_level_restriction_2D: Refining node %s",
                #     node,
                # )
                add_four_children(node, root=root, q=q)
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
                        bounds = patches_to_check(for_check, root)
                        for vol in bounds:
                            # logging.debug(
                            #     "generate_adaptive_mesh_level_restriction: Calling find_or_add_child"
                            # )
                            n = find_or_add_child_2D(root, root, q, *vol)
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


def patches_to_check(
    newly_refined: DiscretizationNode2D, root: DiscretizationNode2D
) -> jax.Array:
    """
    Given a newly refined node, find the bounds that need to be checked for the level restriction criterion.
    For instance, if we refine a node with bounds
    [xmin, xmax, ymin, ymax] == [0, 0.5, 0, 0.5],

    and the root bounds are
    [xmin, xmax, ymin, ymax] == [0, 1, 0, 1],

    then the volumes to check are volumes with sidelength 0.5 that are adjacent to the newly refined node:
    [xmin, xmax, ymin, ymax] == [0.5, 1.0, 0, 0.5],
    [xmin, xmax, ymin, ymax] == [0, 0.5, 0.5, 1.0],


    Args:
        newly_refined (Node): _description_
        root (Node): _description_

    Returns:
        jax.Array: Has shape [???, 4] and lists the 2D volume bounds that need to be checked. Second axis lists
        [xmin, xmax, ymin, ymax, zmin, zmax]
    """

    node_sidelen = newly_refined.xmax - newly_refined.xmin
    new_xmin = newly_refined.xmin - node_sidelen
    new_xmax = newly_refined.xmax + node_sidelen
    new_ymin = newly_refined.ymin - node_sidelen
    new_ymax = newly_refined.ymax + node_sidelen

    volumes = []
    if new_xmin >= root.xmin:
        volumes.append(
            [
                new_xmin,
                newly_refined.xmin,
                newly_refined.ymin,
                newly_refined.ymax,
            ]
        )
    if new_xmax <= root.xmax:
        volumes.append(
            [
                newly_refined.xmax,
                new_xmax,
                newly_refined.ymin,
                newly_refined.ymax,
            ]
        )
    if new_ymin >= root.ymin:
        volumes.append(
            [
                newly_refined.xmin,
                newly_refined.xmax,
                new_ymin,
                newly_refined.ymin,
            ]
        )
    if new_ymax <= root.ymax:
        volumes.append(
            [
                newly_refined.xmin,
                newly_refined.xmax,
                newly_refined.ymax,
                new_ymax,
            ]
        )

    return jnp.array(volumes)


def find_or_add_child_2D(
    node: DiscretizationNode2D,
    root: DiscretizationNode2D,
    q: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> DiscretizationNode2D | None:
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
            add_four_children(node, root=root, q=q)
            return node
        else:
            return None

    node_xmid = (node.xmax + node.xmin) / 2
    node_ymid = (node.ymax + node.ymin) / 2

    # Find which part of the octree to look in
    if xmin < node_xmid:
        # It could be in children (a,d) which are idxes (0,3)
        if ymin < node_ymid:
            # Child a
            child = node.children[0]
        else:
            # Child d
            child = node.children[3]
    else:
        # It could be children (b,c) which are idxes (1,2)
        if ymin < node_ymid:
            # Child b
            child = node.children[1]
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

            add_four_children(child, root=root, q=q)
            return child
        else:
            return None

    elif node_sidelen < 2 * requested_sidelen:
        raise ValueError("Requested volume is too large for the current node")

    else:
        return find_or_add_child_2D(child, root, q, xmin, xmax, ymin, ymax)


@partial(jax.jit, static_argnums=(2,))
def get_squared_l2_norm_four_panels(
    f_evals: jax.Array, bounds: jax.Array, p: int
) -> float:
    # Split the corners into eight children
    bounds_lst = bounds_for_quad_subdivision(bounds)
    n_per_voxel = p**2

    # Call get_squared_l2_norm_single_voxel on each child
    out_lst = []
    for i, b in enumerate(bounds_lst):
        f_i = f_evals[i * n_per_voxel : (i + 1) * n_per_voxel]
        out_lst.append(get_squared_l2_norm_single_panel(f_i, b, p))

    return jnp.sum(jnp.array(out_lst))


@partial(jax.jit, static_argnums=(2,))
def get_squared_l2_norm_single_panel(
    f_evals: jax.Array, bounds: jax.Array, p: int
) -> float:
    """
    for f_evals evaluated on a 2D Cheby panel, evaluate the l2 norm of the
    function

    Args:
        f_evals (jnp.array): Has shape (p**2,)
        cheby_weights_1d (jnp.array): The Chebyshev weights for the 1D case. Has shape (p,)
        patch_area (float): The area of the patch
        p (int): The number of Chebyshev points in each dimension
    Returns:
        float: Estimate of the L2 norm
    """
    cheby_weights_x = chebyshev_weights(p, bounds[:2])
    cheby_weights_y = chebyshev_weights(p, bounds[2:])
    r_idxes = rearrange_indices_ext_int(p)
    cheby_weights_2d = jnp.outer(cheby_weights_x, cheby_weights_y).reshape(-1)[
        r_idxes
    ]
    out_val = jnp.sum(f_evals**2 * cheby_weights_2d)
    return out_val
