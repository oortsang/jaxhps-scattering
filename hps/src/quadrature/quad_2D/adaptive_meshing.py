"""This file has functions to create an adaptive mesh for 3D problems."""

import logging
from functools import partial
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from hps.src.quadrature.quadrature_utils import (
    check_current_discretization_global_linf_norm,
    EPS,
    chebyshev_weights,
)
from hps.src.quadrature.quad_2D.indexing import (
    _rearrange_indices,
)
from hps.src.quadrature.quad_2D.grid_creation import (
    get_all_leaf_2d_cheby_points_uniform_refinement,
)

from hps.src.quadrature.trees import (
    Node,
    get_all_leaves,
    add_four_children,
    get_all_leaves_jitted,
    _corners_for_quad_subdivision,
)


@jax.jit
def node_corners_to_2d_corners(node: Node) -> jnp.ndarray:
    return jnp.array(
        [
            [node.xmin, node.ymin],
            [node.xmax, node.ymin],
            [node.xmax, node.ymax],
            [node.xmin, node.ymax],
        ]
    )


@partial(jax.jit, static_argnums=(2,))
def get_squared_l2_norm_single_panel(
    f_evals: jnp.array, corners: jnp.array, p: int
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
    xmin = corners[0, 0]
    xmax = corners[1, 0]
    ymin = corners[0, 1]
    ymax = corners[2, 1]
    cheby_weights_x = chebyshev_weights(p, jnp.array([xmin, xmax]))
    cheby_weights_y = chebyshev_weights(p, jnp.array([ymin, ymax]))
    r_idxes = _rearrange_indices(p)
    cheby_weights_2d = jnp.outer(cheby_weights_x, cheby_weights_y).reshape(-1)[
        r_idxes
    ]
    out_val = jnp.sum(f_evals**2 * cheby_weights_2d)
    return out_val


@partial(jax.jit, static_argnums=(6,))
def check_current_discretization_relative_global_l2_norm(
    f_evals: jnp.ndarray,
    f_evals_refined: jnp.ndarray,
    refinement_op: jnp.ndarray,
    tol: float,
    global_l2_norm: float,
    corner_set: jnp.ndarray,
    p: int,
) -> bool:
    # first, interpolate f_evals to the four-panel grid
    f_interp = refinement_op @ f_evals

    # compute the errors
    errs = f_interp - f_evals_refined

    corners_of_children = _corners_for_quad_subdivision(corner_set)

    n_per_child = p**2

    # compute the L2 norm of the error of each panel
    l2_nrms = []
    for i, child in enumerate(corners_of_children):
        errs_i = errs[i * n_per_child : (i + 1) * n_per_child]
        l2_nrm = get_squared_l2_norm_single_panel(errs_i, child, p)
        l2_nrms.append(l2_nrm)

    # compute the L2 norm of the four panels
    err_l2_norm = jnp.sqrt(jnp.sum(jnp.array(l2_nrms)))

    # enforce the error criterion
    return (err_l2_norm / global_l2_norm) < tol


def generate_adaptive_mesh_l2(
    root: Node,
    refinement_op: jnp.ndarray,
    f_fn: Callable[[jnp.ndarray], jnp.ndarray],
    tol: float,
    p: int,
    q: int,
    level_restriction_bool: bool = False,
) -> None:
    ###################################################
    # Define functions to be used in the main routine #

    def get_l2_norm(corner_set: jnp.array) -> float:
        pts_0 = get_all_leaf_2d_cheby_points_uniform_refinement(
            p, 0, corner_set
        ).reshape(-1, 2)
        f_evals = f_fn(pts_0)
        out_val = jnp.array(
            [
                get_squared_l2_norm_single_panel(f_evals, corner_set, p),
            ]
        )
        return out_val

    vmapped_get_l2_norm = jax.vmap(get_l2_norm, in_axes=(0,))

    def check_single_node(
        corner_set: jnp.array, global_l2_norm: float
    ) -> bool:
        """Input has shape (4, 2). Gives the corners of the panel, starting with
        the SW corner and going counter-clockwise."""
        pts_0 = get_all_leaf_2d_cheby_points_uniform_refinement(
            p, 0, corner_set
        ).reshape(-1, 2)
        pts_1 = get_all_leaf_2d_cheby_points_uniform_refinement(
            p, 1, corner_set
        ).reshape(-1, 2)
        f_evals = f_fn(pts_0)
        f_evals_refined = f_fn(pts_1)

        return check_current_discretization_relative_global_l2_norm(
            f_evals,
            f_evals_refined,
            refinement_op,
            tol,
            global_l2_nrm,
            corner_set,
            p,
        )

    vmapped_check_queue = jax.vmap(check_single_node, in_axes=(0, None))

    refinement_check_queue = list(get_all_leaves(root))
    refinement_check_corners = jnp.array(
        [node_corners_to_2d_corners(node) for node in refinement_check_queue]
    )

    l2_nrms = vmapped_get_l2_norm(refinement_check_corners)
    global_l2_nrm = jnp.sqrt(jnp.sum(jnp.array(l2_nrms)))

    # Loop through the queue and refine nodes as necessary
    while len(refinement_check_queue):
        logging.debug(
            "generate_adaptive_mesh_l2: Queue length: %s",
            len(refinement_check_queue),
        )
        checks_bool = vmapped_check_queue(
            refinement_check_corners, global_l2_nrm
        )

        new_refinement_check_queue = []

        for i, node in enumerate(refinement_check_queue):
            if not checks_bool[i]:
                add_four_children(add_to=node, root=root, q=q)
                new_refinement_check_queue.extend(node.children)

                # Check the level restriction criterion
                if level_restriction_bool:
                    level_restriction_check_queue = [
                        node,
                    ]
                    # find_or_add_child will sometimes refine a node. If it does so,
                    # we need to do new checks to make sure the level restriction is satisfied.
                    # ref_check_lst is a stack of nodes whose neighbors need to be checked.
                    while len(level_restriction_check_queue):
                        # Get one of the nodes which needs to be checked
                        for_check = level_restriction_check_queue.pop()
                        # Compute the neighbors of the node
                        bounds = patches_to_check(for_check, root)
                        for patch in bounds:
                            # Check each one of the neighbors. If this results in a refinement,
                            # we need to add the newly refined node to the stack.
                            n = find_or_add_child(root, *patch)
                            if n is not None:
                                # print("Adding node due to level restriction: ", n)
                                level_restriction_check_queue.extend(n)
                                new_refinement_check_queue.extend(n)

        refinement_check_queue = new_refinement_check_queue
        refinement_check_corners = jnp.array(
            [
                node_corners_to_2d_corners(node)
                for node in refinement_check_queue
            ]
        )
        all_leaf_corners = jnp.array(
            [node_corners_to_2d_corners(node) for node in get_all_leaves(root)]
        )
        all_l2_norms = vmapped_get_l2_norm(all_leaf_corners)
        logging.debug(
            "generate_adaptive_mesh_l2: # leaves: %i", len(all_leaf_corners)
        )
        global_l2_nrm = jnp.sqrt(jnp.sum(jnp.array(all_l2_norms)))
        logging.debug(
            "generate_adaptive_mesh_l2: global_l2_nrm: %f", global_l2_nrm
        )
        if jnp.isnan(global_l2_nrm):
            # Find the index of the NaN
            idx = jnp.where(jnp.isnan(all_l2_norms))[0][0]
            print("generate_adaptive_mesh_l2: NaN idx = ", idx)
            # Print out the particular node which gives us this index
            print(
                "generate_adaptive_mesh_l2: NaN node = ",
                get_all_leaves(root)[idx],
            )

            raise ValueError("global_l2_nrm is NaN")


def generate_adaptive_mesh_linf(
    root: Node,
    refinement_op: jnp.ndarray,
    f_fn: Callable[[jnp.ndarray], jnp.ndarray],
    tol: float,
    p: int,
    q: int,
    level_restriction_bool: bool = False,
) -> None:
    ###################################################
    # Define functions to be used in the main routine #

    global_linf_nrm = EPS

    def check_single_node(
        corner_set: jnp.array, global_linf_norm: float
    ) -> Tuple[bool, float]:
        """Input has shape (4, 2). Gives the corners of the panel, starting with
        the SW corner and going counter-clockwise."""
        pts_0 = get_all_leaf_2d_cheby_points_uniform_refinement(
            p, 0, corner_set
        ).reshape(-1, 2)
        pts_1 = get_all_leaf_2d_cheby_points_uniform_refinement(
            p, 1, corner_set
        ).reshape(-1, 2)
        f_evals = f_fn(pts_0)
        f_evals_refined = f_fn(pts_1)

        return check_current_discretization_global_linf_norm(
            f_evals,
            f_evals_refined,
            refinement_op,
            tol,
            global_linf_norm,
        )

    vmapped_check_queue = jax.vmap(
        check_single_node, in_axes=(0, None), out_axes=(0, 0)
    )

    refinement_check_queue = list(get_all_leaves(root))
    refinement_check_corners = jnp.array(
        [node_corners_to_2d_corners(node) for node in refinement_check_queue]
    )

    # Loop through the queue and refine nodes as necessary
    while len(refinement_check_queue):
        logging.debug(
            "generate_adaptive_mesh_linf: Queue length: %i",
            len(refinement_check_queue),
        )
        checks_bool, linf_nrms_arr = vmapped_check_queue(
            refinement_check_corners, global_linf_nrm
        )
        # print("generate_adaptive_mesh_l2: checks_bool shape: ", checks_bool.shape)
        # print("generate_adaptive_mesh_l2: linf_nrms_arr shape: ", linf_nrms_arr.shape)
        # print("generate_adaptive_mesh_l2: checks_bool: ", checks_bool)
        # print("generate_adaptive_mesh_l2: linf_nrms_arr: ", linf_nrms_arr)
        global_linf_nrm = jnp.max(linf_nrms_arr)

        new_refinement_check_queue = []
        for i, node in enumerate(refinement_check_queue):
            if not checks_bool[i]:
                add_four_children(add_to=node, root=root, q=q)
                new_refinement_check_queue.extend(node.children)

                # Check the level restriction criterion
                if level_restriction_bool:
                    level_restriction_check_queue = [
                        node,
                    ]
                    # find_or_add_child will sometimes refine a node. If it does so,
                    # we need to do new checks to make sure the level restriction is satisfied.
                    # ref_check_lst is a stack of nodes whose neighbors need to be checked.
                    while len(level_restriction_check_queue):
                        # Get one of the nodes which needs to be checked
                        for_check = level_restriction_check_queue.pop()
                        # Compute the neighbors of the node
                        bounds = patches_to_check(for_check, root)
                        for patch in bounds:
                            # Check each one of the neighbors. If this results in a refinement,
                            # we need to add the newly refined node to the stack.
                            n = find_or_add_child(root, root, q, *patch)
                            if n is not None:
                                # print(
                                #     "generate_adaptive_mesh_linf: Adding node due to level restriction: ",
                                #     n,
                                # )
                                level_restriction_check_queue.append(n)
                                new_refinement_check_queue.extend(n.children)

        refinement_check_queue = new_refinement_check_queue
        refinement_check_corners = jnp.array(
            [
                node_corners_to_2d_corners(node)
                for node in refinement_check_queue
            ]
        )
        logging.debug(
            "generate_adaptive_mesh_linf: # leaves: %i",
            len(get_all_leaves_jitted(root)),
        )


def patches_to_check(newly_refined: Node, root: Node) -> jnp.array:
    """Given a node which has been refined, make a list of bounds for the patches which also need to be refined.
    This will be used in conjunction with find_or_add_child to make sure the tree is properly refined.

    Imagine newly_refined is a patch inside root. There are four patches which are adjacent to newly_refined.
    The conditions check which of these four patches are wholly inside root. If they are, they are added to the list.
    """
    node_sidelen = newly_refined.xmax - newly_refined.xmin
    new_xmin = newly_refined.xmin - node_sidelen
    new_xmax = newly_refined.xmax + node_sidelen
    new_ymin = newly_refined.ymin - node_sidelen
    new_ymax = newly_refined.ymax + node_sidelen

    patches = []
    if new_xmin >= root.xmin:
        # print("patches_to_check: new_xmin >= root.xmin")
        patches.append(
            (
                new_xmin,
                newly_refined.xmin,
                newly_refined.ymin,
                newly_refined.ymax,
            )
        )
    if new_xmax <= root.xmax:
        # print("patches_to_check: new_xmax <= root.xmax")
        patches.append(
            (
                newly_refined.xmax,
                new_xmax,
                newly_refined.ymin,
                newly_refined.ymax,
            )
        )
    if new_ymin >= root.ymin:
        # print("patches_to_check: new_ymin >= root.ymin")
        patches.append(
            (
                newly_refined.xmin,
                newly_refined.xmax,
                new_ymin,
                newly_refined.ymin,
            )
        )
    if new_ymax <= root.ymax:
        # print("patches_to_check: new_ymax <= root.ymax")
        patches.append(
            (
                newly_refined.xmin,
                newly_refined.xmax,
                newly_refined.ymax,
                new_ymax,
            )
        )

    # print("patches_to_check: patches = ", patches)
    return patches


def find_or_add_child(
    node: Node,
    root: Node,
    q: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Tuple[Node] | None:
    """Given a node, search through the tree to see if a child with the specified bounds exist.
    This function expects such a child to be geometrically possible.

    Args:
        node (Node): The root node of the tree
        xmin (float): The minimum x-coordinate of the child
        xmax (float): The maximum x-coordinate of the child
        ymin (float): The minimum y-coordinate of the child
        ymax (float): The maximum y-coordinate of the child
    """

    node_sidelen = node.xmax - node.xmin
    requested_sidelen = xmax - xmin

    # if requested_sidelen * 2 = node_sidelen, we can make sure
    # node has children and early exit.
    if jnp.allclose(requested_sidelen * 2, node_sidelen):
        if len(node.children) == 0:
            add_four_children(add_to=node, root=root, q=q)
            return node
        return None

    node_xmid = (node.xmin + node.xmax) / 2
    node_ymid = (node.ymin + node.ymax) / 2

    # Find which quadrant of the node the patch belongs to
    if xmin < node_xmid:
        # The patch either belongs to child 0 or 3
        if ymin < node_ymid:
            # The patch belongs to child 0
            child = node.children[0]
        else:
            # The patch belongs to child 3
            child = node.children[3]
    else:
        # The patch either belongs to child 1 or 2
        if ymin < node_ymid:
            # The paatch belongs to child 1
            child = node.children[1]
        else:
            # The patch belongs to child 2
            child = node.children[2]

    # If node_sidelen == 4 * requested_sidelen, then we can
    # add the requested patch to the tree. Otherwise, we need to
    # recurse by calling find_or_add_child on the child.
    if jnp.allclose(node_sidelen, 4 * requested_sidelen):
        if len(child.children) == 0:
            add_four_children(add_to=child, root=root, q=q)
            return child
        else:
            return None

    elif node_sidelen < 2 * requested_sidelen:
        raise ValueError("The requested patch is too large for the node.")

    else:
        # print("find_or_add_child: Recursing on child: ", child)
        return find_or_add_child(child, root, q, xmin, xmax, ymin, ymax)
