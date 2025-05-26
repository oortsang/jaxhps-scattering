from typing import Tuple, List
from ._discretization_tree import DiscretizationNode2D
import jax.numpy as jnp
import jax


@jax.jit
def get_four_children(
    parent: DiscretizationNode2D,
) -> Tuple[DiscretizationNode2D]:
    xmid = (parent.xmin + parent.xmax) / 2
    ymid = (parent.ymin + parent.ymax) / 2

    child_a = DiscretizationNode2D(
        depth=parent.depth + 1,
        xmin=parent.xmin,
        xmax=xmid,
        ymin=parent.ymin,
        ymax=ymid,
    )
    child_b = DiscretizationNode2D(
        depth=parent.depth + 1,
        xmin=xmid,
        xmax=parent.xmax,
        ymin=parent.ymin,
        ymax=ymid,
    )
    child_c = DiscretizationNode2D(
        depth=parent.depth + 1,
        xmin=xmid,
        xmax=parent.xmax,
        ymin=ymid,
        ymax=parent.ymax,
    )
    child_d = DiscretizationNode2D(
        depth=parent.depth + 1,
        xmin=parent.xmin,
        xmax=xmid,
        ymin=ymid,
        ymax=parent.ymax,
    )
    return (
        child_a,
        child_b,
        child_c,
        child_d,
    )


def add_four_children(
    add_to: DiscretizationNode2D,
    root: DiscretizationNode2D = None,
    q: int = None,
) -> None:
    """Splits a 2D node into four children. Updates the .children() attribute of the input Node.
    If the root and q are specified, this function will also update the number of quadrature points
    along intermediate nodes' boundaries. Adding nodes will change these quantities, which is why
    this function handles that logic.

    Args:
        add_to (Node): The node which must be split into four children.
        root (Node, optional): The root of the tree. Specified if we want to count number of quadrature points along node boundaries. Defaults to None.
        q (int, optional): Number of quadrature points along the boundary of leaves of the tree. Specified if we want to count number of quadrature points along node boundaries. Defaults to None.
    """
    # logging.debug(
    #     "add_four_children: Called with add_to=%s, root=%s, q=%s",
    #     add_to,
    #     root,
    #     q,
    # )
    if len(add_to.children) == 0:
        children = get_four_children(add_to)
        add_to.children = children

        if q is not None:
            # Set the number of quadrature points along the boundary of each child.
            for child in add_to.children:
                child.n_0 = q
                child.n_1 = q
                child.n_2 = q
                child.n_3 = q

            # Update the number of quadrature points in add_to.
            add_to.n_0 = 2 * q
            add_to.n_1 = 2 * q
            add_to.n_2 = 2 * q
            add_to.n_3 = 2 * q

            # Update the number of quadrature points along the path from add_to to the root. Only do this
            # if the root is not the same as add_to.
            if not tree_equal(add_to, root):
                path_info = find_path_from_root_2D(root, add_to)
                for node in path_info:
                    # If they share a boundary, then we need to update the number of quadrature points.
                    if node.ymin == add_to.ymin:
                        node.n_0 += q

                    if node.xmax == add_to.xmax:
                        node.n_1 += q

                    if node.ymax == add_to.ymax:
                        node.n_2 += q

                    if node.xmin == add_to.xmin:
                        node.n_3 += q


def find_path_from_root_2D(
    root: DiscretizationNode2D,
    node: DiscretizationNode2D,
) -> List[DiscretizationNode2D]:
    """Find the path from the root to the node in a 2D tree.

    Args:
        root (Node): The root of the tree.
        node (Node): The node who is being searched for.

    Returns:
        List[Tuple[Node, int]]: A list of nodes, which provides a traversal from the root to the node.
    """
    if root.children == ():
        raise ValueError("Specified root has no children.")

    for i, child in enumerate(root.children):
        if tree_equal(child, node):
            return [
                root,
            ]

    # Find which child we need to continue searching in.
    root_xmid = (root.xmin + root.xmax) / 2
    root_ymid = (root.ymin + root.ymax) / 2

    if node.xmin < root_xmid:
        # Either child 0 or 3
        if node.ymin < root_ymid:
            # Child 0
            return [root] + find_path_from_root_2D(root.children[0], node)
        else:
            # Child 3
            return [root] + find_path_from_root_2D(root.children[3], node)
    else:
        # Either child 1 or 2
        if node.ymin < root_ymid:
            # Child 1
            return [root] + find_path_from_root_2D(root.children[1], node)
        else:
            # Child 2
            return [root] + find_path_from_root_2D(root.children[2], node)


def get_ordered_lst_of_boundary_nodes(
    root: DiscretizationNode2D,
) -> Tuple[Tuple[DiscretizationNode2D]]:
    """
    Given the root node of an adaptive quadtree, return a list of the boundary nodes in order.
    Starting from the SW corner and moving counter-clockwise around the boundary.

    Args:
        root (DiscretizationNode2D): Is the root of the adaptive quadtree.

    Returns:
        Tuple[DiscretizationNode2D]: All boundary elements, in order.
    """
    s_bdry_nodes = _get_next_S_boundary_node(
        root, (find_node_at_corner(root, xmin=root.xmin, ymin=root.ymin),)
    )
    e_bdry_nodes = _get_next_E_boundary_node(root, (s_bdry_nodes[-1],))
    n_bdry_nodes = _get_next_N_boundary_node(root, (e_bdry_nodes[-1],))
    w_bdry_nodes = _get_next_W_boundary_node(root, (n_bdry_nodes[-1],))
    return (s_bdry_nodes, e_bdry_nodes, n_bdry_nodes, w_bdry_nodes)


def _get_next_S_boundary_node(
    root: DiscretizationNode2D, carry: Tuple[DiscretizationNode2D]
) -> Tuple[DiscretizationNode2D]:
    """Recursively finds the next node on the S boundary of the quadtree."""

    # Base case: start with the child at the SW corner
    last = carry[-1]

    if last.xmax == root.xmax:
        # We've reached the end of the S boundary
        return carry

    next = find_node_at_corner(root, xmin=last.xmax, ymin=root.ymin)
    together = carry + (next,)
    if next.xmax == root.xmax:
        # We've reached the end of the S boundary
        return together
    else:
        return _get_next_S_boundary_node(root, together)


def _get_next_E_boundary_node(
    root: DiscretizationNode2D, carry: Tuple[DiscretizationNode2D]
) -> Tuple[DiscretizationNode2D]:
    """Recursively finds the next node on the E boundary of the quadtree."""

    # Base case: start with the child at the SE corner
    last = carry[-1]

    if last.ymax == root.ymax:
        # We've reached the end of the E boundary
        return carry

    next = find_node_at_corner(root, xmax=root.xmax, ymin=last.ymax)
    together = carry + (next,)
    if next.ymax == root.ymax:
        # We've reached the end of the E boundary
        return together
    else:
        return _get_next_E_boundary_node(root, together)


def _get_next_N_boundary_node(
    root: DiscretizationNode2D, carry: Tuple[DiscretizationNode2D]
) -> Tuple[DiscretizationNode2D]:
    """Recursively finds the next node on the N boundary of the quadtree."""

    # Base case: start with the child at the NE corner
    last = carry[-1]

    if last.xmin == root.xmin:
        # We've reached the end of the N boundary
        return carry

    next = find_node_at_corner(root, xmax=last.xmin, ymax=root.ymax)
    together = carry + (next,)
    if next.xmin == root.xmin:
        # We've reached the end of the N boundary
        return together
    else:
        return _get_next_N_boundary_node(root, together)


def _get_next_W_boundary_node(
    root: DiscretizationNode2D, carry: Tuple[DiscretizationNode2D]
) -> Tuple[DiscretizationNode2D]:
    """Recursively finds the next node on the W boundary of the quadtree."""

    # Base case: start with the child at the NW corner
    last = carry[-1]

    if last.ymin == root.ymin:
        # We've reached the end of the W boundary
        return carry

    next = find_node_at_corner(root, xmin=root.xmin, ymax=last.ymin)
    together = carry + (next,)
    if next.ymin == root.ymin:
        # We've reached the end of the W boundary
        return together
    else:
        return _get_next_W_boundary_node(root, together)


def find_node_at_corner(
    root: DiscretizationNode2D,
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
) -> DiscretizationNode2D:
    at_corner = node_at(root, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    if at_corner:
        # Check whether the root has a child that also matches the specified location.
        # This will happen when searching for a particular corner, i.e. only specifying
        # xmin and ymin, or xmax and ymax.
        if root.children != ():
            for child in root.children:
                if node_at(child, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax):
                    return find_node_at_corner(
                        child, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
                    )
        else:
            # If the root has no children, then the root is the node we are looking for.
            return root

    else:
        node_xmid = (root.xmin + root.xmax) / 2
        node_ymid = (root.ymin + root.ymax) / 2

        x_pred_1 = (xmin is None) or (xmin >= node_xmid)
        x_pred_2 = (xmax is None) or (xmax > node_xmid)
        in_upper_half_x = x_pred_1 and x_pred_2

        y_pred_1 = (ymin is None) or (ymin >= node_ymid)
        y_pred_2 = (ymax is None) or (ymax > node_ymid)
        in_upper_half_y = y_pred_1 and y_pred_2

        if in_upper_half_x:
            if in_upper_half_y:
                return find_node_at_corner(
                    root.children[2],
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                )
            else:
                return find_node_at_corner(
                    root.children[1],
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                )
        else:
            if in_upper_half_y:
                return find_node_at_corner(
                    root.children[3],
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                )
            else:
                return find_node_at_corner(
                    root.children[0],
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                )


@jax.jit
def node_at(
    node: DiscretizationNode2D,
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
) -> DiscretizationNode2D:
    """Tests whether the node is at the specified location. Robust to presence of Nones in the input."""
    bool_xmin = jnp.logical_or((xmin is None), (node.xmin == xmin))
    bool_xmax = jnp.logical_and(
        jnp.logical_or((xmax is None), (node.xmax == xmax)), bool_xmin
    )
    bool_ymin = jnp.logical_and(
        jnp.logical_or((ymin is None), (node.ymin == ymin)), bool_xmax
    )
    bool_ymax = jnp.logical_and(
        jnp.logical_or((ymax is None), (node.ymax == ymax)), bool_ymin
    )
    return bool_ymax


@jax.jit
def tree_equal(
    node_a: DiscretizationNode2D, node_b: DiscretizationNode2D
) -> bool:
    """Checks equality between the metadata of two nodes."""
    _, a = jax.tree_util.tree_flatten(node_a)
    _, b = jax.tree_util.tree_flatten(node_b)
    return a == b
