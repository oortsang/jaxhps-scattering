import jax.numpy as jnp
import jax
from ._discretization_tree import DiscretizationNode3D, get_all_leaves
from typing import Tuple, List


def get_all_leaves_special_ordering_3D(
    node: DiscretizationNode3D, child_traversal_order: jnp.array = None
) -> Tuple[DiscretizationNode3D]:
    if child_traversal_order is None:
        child_traversal_order = jnp.arange(8, dtype=jnp.int32)

    # Different if block from the one above.
    if node.children == ():
        return (node,)
    else:
        leaves = ()
        for child_idx in child_traversal_order:
            child = node.children[child_idx]
            leaves += get_all_leaves_special_ordering_3D(
                child, child_traversal_order=child_traversal_order
            )
        return leaves


def add_eight_children(
    add_to: DiscretizationNode3D,
    root: DiscretizationNode3D = None,
    q: int = None,
) -> None:
    """
    Splits the node's 3D volume into 8 children, and assigns them to the children attribute of the node.

    Lists the children in this order:

    a: xmin...xmid, ymin...ymid, zmid...zmax
    b: xmid...xmax, ymin...ymid, zmid...zmax
    c: xmid...xmax, ymid...ymax, zmid...zmax
    d: xmin...xmid, ymid...ymax, zmid...zmax
    e: xmin...xmid, ymin...ymid, zmin...zmid
    f: xmid...xmax, ymin...ymid, zmin...zmid
    g: xmid...xmax, ymid...ymax, zmin...zmid
    h: xmin...xmid, ymid...ymax, zmin...zmid

    If the root and q qre specified, this function will also update the number of quadrature points
    along intermediate nodes' boundaries. Adding nodes will change these quantities, which is why
    this function handles that logic.

    Args:
        add_to (DiscretizationNode3D): The node whose volume is being split into 8 children.
    """
    # This if statement meant to guard against adding children to a node that already has children.
    # and messing up the n_i values.
    if len(add_to.children) == 0:
        add_to.children = get_eight_children(add_to)

        if q is not None:
            # Set the number of quadrature points along the boundary of each child.
            q_squared = q**2
            for child in add_to.children:
                child.n_0 = q_squared
                child.n_1 = q_squared
                child.n_2 = q_squared
                child.n_3 = q_squared
                child.n_4 = q_squared
                child.n_5 = q_squared

            # Update the number of quadrature points in add_to.
            add_to.n_0 = 4 * q_squared
            add_to.n_1 = 4 * q_squared
            add_to.n_2 = 4 * q_squared
            add_to.n_3 = 4 * q_squared
            add_to.n_4 = 4 * q_squared
            add_to.n_5 = 4 * q_squared

            # Update the number of quadrature points along the path from add_to to the root. Only do this
            # if the root is not the same as add_to.
            if not tree_equal(add_to, root):
                path_info = find_path_from_root_3D(root, add_to)
                for node in path_info:
                    # If they share a boundary, then we need to update the number of quadrature points.
                    if node.xmin == add_to.xmin:
                        node.n_0 += 3 * q_squared

                    if node.xmax == add_to.xmax:
                        node.n_1 += 3 * q_squared

                    if node.ymin == add_to.ymin:
                        node.n_2 += 3 * q_squared

                    if node.ymax == add_to.ymax:
                        node.n_3 += 3 * q_squared

                    if node.zmin == add_to.zmin:
                        node.n_4 += 3 * q_squared

                    if node.zmax == add_to.zmax:
                        node.n_5 += 3 * q_squared


@jax.jit
def get_eight_children(
    parent: DiscretizationNode3D,
) -> Tuple[DiscretizationNode3D]:
    """
    Splits the node's 3D volume into 8 children, and returns a tuple of these children.

    Lists the children in this order:

    a: xmin...xmid, ymin...ymid, zmid...zmax
    b: xmid...xmax, ymin...ymid, zmid...zmax
    c: xmid...xmax, ymid...ymax, zmid...zmax
    d: xmin...xmid, ymid...ymax, zmid...zmax
    e: xmin...xmid, ymin...ymid, zmin...zmid
    f: xmid...xmax, ymin...ymid, zmin...zmid
    g: xmid...xmax, ymid...ymax, zmin...zmid
    h: xmin...xmid, ymid...ymax, zmin...zmid


    Args:
        parent (DiscretizationNode3D): The node whose volume is being split into 8 children.
    """
    xmid = (parent.xmin + parent.xmax) / 2
    ymid = (parent.ymin + parent.ymax) / 2
    zmid = (parent.zmin + parent.zmax) / 2

    child_a = DiscretizationNode3D(
        xmin=parent.xmin,
        xmax=xmid,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=zmid,
        zmax=parent.zmax,
        depth=parent.depth + 1,
    )
    child_b = DiscretizationNode3D(
        xmin=xmid,
        xmax=parent.xmax,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=zmid,
        zmax=parent.zmax,
        depth=parent.depth + 1,
    )
    child_c = DiscretizationNode3D(
        xmin=xmid,
        xmax=parent.xmax,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=zmid,
        zmax=parent.zmax,
        depth=parent.depth + 1,
    )
    child_d = DiscretizationNode3D(
        xmin=parent.xmin,
        xmax=xmid,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=zmid,
        zmax=parent.zmax,
        depth=parent.depth + 1,
    )
    child_e = DiscretizationNode3D(
        xmin=parent.xmin,
        xmax=xmid,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=parent.zmin,
        zmax=zmid,
        depth=parent.depth + 1,
    )
    child_f = DiscretizationNode3D(
        xmin=xmid,
        xmax=parent.xmax,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=parent.zmin,
        zmax=zmid,
        depth=parent.depth + 1,
    )
    child_g = DiscretizationNode3D(
        xmin=xmid,
        xmax=parent.xmax,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=parent.zmin,
        zmax=zmid,
        depth=parent.depth + 1,
    )
    child_h = DiscretizationNode3D(
        xmin=parent.xmin,
        xmax=xmid,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=parent.zmin,
        zmax=zmid,
        depth=parent.depth + 1,
    )

    return (
        child_a,
        child_b,
        child_c,
        child_d,
        child_e,
        child_f,
        child_g,
        child_h,
    )


def find_path_from_root_3D(
    root: DiscretizationNode3D, node: DiscretizationNode3D
) -> List[DiscretizationNode3D]:
    """Find the path from the root to the node in a 3D tree.

    Args:
        root (Node): The root of the tree.
        node (Node): The node who is being searched for.
    Returns:
        List[Node]: A list of nodes, which provides a traversal from the root to the node.
    """

    if root.children == ():
        raise ValueError("Specified root has no children.")

    for child in root.children:
        if tree_equal(child, node):
            return [root]

    # Find which child we need to continue searching in.
    root_xmid = (root.xmin + root.xmax) / 2
    root_ymid = (root.ymin + root.ymax) / 2
    root_zmid = (root.zmin + root.zmax) / 2

    if node.xmin < root_xmid:
        # Either child 0, 3, 4, or 7
        if node.ymin < root_ymid:
            # Either child 0 or 4
            if node.zmin < root_zmid:
                # Child 4
                return [root] + find_path_from_root_3D(root.children[4], node)
            else:
                # Child 0
                return [root] + find_path_from_root_3D(root.children[0], node)
        else:
            # Either child 3 or 7
            if node.zmin < root_zmid:
                # Child 7
                return [root] + find_path_from_root_3D(root.children[7], node)
            else:
                # Child 3
                return [root] + find_path_from_root_3D(root.children[3], node)
    else:
        # Either child 1, 2, 5, or 6
        if node.ymin < root_ymid:
            # Either child 1 or 5
            if node.zmin < root_zmid:
                # Child 5
                return [root] + find_path_from_root_3D(root.children[5], node)
            else:
                # Child 1
                return [root] + find_path_from_root_3D(root.children[1], node)
        else:
            # Either child 2 or 6
            if node.zmin < root_zmid:
                # Child 6
                return [root] + find_path_from_root_3D(root.children[6], node)
            else:
                # Child 2
                return [root] + find_path_from_root_3D(root.children[2], node)


def find_nodes_along_interface_3D(
    root: DiscretizationNode3D,
    xval: float = None,
    yval: float = None,
    zval: float = None,
) -> Tuple[List[DiscretizationNode3D]]:
    """
    For a given xval, yval, or zval, find all the leaves of the octree are bordered by the plane
    x = xval, y = yval, or z = zval. This function expects only one of xval, yval, or zval to be specified.

    This function also assumes that xval, yval, or zval falls along leaf boundaries, but does not
    intersect the interior of any leaves. Thus the function returns two different Lists of
    Nodes, one for each side of the interface.

    Args:
        root (Node): The root of the possibly non-uniform quadtree.
        xval (float, optional): x value defining an interface. Defaults to None.
        yval (float, optional): y value defining an interface. Defaults to None.
        zval (float, optional): z value defining an interface. Defaults to None.

    Returns:
        Tuple[List[Node]]: neg_side_lst: List of nodes on the negative side of the interface.
                           pos_side_lst: List of nodes on the positive side of the interface.
    """
    # Ensure only one of xval, yval, zval are specified
    if sum([xval is not None, yval is not None, zval is not None]) != 1:
        raise ValueError(
            f"Only one of xval, yval, or zval can be specified. Input args: {xval}, {yval}, {zval}"
        )

    if xval is not None:
        # Find the leaf nodes that are bordered by the plane x = xval.
        neg_side_lst = []
        pos_side_lst = []
        for leaf in get_all_leaves(root):
            if leaf.xmin == xval:
                pos_side_lst.append(leaf)
            elif leaf.xmax == xval:
                neg_side_lst.append(leaf)
        return neg_side_lst, pos_side_lst

    if yval is not None:
        # Find the leaf nodes that are bordered by the plane y = yval.
        neg_side_lst = []
        pos_side_lst = []
        for leaf in get_all_leaves(root):
            if leaf.ymin == yval:
                pos_side_lst.append(leaf)
            elif leaf.ymax == yval:
                neg_side_lst.append(leaf)
        return neg_side_lst, pos_side_lst

    if zval is not None:
        # Find the leaf nodes that are bordered by the plane z = zval.
        neg_side_lst = []
        pos_side_lst = []
        for leaf in get_all_leaves(root):
            if leaf.zmin == zval:
                pos_side_lst.append(leaf)
            elif leaf.zmax == zval:
                neg_side_lst.append(leaf)

        return neg_side_lst, pos_side_lst


@jax.jit
def tree_equal(
    node_a: DiscretizationNode3D, node_b: DiscretizationNode3D
) -> bool:
    """Checks equality between the metadata of two nodes and the metadata of all of their children."""
    _, a = jax.tree_util.tree_flatten(node_a)
    _, b = jax.tree_util.tree_flatten(node_b)
    return a == b
