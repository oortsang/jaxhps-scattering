from jaxhps._discretization_tree import (
    DiscretizationNode3D,
    get_all_leaves,
    get_discretization_node_area,
)
from jaxhps._discretization_tree_operations_3D import (
    add_eight_children,
    get_all_leaves_special_ordering_3D,
    find_nodes_along_interface_3D,
    find_path_from_root_3D,
)
import jax


class Test_get_all_leaves_special_ordering_3D:
    def test_0(self) -> None:
        """Check that specifying a non-standard child traversal order works as
        expected.
        """
        node = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        add_eight_children(node)
        add_eight_children(node.children[0])
        # add_eight_children(node.children[0].children[0])

        child_traversal_order = [7, 6, 5, 4, 3, 2, 1, 0]
        leaves = get_all_leaves_special_ordering_3D(
            node, child_traversal_order=child_traversal_order
        )
        assert len(leaves) == 15

        for i in [0, 1, 2, 3, 4, 5, 6]:
            assert leaves[i].depth == 1

        for i in [7, 8, 9, 10, 11, 12, 13, 14]:
            assert leaves[i].depth == 2
        # assert get_discretization_node_area(leaves[0]) == 1 / 4
        # assert get_discretization_node_area(leaves[1]) == 1 / 4
        # assert get_discretization_node_area(leaves[2]) == 1 / 4
        # assert get_discretization_node_area(leaves[3]) == 1 / 16
        # assert get_discretization_node_area(leaves[4]) == 1 / 16
        # assert get_discretization_node_area(leaves[5]) == 1 / 16
        # assert get_discretization_node_area(leaves[6]) == 1 / 64
        # assert get_discretization_node_area(leaves[7]) == 1 / 64
        # assert get_discretization_node_area(leaves[8]) == 1 / 64
        # assert get_discretization_node_area(leaves[9]) == 1 / 64

        # assert leaves[0].depth == 1
        # assert leaves[1].depth == 1
        # assert leaves[2].depth == 1
        # assert leaves[3].depth == 2
        # assert leaves[4].depth == 2
        # assert leaves[5].depth == 2
        # assert leaves[6].depth == 3
        # assert leaves[7].depth == 3
        # assert leaves[8].depth == 3
        # assert leaves[9].depth == 3
        jax.clear_caches()


class Test_add_eight_children:
    def test_0(self) -> None:
        """Checks function returns without error with uniform refinement."""
        node = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
        )
        add_eight_children(node)
        assert len(node.children) == 8
        for child in node.children:
            assert get_discretization_node_area(child) == 1 / 8
            add_eight_children(child)
            for gchild in child.children:
                assert get_discretization_node_area(gchild) == 1 / 64

    def test_1(self) -> None:
        """Checks discretization point counting in 3 levels of uniform refinement."""
        q = 4
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        add_eight_children(add_to=root, root=root, q=q)
        for child in root.children:
            add_eight_children(add_to=child, root=root, q=q)
            for gchild in child.children:
                add_eight_children(add_to=gchild, root=root, q=q)
        # for leaf in get_all_leaves(root):
        #     add_eight_children(add_to=leaf, root=root, q=q)
        print("test_1: expected_n: ", 64 * q**2)

        assert root.n_0 == 64 * q**2
        assert root.n_1 == 64 * q**2
        assert root.n_2 == 64 * q**2
        assert root.n_3 == 64 * q**2
        assert root.n_4 == 64 * q**2
        assert root.n_5 == 64 * q**2

    def test_2(self) -> None:
        """Checks discretization point counting in non-uniform refinement."""
        q = 4
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        add_eight_children(add_to=root, root=root, q=q)
        add_eight_children(add_to=root.children[0], root=root, q=q)
        add_eight_children(add_to=root.children[0].children[0], root=root, q=q)

        # Faces 1, 3, 4 have only been refined once, so they each have 4 * q**2 points.
        assert root.n_1 == 4 * q**2
        assert root.n_3 == 4 * q**2
        assert root.n_4 == 4 * q**2

        # The other faces have 10 * q**2 points.
        assert root.n_0 == 10 * q**2
        assert root.n_2 == 10 * q**2
        assert root.n_5 == 10 * q**2
        jax.clear_caches()


class Test_find_path_from_root_3D:
    def test_0(self) -> None:
        # Non-uniform grid.
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        q = 3
        add_eight_children(add_to=root, root=root, q=q)
        add_eight_children(add_to=root.children[0], root=root, q=q)

        node = root.children[0].children[0]

        computed_path = find_path_from_root_3D(root, node)
        path_len = len(computed_path)

        assert path_len == 2

        # for i, xx in enumerate(computed_path):
        #     current_node = xx

        #     if i != path_len - 1:
        #         next_node = computed_path[i + 1][0]
        #         assert tree_equal(
        #             current_node.children[current_child_idx],
        #             next_node,
        #         )
        #     else:
        #         assert tree_equal(
        #             current_node.children[current_child_idx], node
        #         )
        jax.clear_caches()


class Test_find_nodes_along_interface_3D:
    def test_0(self) -> None:
        # Uniform case
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        l = 3
        q = 4
        for _ in range(l):
            leaves = get_all_leaves(root)
            for leaf in leaves:
                add_eight_children(add_to=leaf, root=root, q=q)

        x_interface = 0.5
        nodes = find_nodes_along_interface_3D(root, xval=x_interface)
        assert len(nodes) == 2
        for node_lst in nodes:
            assert len(node_lst) == 4**l

        y_interface = 0.25
        nodes = find_nodes_along_interface_3D(root, yval=y_interface)
        assert len(nodes) == 2
        for node_lst in nodes:
            assert len(node_lst) == 4**l

        for node in nodes[0]:
            assert node.ymax == y_interface
        for node in nodes[1]:
            assert node.ymin == y_interface
        jax.clear_caches()

    def test_1(self) -> None:
        # Non-uniform case
        root = DiscretizationNode3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        add_eight_children(add_to=root)
        add_eight_children(add_to=root.children[0])

        x_interface = 0.5
        nodes = find_nodes_along_interface_3D(root, xval=x_interface)
        print("test_1: nodes lengths: ", [len(x) for x in nodes])
        assert len(nodes) == 2
        assert len(nodes[0]) == 7
        assert len(nodes[1]) == 4
        jax.clear_caches()
