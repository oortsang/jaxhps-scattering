from jaxhps._discretization_tree import (
    DiscretizationNode2D,
    get_discretization_node_area,
    get_all_leaves,
)
from jaxhps._discretization_tree_operations_2D import (
    add_four_children,
    find_node_at_corner,
    node_at,
    tree_equal,
    find_path_from_root_2D,
    get_four_children,
)
import jax


class Test_get_four_children:
    def test_0(self) -> None:
        node = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        c = get_four_children(node)
        assert len(c) == 4
        for i in range(4):
            assert isinstance(c[i], DiscretizationNode2D)
            print("test_0: c_i.children= ", c[i].children)
            # Check that children is a tuple
            assert isinstance(c[i].children, tuple)
            assert len(c[0].children) == 0
        jax.clear_caches()


class Test_add_four_children:
    def test_0(self) -> None:
        node = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        print("test_0: node = ", id(node))
        print("test_0: node: ", node)
        print("test_0: node.children = ", node.children)
        add_four_children(node)
        print("test_0: node.children = ", node.children)

        assert len(node.children) == 4

    def test_1(self, caplog) -> None:
        """Checks two levels works"""
        print("test_1: Started")
        node = DiscretizationNode2D(
            children=(),
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        q = 3

        print("test_1: Making children")
        add_four_children(root=node, add_to=node, q=q)
        print("test_1: node.children = ", node.children)
        assert len(node.children) == 4
        for child in node.children:
            print("test_1: working on child = ", child)
            add_four_children(root=node, add_to=child, q=q)
            assert len(child.children) == 4
            for gchild in child.children:
                print("test_1: working on gchild = ", gchild)

                assert get_discretization_node_area(gchild) == 1 / 16

        # Check that counting number of boundary points is performed correctly.
        for child in node.children:
            assert child.n_0 == 2 * q
            assert child.n_1 == 2 * q
            assert child.n_2 == 2 * q
            assert child.n_3 == 2 * q

        print(
            "test_1: root # of quadrature points: ",
            node.n_0,
            node.n_1,
            node.n_2,
            node.n_3,
        )
        print(
            "test_1: Expected # of quadrature points: ",
            4 * q,
            4 * q,
            4 * q,
            4 * q,
        )
        assert node.n_0 == 4 * q
        assert node.n_1 == 4 * q
        assert node.n_2 == 4 * q
        assert node.n_3 == 4 * q

    def test_2(self) -> None:
        """Checks discretization point counting in non-uniform grid."""
        root = DiscretizationNode2D(
            children=(),
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        q = 3
        # Each side of root now has 2 panels.
        add_four_children(add_to=root, root=root, q=q)

        for child in root.children:
            # Each side of root now has 4 panels.
            add_four_children(add_to=child, root=root, q=q)

        # n_0 and n_3 of root now have 5 panels each.
        add_four_children(add_to=root.children[0].children[0], root=root, q=q)

        # Check that counting number of boundary points is performed correctly.
        # Root
        print(
            "test_1: root # of quadrature points: ",
            root.n_0,
            root.n_1,
            root.n_2,
            root.n_3,
        )
        assert root.n_0 == 5 * q
        assert root.n_1 == 4 * q
        assert root.n_2 == 4 * q
        assert root.n_3 == 5 * q

    def test_3(self) -> None:
        """Checks discretization point counting in uniform grid of 3 levels of refinement."""
        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        l = 3
        q = 3
        for i in range(l):
            print("test_3: level i = ", i)
            for leaf in get_all_leaves(root):
                print("test_3: splitting leaf = ", leaf)
                add_four_children(add_to=leaf, root=root, q=q)
                print(
                    "test_3: root # of quadrature points: ",
                    root.n_0,
                    root.n_1,
                    root.n_2,
                    root.n_3,
                )

        for i in range(4):
            assert root.children[i].n_0 == (2 ** (l - 1)) * q
            assert root.children[i].n_1 == (2 ** (l - 1)) * q
            assert root.children[i].n_2 == (2 ** (l - 1)) * q
            assert root.children[i].n_3 == (2 ** (l - 1)) * q

        print(
            "test_3: root # of quadrature points: ",
            root.n_0,
            root.n_1,
            root.n_2,
            root.n_3,
        )
        print(
            "test_3: Expected # of quadrature points: ",
            (2**l) * q,
            (2**l) * q,
            (2**l) * q,
            (2**l) * q,
        )
        assert root.n_0 == (2**l) * q
        assert root.n_1 == (2**l) * q
        assert root.n_2 == (2**l) * q
        assert root.n_3 == (2**l) * q

        jax.clear_caches()


class Test_find_node_at_corner:
    def test_0(self) -> None:
        """Checks that find_node_at works without error."""
        node = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(node)
        add_four_children(node.children[0])
        add_four_children(node.children[0].children[0])

        # Search on xmin, ymin at the corner of the domain
        found_a = find_node_at_corner(node, xmin=node.xmin, ymin=node.ymin)
        expected_a = node.children[0].children[0].children[0]
        assert tree_equal(found_a, expected_a)

        # Search on xmin, ymin NOT at the corner of the domain
        found_b = find_node_at_corner(
            node, xmin=node.children[0].children[1].xmin, ymin=node.ymin
        )
        expected_b = node.children[0].children[1]
        assert tree_equal(found_b, expected_b)

        # Search on xmin, ymax NOT at the corner of the domain.
        expected_c = node.children[0].children[0].children[1]
        found_c = find_node_at_corner(
            node, xmin=expected_c.xmin, ymax=expected_c.ymax
        )
        assert tree_equal(found_c, expected_c)

        # Search on xmax, ymin NOT at the corner of the domain.
        expected_d = node.children[0].children[0].children[1]
        found_d = find_node_at_corner(
            node, xmax=expected_d.xmax, ymin=expected_d.ymin
        )
        assert tree_equal(found_d, expected_d)

        # Search on xmax, ymax NOT at the corner of the domain.
        expected_e = node.children[0].children[0].children[1]
        found_e = find_node_at_corner(
            node, xmax=expected_e.xmax, ymax=expected_e.ymax
        )
        assert tree_equal(found_e, expected_e)
        jax.clear_caches()


class Test_node_at:
    def test_0(self) -> None:
        a = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        x = node_at(a, xmin=0.0)
        assert x

        y = node_at(a, ymin=1.0)
        assert not y
        jax.clear_caches()


class Test_find_path_from_root_2D:
    def test_0(self) -> None:
        # Non-uniform grid.
        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(add_to=root, root=root)
        add_four_children(add_to=root.children[0], root=root)

        node = root.children[0].children[0]

        computed_path = find_path_from_root_2D(root, node)
        path_len = len(computed_path)

        assert path_len == 2
        # for i, xx in enumerate(computed_path):
        #     current_node = xx

        jax.clear_caches()
