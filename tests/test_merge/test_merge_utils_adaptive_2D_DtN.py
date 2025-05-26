import pytest
import jax.numpy as jnp
import jax
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._discretization_tree_operations_2D import add_four_children


from jaxhps.merge._utils_adaptive_2D_DtN import (
    _find_compression_list_5,
    _find_compression_list_6,
    find_compression_lists_2D,
)


class Test__find_compression_list_5:
    def test_0(self) -> None:
        """
        Test that we can find the compression list for a simple case.
        """
        # Create a tree with 1 node
        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(root)

        out_a, out_b = _find_compression_list_5(
            root.children[0], root.children[1]
        )

        assert len(out_a) == 1
        assert len(out_b) == 1

        assert not out_a[0]
        assert not out_b[0]

    def test_1(self) -> None:
        """
        More complicated case with non-uniform refinement"""
        # p = 4
        q = 2

        print("test_3: q = ", q)

        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(add_to=root, root=root, q=q)
        for child in root.children:
            add_four_children(add_to=child, root=root, q=q)

        add_four_children(add_to=root.children[0].children[2], root=root, q=q)

        # Expect out_a to look like [False, True, True]
        # # and out_b to look like [False, False]
        out_a, out_b = _find_compression_list_5(
            root.children[0], root.children[1]
        )

        assert len(out_a) == 3
        assert len(out_b) == 2

        assert jnp.all(out_a == jnp.array([False, True, True]))
        assert jnp.all(out_b == jnp.array([False, False]))

    def test_2(self) -> None:
        """
        More complicated case with non-uniform refinement.
        """
        # p = 4
        q = 2

        print("test_3: q = ", q)

        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(add_to=root, root=root, q=q)
        for child in root.children:
            add_four_children(add_to=child, root=root, q=q)
            for grandchild in child.children:
                add_four_children(add_to=grandchild, root=root, q=q)

        # Add a few different leaves inside child a
        add_four_children(
            add_to=root.children[0].children[1].children[1], root=root, q=q
        )
        add_four_children(
            add_to=root.children[0].children[1].children[2], root=root, q=q
        )
        add_four_children(
            add_to=root.children[0].children[2].children[1], root=root, q=q
        )

        out_a, out_b = _find_compression_list_5(
            root.children[0], root.children[1]
        )

        expected_out_a = jnp.array([True, True, True, True, True, True, False])
        expected_out_b = jnp.array([False, False, False, False])

        print("test_2: out_a: ", out_a)
        print("test_2: out_b: ", out_b)
        assert jnp.all(out_a == expected_out_a)
        assert jnp.all(out_b == expected_out_b)
        jax.clear_caches()


class Test__find_compression_lst_6:
    def test_0(self) -> None:
        """
        Non-uniform refinement"""
        # p = 4
        q = 2

        print("test_3: q = ", q)

        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(add_to=root, root=root, q=q)
        for child in root.children:
            add_four_children(add_to=child, root=root, q=q)

        add_four_children(add_to=root.children[1].children[3], root=root, q=q)

        # Expect out_b to look like [False, True, True]
        # # and out_c to look like [False, False]
        out_b, out_c = _find_compression_list_6(
            root.children[1], root.children[2]
        )

        print("test_0: out_b: ", out_b)
        print("test_0: out_c: ", out_c)

        expected_out_b = jnp.array([False, True, True])
        expected_out_c = jnp.array([False, False])

        assert jnp.all(out_b == expected_out_b)
        assert jnp.all(out_c == expected_out_c)
        jax.clear_caches()


class Test_find_compression_lists_2D:
    def test_0(self) -> None:
        """
        Test that we can find the compression list for a simple case.
        """
        # Create a tree with 1 node
        q = 2
        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(root, root=root, q=q)
        for child in root.children:
            add_four_children(add_to=child, root=root, q=q)

        out_arrs = find_compression_lists_2D(
            root.children[0],
            root.children[1],
            root.children[2],
            root.children[3],
        )
        expected_arr = jnp.array([False, False])

        assert len(out_arrs) == 8

        for i in range(8):
            assert jnp.all(out_arrs[i] == expected_arr)
        jax.clear_caches()


if __name__ == "__main__":
    pytest.main()
