import pytest
import logging
import numpy as np
import jax.numpy as jnp
import jax
from hps.src.solver_obj import (
    create_solver_obj_2D,
    create_solver_obj_3D,
    SolverObj,
    get_bdry_data_evals_lst_2D,
    get_bdry_data_evals_lst_3D,
)
from hps.src.quadrature.trees import (
    Node,
    get_all_leaves,
    add_four_children,
    get_nodes_at_level,
    add_eight_children,
)

from hps.src.up_down_passes import (
    build_stage,
    down_pass,
    # merge_single_level,
    local_solve_stage,
    fused_pde_solve_2D,
    fused_pde_solve_2D_ItI,
    baseline_pde_solve_2D,
)
from hps.src.config import DEVICE_ARR


class Test_local_solve_stage:
    def test_0(self) -> None:
        """Tests the local_solve_stage function returns without error when using DtN maps. 2D case w/ uniform refinement."""
        p = 16
        q = 14
        l = 2

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1, depth=0, zmin=None, zmax=None)
        t = create_solver_obj_2D(p, q, root, uniform_levels=l)
        print("test_0: l = ", l)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])

        local_solve_stage(
            t, source_term=source_term, D_xx_coeffs=d_xx_coeffs, D_yy_coeffs=d_yy_coeffs
        )

        n_leaves = 4**l

        assert len(t.interior_node_DtN_maps) == 1
        assert t.interior_node_DtN_maps[0].shape == (n_leaves, 4 * q, 4 * q)
        assert t.leaf_node_v_vecs.shape == (n_leaves, p**2)
        assert t.leaf_node_v_prime_vecs.shape == (n_leaves, 4 * q)

    def test_1(self) -> None:
        """Tests the local_solve_stage function returns without error when using DtN maps. 2D case w/ non-uniform refinement."""
        p = 16
        q = 14
        n_leaves = 7

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1, depth=0, zmin=None, zmax=None)
        add_four_children(root, root=root, q=q)
        add_four_children(root.children[0], root=root, q=q)
        t = create_solver_obj_2D(p, q, root)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])

        local_solve_stage(
            t, source_term=source_term, D_xx_coeffs=d_xx_coeffs, D_yy_coeffs=d_yy_coeffs
        )

        # assert len(t.interior_node_DtN_maps) == 1
        # assert t.leaf_Y_maps.shape == (n_leaves, p**2, 4 * q)
        # assert t.leaf_node_v_vecs.shape == (n_leaves, p**2)
        # assert t.interior_node_DtN_maps[0].shape == (n_leaves, 4 * q, 4 * q)
        # assert t.leaf_node_v_prime_vecs.shape == (n_leaves, 4 * q)

        # Check the tree object contains the DtN arrays
        leaves = get_all_leaves(t.root)
        for leaf in leaves:
            assert leaf.DtN is not None
            assert leaf.v_prime is not None

    def test_2(self) -> None:
        """Tests the local_solve_stage function returns without error when using DtN maps. 3D uniform case."""
        p = 6
        q = 4
        l = 2

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1, depth=0)
        t = create_solver_obj_3D(p, q, root, uniform_levels=l)

        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_zz_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])
        local_solve_stage(
            t,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            D_zz_coeffs=d_zz_coeffs,
        )
        n_leaves = 8**l
        n_merge_octs = 8 ** (l - 1)
        assert len(t.interior_node_DtN_maps) == 1
        assert t.interior_node_DtN_maps[0].shape == (
            n_leaves,
            6 * q**2,
            6 * q**2,
        )
        assert t.leaf_Y_maps.shape == (n_leaves, p**3, 6 * q**2)
        assert t.leaf_node_v_vecs.shape == (n_leaves, p**3)
        assert t.leaf_node_v_prime_vecs.shape == (n_leaves, 6 * q**2)

    def test_3(self, caplog) -> None:
        """Tests the local_solve_stage function returns without error when using ItI maps. 2D case."""
        caplog.set_level(logging.DEBUG)
        p = 16
        q = 14
        l = 2
        n_src = 2
        eta = 4.0

        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]
        root = Node(
            xmin=0,
            xmax=1,
            ymin=0,
            ymax=1,
        )

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, use_ItI=True, eta=eta)
        print("test_3: l = ", l)

        print("test_3: q = ", q)
        print("test_3: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.concatenate(
            [jnp.zeros_like(t.leaf_cheby_points[..., 0, None]) for _ in range(n_src)],
            axis=-1,
        )
        print("test_2: source_term.shape = ", source_term.shape)

        local_solve_stage(
            t, source_term=source_term, D_xx_coeffs=d_xx_coeffs, D_yy_coeffs=d_yy_coeffs
        )

        n_leaves = 4**l
        n_merge_quads = 4 ** (l - 1)

        assert len(t.interior_node_R_maps) == 1
        assert t.interior_node_R_maps[0].shape == (n_merge_quads, 4, 4 * q, 4 * q)
        assert t.leaf_Y_maps.shape == (n_leaves, p**2, 4 * q)
        assert t.leaf_node_v_vecs.shape == (n_leaves, p**2)
        assert t.leaf_node_h_vecs.shape == (n_merge_quads, 4, 4 * q)


class Test_build_stage:
    def test_0(self) -> None:
        """Tests the build_stage function returns without error when using DtN maps. 2D case."""
        p = 16
        q = 14
        l = 2

        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1, depth=0, zmin=None, zmax=None)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)
        print("test_0: l = ", l)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])

        local_solve_stage(
            t, source_term=source_term, D_xx_coeffs=d_xx_coeffs, D_yy_coeffs=d_yy_coeffs
        )
        build_stage(t)

        # assert t.interior_node_DtN_maps[-1].shape == (
        #     t.root_boundary_points.shape[0],
        #     t.root_boundary_points.shape[0],
        # )
        assert len(t.interior_node_S_maps) == l
        # l = 2 so we expect the root node to have 8q boundary points
        n_interface_pts = t.root_boundary_points.shape[0] // 2
        assert t.interior_node_S_maps[-1].shape == (
            1,
            n_interface_pts,
            t.root_boundary_points.shape[0],
        )

    def test_1(self) -> None:
        """Tests the build_stage function returns without error when using DtN maps. 3D uniform case."""

        p = 4
        q = 2
        l = 3

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1, depth=0, zmin=0, zmax=1.0)

        t = create_solver_obj_3D(p, q, root, uniform_levels=l)

        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_zz_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])
        local_solve_stage(
            t,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            D_zz_coeffs=d_zz_coeffs,
        )
        build_stage(t)

        # Check that the solver obj has the outputs of the build stage saved.
        assert t.interior_node_DtN_maps is not None
        assert t.interior_node_S_maps is not None
        assert t.interior_node_v_int_vecs is not None

    def test_2(self, caplog) -> None:
        """Tests the build_stage function returns without error when using DtN maps. 3D non-uniform case."""
        caplog.set_level(logging.DEBUG)
        p = 4
        q = 2
        l = 3

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1, depth=0, zmin=0, zmax=1.0)
        add_eight_children(root, root, q)
        add_eight_children(root.children[0], root, q)

        t = create_solver_obj_3D(p, q, root)

        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_zz_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])
        local_solve_stage(
            t,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            D_zz_coeffs=d_zz_coeffs,
        )
        build_stage(t)

        # Check that the non-leaf nodes have the correct data set.
        for level in [0, 1, 2]:
            for node in get_nodes_at_level(root, level):
                assert node.DtN is not None
                assert node.v_prime is not None
                if len(node.children):
                    assert node.S is not None

    def test_3(self) -> None:
        """Tests the build_stage function returns without error when using ItI maps. 2D case."""
        p = 16
        q = 14
        l = 2
        eta = 4.0
        n_src = 2

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, eta=eta, use_ItI=True)
        print("test_2: l = ", l)

        print("test_2: q = ", q)
        print("test_2: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.zeros_like(t.leaf_cheby_points)

        local_solve_stage(
            t, source_term=source_term, D_xx_coeffs=d_xx_coeffs, D_yy_coeffs=d_yy_coeffs
        )
        build_stage(t)

        assert len(t.interior_node_R_maps) == l
        assert t.interior_node_R_maps[-1].shape == (
            t.root_boundary_points.shape[0],
            t.root_boundary_points.shape[0],
        )
        assert len(t.interior_node_S_maps) == l
        # l = 2 so we expect the root node to have 8q boundary points
        n_interface_pts = t.root_boundary_points.shape[0]
        assert t.interior_node_S_maps[-1].shape == (
            1,
            n_interface_pts,
            t.root_boundary_points.shape[0],
        )


class Test_down_pass:
    def test_0(self) -> None:
        """Tests the down_pass function returns without error when using DtN maps. 2D case with uniform tree."""
        p = 5
        q = 3
        l = 2

        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1, depth=0, zmin=None, zmax=None)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)
        print("test_0: l = ", l)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])

        d_xx_coeffs = jax.device_put(d_xx_coeffs, DEVICE_ARR[0])
        d_yy_coeffs = jax.device_put(d_yy_coeffs, DEVICE_ARR[0])
        source_term = jax.device_put(source_term, DEVICE_ARR[0])

        local_solve_stage(
            t, source_term=source_term, D_xx_coeffs=d_xx_coeffs, D_yy_coeffs=d_yy_coeffs
        )
        build_stage(t)

        print("test_0: Boundary points shape: ", t.root_boundary_points.shape)
        f = lambda x: jnp.ones_like(x[..., 0])

        boundary_data_lst = get_bdry_data_evals_lst_2D(t, f)

        down_pass(t, boundary_data_lst)

        assert t.interior_solns.shape == (4**l, p**2)

    def test_1(self, caplog) -> None:
        """Tests the down_pass function returns without error when using DtN maps. 3D uniform case."""
        caplog.set_level(logging.DEBUG)
        p = 4
        q = 2
        l = 2

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1, depth=0, zmin=0, zmax=1.0)
        t = create_solver_obj_3D(p=p, q=q, root=root, uniform_levels=l)

        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_zz_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])
        local_solve_stage(
            t,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            D_zz_coeffs=d_zz_coeffs,
        )
        build_stage(t)
        f = lambda x: jnp.ones_like(x[..., 0])

        boundary_data_lst = get_bdry_data_evals_lst_3D(t, f)
        down_pass(t, boundary_data_lst)
        assert t.interior_solns.shape == (8**l, p**3)

    def test_2(self) -> None:
        """Tests the down_pass function returns without error when using ItI maps. 2D case."""
        p = 16
        q = 14
        l = 2
        eta = 4.0

        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]
        root = Node(
            xmin=0,
            xmax=1,
            ymin=0,
            ymax=1,
        )

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, eta=eta, use_ItI=True)
        print("test_2: l = ", l)

        print("test_2: q = ", q)
        print("test_2: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.zeros_like(t.leaf_cheby_points)

        local_solve_stage(
            t, source_term=source_term, D_xx_coeffs=d_xx_coeffs, D_yy_coeffs=d_yy_coeffs
        )
        build_stage(t)

        boundary_data_lst = get_bdry_data_evals_lst_2D(
            t, lambda x: jnp.zeros_like(x[..., 0])
        )

        down_pass(t, boundary_data_lst)

        assert t.interior_solns.shape == (4**l, p**2)


class Test_fused_pde_solve_2D:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        p = 7
        q = 5
        l = 4

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1)
        t = create_solver_obj_2D(p, q, root, uniform_levels=l)

        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        source = jnp.array(np.random.normal(size=(4**l, p**2)))
        bdry_data = jnp.array(np.random.normal(size=t.root_boundary_points.shape[0]))
        fused_pde_solve_2D(
            t=t,
            boundary_data=bdry_data,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )

        assert t.interior_solns.shape == (4**l, p**2)


class Test_baseline_pde_solve_2D:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        p = 7
        q = 5
        l = 4

        root = Node(xmin=0, xmax=1, ymin=0, ymax=1)
        t = create_solver_obj_2D(p, q, root, uniform_levels=l)

        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        source = jnp.array(np.random.normal(size=(4**l, p**2)))
        bdry_data = jnp.array(np.random.normal(size=t.root_boundary_points.shape[0]))
        baseline_pde_solve_2D(
            t=t,
            boundary_data=bdry_data,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )

        assert t.interior_solns.shape == (4**l, p**2)


class Test_fused_pde_solve_2D_ItI:
    def test_0(self) -> None:
        p = 7
        q = 5
        l = 4
        eta = 3.0
        root = Node(xmin=0, xmax=1, ymin=0, ymax=1)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, eta=eta, use_ItI=True)

        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        source = jnp.array(np.random.normal(size=(4**l, p**2)))
        bdry_data = jnp.array(np.random.normal(size=t.root_boundary_points.shape[0]))
        fused_pde_solve_2D_ItI(
            t=t,
            boundary_data=bdry_data,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )

        assert t.interior_solns.shape == (4**l, p**2)

    def test_1(self) -> None:
        p = 4
        q = 2
        l = 2
        eta = 4.0
        root = Node(xmin=0, xmax=1, ymin=0, ymax=1)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, eta=eta, use_ItI=True)

        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        source = jnp.array(np.random.normal(size=(4**l, p**2)))
        bdry_data = jnp.array(np.random.normal(size=t.root_boundary_points.shape[0]))
        fused_pde_solve_2D_ItI(
            t=t,
            boundary_data=bdry_data,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )

        assert t.interior_solns.shape == (4**l, p**2)


if __name__ == "__main__":
    pytest.main()
