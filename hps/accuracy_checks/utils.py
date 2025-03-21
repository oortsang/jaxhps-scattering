from typing import Callable, Tuple
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.interpolate import LinearNDInterpolator

from hps.src.solver_obj import (
    create_solver_obj_2D,
)
from hps.src.quadrature.trees import Node
from hps.src.utils import meshgrid_to_lst_of_pts


def setup_tree_quad_merge(p: int, eta: float = None, use_ItI: bool = False):
    # Set up the domain
    south = -jnp.pi / 2
    north = jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2
    root = Node(
        xmin=west, xmax=east, ymin=south, ymax=north, depth=0, zmin=None, zmax=None
    )
    # corners = [(west, south), (east, south), (east, north), (west, north)]
    q = p - 2

    t = create_solver_obj_2D(
        p=p, q=q, root=root, uniform_levels=1, use_ItI=use_ItI, eta=eta
    )
    return t


def _distance_around_boundary(node) -> jnp.ndarray:
    south = node.quad_obj.corners[0][1]
    north = node.quad_obj.corners[2][1]
    east = node.quad_obj.corners[1][0]
    west = node.quad_obj.corners[0][0]
    ns_len = north - south
    ew_len = east - west

    distance_around_boundary = jnp.concatenate(
        [
            jnp.abs(node.quad_obj.boundary_points_dd["S"][:, 0] - west),
            ew_len + jnp.abs(node.quad_obj.boundary_points_dd["E"][:, 1] - south),
            ns_len
            + ew_len
            + jnp.abs(node.quad_obj.boundary_points_dd["N"][:, 0] - east),
            ns_len
            + 2 * ew_len
            + jnp.abs(node.quad_obj.boundary_points_dd["W"][:, 1] - north),
        ]
    )
    return distance_around_boundary


def _distance_around_boundary_nonode(
    corners: jnp.ndarray, boundary_points: jnp.ndarray
):
    south = corners[0][1]
    north = corners[2][1]
    east = corners[1][0]
    west = corners[0][0]
    ns_len = north - south
    ew_len = east - west

    n_per_side = boundary_points.shape[0] // 4

    distance_around_boundary = jnp.concatenate(
        [
            jnp.abs(boundary_points[:n_per_side, 0] - west),
            ew_len + jnp.abs(boundary_points[n_per_side : 2 * n_per_side, 1] - south),
            ns_len
            + ew_len
            + jnp.abs(boundary_points[2 * n_per_side : 3 * n_per_side, 0] - east),
            ns_len + 2 * ew_len + jnp.abs(boundary_points[3 * n_per_side :, 1] - north),
        ]
    )
    return distance_around_boundary


def exact_computed_neumann_data(
    node,
    f: Callable[[jnp.ndarray], jnp.ndarray],
    dfdx: Callable[[jnp.ndarray], jnp.ndarray],
    dfdy: Callable[[jnp.ndarray], jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gauss_boundary_pts = jnp.concatenate(
        [
            node.quad_obj.boundary_points_dd["S"],
            node.quad_obj.boundary_points_dd["E"],
            node.quad_obj.boundary_points_dd["N"],
            node.quad_obj.boundary_points_dd["W"],
        ]
    )

    boundary_data = f(gauss_boundary_pts)

    neumann_data = node.map_DtN @ boundary_data

    expected_neumann_data = jnp.concatenate(
        [
            dfdy(node.quad_obj.boundary_points_dd["S"]),
            dfdx(node.quad_obj.boundary_points_dd["E"]),
            dfdy(node.quad_obj.boundary_points_dd["N"]),
            dfdx(node.quad_obj.boundary_points_dd["W"]),
        ]
    )

    return neumann_data, expected_neumann_data, _distance_around_boundary(node)


def exact_computed_dirichlet_data(
    node, f: Callable[[jnp.ndarray], jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gauss_boundary_pts = jnp.concatenate(
        [
            node.quad_obj.boundary_points_dd["S"],
            node.quad_obj.boundary_points_dd["E"],
            node.quad_obj.boundary_points_dd["N"],
            node.quad_obj.boundary_points_dd["W"],
        ]
    )

    boundary_data = f(gauss_boundary_pts)

    dirichlet_data = node.boundary_data_vec

    return dirichlet_data, boundary_data, _distance_around_boundary(node)


def plot_soln_from_cheby_nodes(
    cheby_nodes: jnp.ndarray,
    corners: jnp.ndarray,
    expected_soln: jnp.ndarray,
    computed_soln: jnp.ndarray,
    t: str = "Part. Soln",
) -> None:
    """Loop through the leaf nodes of the tree, and plot a dot at each Chebyshev node.
    The dot should be colored by the solution value in leaf_node.interior_soln.
    """

    print("plot_soln_from_cheby_nodes: cheby_nodes.shape", cheby_nodes.shape)
    print("plot_soln_from_cheby_nodes: expected_soln.shape", expected_soln.shape)
    print("plot_soln_from_cheby_nodes: computed_soln.shape", computed_soln.shape)

    if corners is None:
        xmin = cheby_nodes[:, 0].min()
        xmax = cheby_nodes[:, 0].max()
        ymin = cheby_nodes[:, 1].min()
        ymax = cheby_nodes[:, 1].max()
        corners = jnp.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    print("plot_soln_from_cheby_nodes: corners", corners)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15, 5)
    ax[0].set_title("Expected " + t)
    ax[1].set_title("Computed " + t)
    ax[2].set_title("Expected - Computed")

    # Get a list of regularly-spaced points in the domain
    n_X = 100
    x = jnp.linspace(corners[0][0], corners[1][0], n_X)
    y = jnp.linspace(corners[0][1], corners[2][1], n_X)
    X, Y = jnp.meshgrid(x, jnp.flipud(y))
    lst_of_pts = meshgrid_to_lst_of_pts(X, Y)

    # Create an interpolator for the expected solution
    interp_expected = LinearNDInterpolator(cheby_nodes, expected_soln)
    expected_soln = interp_expected(lst_of_pts).reshape(n_X, n_X)

    # Create an interpolator for the computed solution
    interp_computed = LinearNDInterpolator(cheby_nodes, computed_soln)
    computed_soln = interp_computed(lst_of_pts).reshape(n_X, n_X)

    # Get all solution values to set color limits
    all_soln_vals = np.concatenate([expected_soln, computed_soln])

    # Create a colormap
    cmap = cm.plasma
    # Find the min among all non-NaN values
    min_val = np.nanmin(all_soln_vals)
    max_val = np.nanmax(all_soln_vals)
    norm = colors.Normalize(vmin=min_val, vmax=max_val)
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])

    # Loop over the list of leaf nodes, plotting the interior + boundary points in the color from the colormap.
    # for j, pt in enumerate(cheby_nodes):
    #     color_0 = cmap(norm(expected_soln[j]))
    #     ax[0].plot(pt[0], pt[1], "o", color=color_0)

    #     color_1 = cmap(norm(computed_soln[j]))
    #     ax[1].plot(pt[0], pt[1], "o", color=color_1)
    extent = [corners[0][0], corners[1][0], corners[0][1], corners[2][1]]

    # Plot the expected and computed solutions
    im_0 = ax[0].imshow(
        expected_soln,
        cmap=cmap,
        norm=norm,
        extent=extent,
    )
    im_1 = ax[1].imshow(
        computed_soln,
        cmap=cmap,
        norm=norm,
        extent=extent,
    )
    im_2 = ax[2].imshow(
        expected_soln - computed_soln,
        cmap="bwr",
        extent=extent,
    )

    # ax.legend()
    plt.colorbar(im_0, ax=ax[0])
    plt.colorbar(im_1, ax=ax[1])
    plt.colorbar(im_2, ax=ax[2])
    plt.show()
