import matplotlib.pyplot as plt
from matplotlib import cm, colors
import jax.numpy as jnp
from .quadrature import meshgrid_to_lst_of_pts
from scipy.interpolate import LinearNDInterpolator
import numpy as np


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
    print(
        "plot_soln_from_cheby_nodes: expected_soln.shape", expected_soln.shape
    )
    print(
        "plot_soln_from_cheby_nodes: computed_soln.shape", computed_soln.shape
    )

    if corners is None:
        xmin = cheby_nodes[:, 0].min()
        xmax = cheby_nodes[:, 0].max()
        ymin = cheby_nodes[:, 1].min()
        ymax = cheby_nodes[:, 1].max()
        corners = jnp.array(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        )

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
