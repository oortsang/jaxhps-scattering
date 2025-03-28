import logging

import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

FIGSIZE = 5

FONTSIZE = 16

TICKSIZE = 15


def plot_field_for_wave_scattering_experiment(
    field: jnp.array,
    target_pts: jnp.array,
    use_bwr_cmap: bool = False,
    cmap_str: str = "parula",
    title: str = None,
    save_fp: str = None,
    ax: plt.Axes = None,
    figsize: float = FIGSIZE,
    ticksize: float = TICKSIZE,
) -> None:
    """
    Expect field to have shape (n,n) and target_pts to have shape (n, n, 2).
    """
    bool_create_ax = ax is None

    if bool_create_ax:
        fig, ax = plt.subplots(figsize=(figsize, figsize))

    extent = [
        target_pts[0, 0, 0],
        target_pts[-1, -1, 0],
        target_pts[-1, -1, 1],
        target_pts[0, 0, 1],
    ]
    logging.debug(
        "plot_field_for_wave_scattering_experiment: max_val: %s",
        jnp.max(field),
    )
    logging.debug(
        "plot_field_for_wave_scattering_experiment: min_val: %s",
        jnp.min(field),
    )

    if use_bwr_cmap:
        max_val = 3.65  # Max val of the fields we plot in the paper

        im = ax.imshow(
            field,
            cmap="bwr",
            vmin=-max_val,
            vmax=max_val,
            extent=extent,
        )
    else:
        im = ax.imshow(
            field,
            cmap=cmap_str,
            extent=extent,
            vmin=-3.65,  # Min val of the fields we plot in the paper
            vmax=3.65,
        )

    # Set ticks to [-1, 0, 1]
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="both", which="major", labelsize=ticksize)

    # Sizing brought to you by https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    make_scaled_colorbar(im, ax, fontsize=ticksize)

    if title is not None:
        ax.set_title(title)

    if bool_create_ax:
        if save_fp is not None:
            fig.tight_layout()
            plt.savefig(save_fp, bbox_inches="tight")


CMAP_PAD = 0.1


def make_scaled_colorbar(im, ax, fontsize: float = None) -> None:
    """
    Make a colorbar that is the same size as the plot.

    Found this on StackExchange

    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=CMAP_PAD)
    cbar = plt.colorbar(im, cax=cax)
    if fontsize is not None:
        cbar.ax.tick_params(labelsize=fontsize)
