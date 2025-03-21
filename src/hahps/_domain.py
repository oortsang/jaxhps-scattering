import jax

from ._discretization_tree import DiscretizationNode2D, DiscretizationNode3D


class Domain:
    def __init__(
        self,
        p: int,
        q: int,
        bounds: jax.Array,
        root: DiscretizationNode2D | DiscretizationNode3D,
        L: int | None = None,
    ):
        pass
