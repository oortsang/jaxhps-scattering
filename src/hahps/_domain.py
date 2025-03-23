from ._discretization_tree import DiscretizationNode2D, DiscretizationNode3D
from ._grid_creation_2D import (
    compute_interior_Chebyshev_points_uniform_2D,
    compute_interior_Chebyshev_points_adaptive_2D,
    compute_boundary_Gauss_points_uniform_2D,
    compute_boundary_Gauss_points_adaptive_2D,
)


class Domain:
    def __init__(
        self,
        p: int,
        q: int,
        root: DiscretizationNode2D | DiscretizationNode3D,
        L: int | None = None,
    ):
        self.p = p
        self.q = q
        self.root = root
        self.L = L

        if self.L is not None:
            # Depending on whether root is a DiscretizationNode2D or DiscretizationNode3D, we compute the grid points accordingly
            if isinstance(root, DiscretizationNode2D):
                self.interior_points = (
                    compute_interior_Chebyshev_points_uniform_2D(root, L, p)
                )
                self.boundary_points = (
                    compute_boundary_Gauss_points_uniform_2D(root, L, q)
                )
            else:
                raise NotImplementedError("3D domain not yet implemented.")

        else:
            # If L is None, we're using an adaptive discretization
            if isinstance(root, DiscretizationNode2D):
                self.interior_points = (
                    compute_interior_Chebyshev_points_adaptive_2D(root, p)
                )
                self.boundary_points = (
                    compute_boundary_Gauss_points_adaptive_2D(root, q)
                )
            else:
                raise NotImplementedError("3D domain not yet implemented.")
