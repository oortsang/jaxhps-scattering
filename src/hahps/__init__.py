from ._grid_creation_2D import (
    compute_boundary_Gauss_points_adaptive_2D,
    compute_boundary_Gauss_points_uniform_2D,
    compute_interior_Chebyshev_points_adaptive_2D,
    compute_interior_Chebyshev_points_uniform_2D,
)
from ._grid_creation_3D import (
    compute_boundary_Gauss_points_adaptive_3D,
    compute_boundary_Gauss_points_uniform_3D,
    compute_interior_Chebyshev_points_adaptive_3D,
    compute_interior_Chebyshev_points_uniform_3D,
)
from ._domain import Domain
from ._discretization_tree import DiscretizationNode2D, DiscretizationNode3D

# These will appear in the module's top-level namespace when imported
__all__ = [
    "Domain",
    "DiscretizationNode2D",
    "DiscretizationNode3D",
    "compute_boundary_Gauss_points_uniform_2D",
    "compute_boundary_Gauss_points_adaptive_2D",
    "compute_interior_Chebyshev_points_uniform_2D",
    "compute_interior_Chebyshev_points_adaptive_2D",
    "compute_boundary_Gauss_points_uniform_3D",
    "compute_boundary_Gauss_points_adaptive_3D",
    "compute_interior_Chebyshev_points_uniform_3D",
    "compute_interior_Chebyshev_points_adaptive_3D",
]
