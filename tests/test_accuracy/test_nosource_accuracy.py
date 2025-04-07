from hahps._discretization_tree import DiscretizationNode2D
from hahps._domain import Domain
from .cases import (
    XMIN,
    XMAX,
    YMIN,
    YMAX,
)

ATOL = 1e-12
RTOL = 0.0

P = 6
Q = 4
ROOT_ITI = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
DOMAIN_ITI = Domain(p=P, q=Q, root=ROOT_ITI, L=1)
