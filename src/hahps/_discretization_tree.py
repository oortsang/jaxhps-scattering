import jax
import jax.numpy as jnp
from typing import Tuple


class NodeData:
    def __init__(
        self,
    ):
        # Arrays set by the HPS computation in local solve & merge stages.
        self.T: jnp.array = None
        self.h: jnp.array = None
        self.S: jnp.array = None  # propagation operator
        self.g_tilde: jnp.array = (
            None  # g_tilde is the interface particular soln.
        )
        # self.L_2f1: jnp.array = None
        # self.L_1f2: jnp.array = None

        # These are only ever computed for leaf nodes. It's the local solution
        # operator and the local particular solution.
        self.Y: jnp.array = None
        self.v: jnp.array = None
        self.u: jnp.array = None
        self.g: jnp.array = None

        # This is set when performing adaptive meshing using an L2 criterion
        self.l2_nrm: float = 0.0

        # Keeping track of indexing
        # in 2D, self.n_i is the number of quadrature points along the boundary of
        # side i of the patch. i = 0, 1, 2, 3 corresponds to the bottom, right,
        # top, and left, or (S, E, N, W), sides of the patch, respectively.
        # in 3D, self.n_i is the number of quadrature points along the boundary of
        # each face of the voxel. See notes for ordering of the faces of voxels.
        self.n_0: int = None
        self.n_1: int = None
        self.n_2: int = None
        self.n_3: int = None
        self.n_4: int = None
        self.n_5: int = None


def flatten_nodedata(nodedata: NodeData) -> Tuple:
    c = None
    d = (
        nodedata.T,
        nodedata.h,
        nodedata.S,
        nodedata.g_tilde,
        # nodedata.L_2f1,
        # nodedata.L_1f2,
        nodedata.Y,
        nodedata.v,
        nodedata.u,
        nodedata.g,
        nodedata.l2_nrm,
    )
    return (c, d)


def unflatten_nodedata(aux_data: Tuple, children: Tuple) -> NodeData:
    z = object.__new__(NodeData)

    # Set the aux data
    z.T = aux_data[0]
    z.h = aux_data[1]
    z.S = aux_data[2]
    z.g_tilde = aux_data[3]
    z.Y = aux_data[4]
    z.v = aux_data[5]
    z.u = aux_data[6]
    z.g = aux_data[7]
    z.l2_nrm = aux_data[8]

    return z


jax.tree_util.register_pytree_node(
    NodeData, flatten_nodedata, unflatten_nodedata
)


class DiscretizationNode2D:
    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        depth: int = 0,
        data: NodeData = NodeData(),
        children: Tuple["DiscretizationNode2D"] = (),
    ):
        # Here is the auxiliary data
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.depth = depth
        self.data = data

        # Keeping track of indexing
        # in 2D, self.n_i is the number of quadrature points along the boundary of
        # side i of the patch. i = 0, 1, 2, 3 corresponds to the bottom, right,
        # top, and left, or (S, E, N, W), sides of the patch, respectively.
        self.n_0: int = None
        self.n_1: int = None
        self.n_2: int = None
        self.n_3: int = None

        # Here is the child pytrees
        self.children = children

    def __repr__(self) -> str:
        return "DiscretizationNode2D(xmin={}, xmax={}, ymin={}, ymax={}, depth={})".format(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.depth,
        )


def flatten_discretizationnode2D(node: DiscretizationNode2D) -> Tuple:
    c = (node.children,)
    d = (
        node.xmin,  # idx 0
        node.xmax,
        node.ymin,
        node.ymax,
        node.depth,
        node.data,  # idx 5
        node.n_0,
        node.n_1,
        node.n_2,
        node.n_3,
    )
    return (c, d)


def unflatten_discretizationnode2D(d: Tuple, c: Tuple) -> DiscretizationNode2D:
    z = DiscretizationNode2D(
        xmin=d[0],
        xmax=d[1],
        ymin=d[2],
        ymax=d[3],
        depth=d[4],
        data=d[5],
        children=c[0],
    )
    z.n_0 = d[6]
    z.n_1 = d[7]
    z.n_2 = d[8]
    z.n_3 = d[9]

    return z


jax.tree_util.register_pytree_node(
    DiscretizationNode2D,
    flatten_discretizationnode2D,
    unflatten_discretizationnode2D,
)


class DiscretizationNode3D:
    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
        depth: int = 0,
        data: NodeData = NodeData(),
        children: Tuple["DiscretizationNode3D"] = (),
    ):
        # Here is the auxiliary data
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.depth = depth
        self.data = data

        # Keeping track of indexing
        # in 3D, self.n_i is the number of quadrature points along the boundary of
        # each face of the voxel. See notes for ordering of the faces of voxels.
        self.n_0: int = None
        self.n_1: int = None
        self.n_2: int = None
        self.n_3: int = None
        self.n_4: int = None
        self.n_5: int = None

        # Here is the child pytrees
        self.children = children

    def __repr__(self):
        return "DiscretizationNode3D(xmin={}, xmax={}, ymin={}, ymax={}, zmin={}, zmax={}, depth={})".format(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.zmin,
            self.zmax,
            self.depth,
        )


def flatten_discretizationnode3D(node: DiscretizationNode3D) -> Tuple:
    c = (node.children,)
    d = (
        node.xmin,  # idx 0
        node.xmax,
        node.ymin,
        node.ymax,
        node.zmin,
        node.zmax,
        node.depth,
        node.data,  # idx 7
        node.n_0,
        node.n_1,
        node.n_2,
        node.n_3,
        node.n_4,
        node.n_5,
    )
    return (c, d)


def unflatten_discretizationnode3D(d: Tuple, c: Tuple) -> DiscretizationNode3D:
    z = DiscretizationNode3D(
        xmin=d[0],
        xmax=d[1],
        ymin=d[2],
        ymax=d[3],
        zmin=d[4],
        zmax=d[5],
        depth=d[6],
        data=d[7],
        children=c,
    )
    z.n_0 = d[8]
    z.n_1 = d[9]
    z.n_2 = d[10]
    z.n_3 = d[11]
    z.n_4 = d[12]
    z.n_5 = d[13]

    return z


@jax.jit
def get_discretization_node_area(
    node: DiscretizationNode2D | DiscretizationNode3D,
) -> float:
    # Check if it's a 3D node
    if isinstance(node, DiscretizationNode3D):
        return (
            (node.xmax - node.xmin)
            * (node.ymax - node.ymin)
            * (node.zmax - node.zmin)
        )
    else:
        return (node.xmax - node.xmin) * (node.ymax - node.ymin)


def get_all_leaves(
    node: DiscretizationNode2D | DiscretizationNode3D,
) -> Tuple[DiscretizationNode2D | DiscretizationNode3D]:
    if not len(node.children):
        return (node,)
    else:
        leaves = ()
        for child in node.children:
            leaves += get_all_leaves(child)
        return leaves
