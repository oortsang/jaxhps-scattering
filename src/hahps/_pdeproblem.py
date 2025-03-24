import jax

from ._domain import Domain
from ._discretization_tree import DiscretizationNode3D
from ._precompute_operators_2D import (
    precompute_diff_operators_2D,
    precompute_P_2D_DtN,
    precompute_Q_2D_DtN,
    precompute_P_2D_ItI,
    precompute_N_tilde_matrix_2D,
    precompute_N_matrix_2D,
    precompute_G_2D_ItI,
    precompute_QH_2D_ItI,
)
from ._precompute_operators_3D import (
    precompute_diff_operators_3D,
    precompute_P_3D_DtN,
    precompute_Q_3D_DtN,
)


class PDEProblem:
    def __init__(
        self,
        domain: Domain,
        source: jax.Array,
        D_xx_coefficients: jax.Array = None,
        D_xy_coefficients: jax.Array = None,
        D_xz_coefficients: jax.Array = None,
        D_yy_coefficients: jax.Array = None,
        D_yz_coefficients: jax.Array = None,
        D_zz_coefficients: jax.Array = None,
        D_x_coefficients: jax.Array = None,
        D_y_coefficients: jax.Array = None,
        D_z_coefficients: jax.Array = None,
        I_coefficients: jax.Array = None,
        use_ItI: bool = False,
        eta: float = None,
    ):
        self.domain = domain

        # Input validation
        # 2D problems shouldn't specify D_z_coefficients
        if isinstance(domain.root, DiscretizationNode3D):
            bool_2D = False
            s = "z coefficients can not be set for 2D problems."
            if D_xz_coefficients is not None:
                raise ValueError(s)
            if D_yz_coefficients is not None:
                raise ValueError(s)
            if D_zz_coefficients is not None:
                raise ValueError(s)
            if D_z_coefficients is not None:
                raise ValueError(s)

            # 3D code doesn't support ItI merges
            if use_ItI:
                raise NotImplementedError(
                    "ItI merges are not supported for 3D problems."
                )
        else:
            bool_2D = True

        # If ItI is being used, eta must be specified
        if use_ItI and eta is None:
            raise ValueError("eta must be specified when using ItI merges.")

        # If ItI is being used, it must be a uniform 2D problem
        if use_ItI and not domain.bool_uniform:
            raise ValueError(
                "ItI merges are only supported for uniform 2D problems."
            )

        # Check input shapes are OK
        check_lst = [
            (D_xx_coefficients, "D_xx_coefficients"),
            (D_xy_coefficients, "D_xy_coefficients"),
            (D_xz_coefficients, "D_xz_coefficients"),
            (D_yy_coefficients, "D_yy_coefficients"),
            (D_yz_coefficients, "D_yz_coefficients"),
            (D_zz_coefficients, "D_zz_coefficients"),
            (D_x_coefficients, "D_x_coefficients"),
            (D_y_coefficients, "D_y_coefficients"),
            (D_z_coefficients, "D_z_coefficients"),
            (I_coefficients, "I_coefficients"),
        ]
        if not use_ItI:
            # Other parts of the ItI code use source terms that have shape [n_leaves, p**2, n_src]
            check_lst.append((source, "source"))
        for arr, name in check_lst:
            if arr is not None:
                if arr.shape != domain.interior_points[..., 0].shape:
                    raise ValueError(
                        f"{name} has shape {arr.shape} but should have shape {domain.interior_points[..., 0].shape} to match the Domain's interior points."
                    )

        # Store coefficients
        self.D_xx_coefficients = D_xx_coefficients
        self.D_xy_coefficients = D_xy_coefficients
        self.D_xz_coefficients = D_xz_coefficients
        self.D_yy_coefficients = D_yy_coefficients
        self.D_yz_coefficients = D_yz_coefficients
        self.D_zz_coefficients = D_zz_coefficients
        self.D_x_coefficients = D_x_coefficients
        self.D_y_coefficients = D_y_coefficients
        self.D_z_coefficients = D_z_coefficients
        self.I_coefficients = I_coefficients
        self.source = source
        self.use_ItI = use_ItI
        self.eta = eta

        if domain.bool_uniform:
            # In this version of the code, we know the side len of each leaf is the same, so we can scale the diff
            # operators ahead of time.
            # half_side_len = (
            #     (domain.root.xmax - domain.root.xmin) / (2**domain.L) / 2
            # )

            half_side_len = (domain.root.xmax - domain.root.xmin) / (
                2 ** (domain.L + 1)
            )
        else:
            # In this version of the code, the diff operators are scaled separately by the sidelen of each leaf.
            half_side_len = 1.0

        # Pre-compute spectral differentiation and interpolation matrices
        if bool_2D:
            # Differentiation operators
            self.D_x, self.D_y, self.D_xx, self.D_yy, self.D_xy = (
                precompute_diff_operators_2D(domain.p, half_side_len)
            )
            if not use_ItI:
                # Interpolation / Differentiation matrices for DtN merges
                self.P = precompute_P_2D_DtN(domain.p, domain.q)
                self.Q = precompute_Q_2D_DtN(
                    domain.p, domain.q, self.D_x, self.D_y
                )
            else:
                # Interpolation / Differentiation matrices for ItI merges
                self.P = precompute_P_2D_ItI(domain.p, domain.q)

                # In the local solve stage code, F is what the paper calls G, and
                # G is what the paper calls H. The notation in this part is following
                # the paper's notation.
                N_tilde = precompute_N_tilde_matrix_2D(
                    self.D_x, self.D_y, domain.p
                )
                self.G = precompute_G_2D_ItI(N_tilde, self.eta)
                # QH always appear together so we can precompute their product.
                N = precompute_N_matrix_2D(self.D_x, self.D_y, domain.p)
                self.QH = precompute_QH_2D_ItI(N, domain.p, domain.q, self.eta)
        else:
            # Differentiation operators
            (
                self.D_x,
                self.D_y,
                self.D_z,
                self.D_xx,
                self.D_yy,
                self.D_zz,
                self.D_xy,
                self.D_xz,
                self.D_yz,
            ) = precompute_diff_operators_3D(
                p=domain.p, half_side_len=half_side_len
            )
            self.P = precompute_P_3D_DtN(domain.p, domain.q)
            self.Q = precompute_Q_3D_DtN(
                domain.p, domain.q, self.D_x, self.D_y, self.D_z
            )
