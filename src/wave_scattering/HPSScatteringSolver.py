from typing import Tuple
import jax.numpy as jnp
import jax
import argparse
import logging
import os
import h5py
import matplotlib.pyplot as plt
# from jaxhps import Domain, PDEProblem, DiscretizationNode2D, build_solver
from src.jaxhps import Domain, PDEProblem, DiscretizationNode2D, build_solver

from src.wave_scattering.ScatteringProblem import ScatteringProblem
from src.wave_scattering.gen_SD_exterior import (
    gen_D_exterior,
    gen_S_exterior,
)
from src.wave_scattering.interp_utils import (
    prep_grids_unif_2d,
    prep_grids_cheb_2d,
)
from src.wave_scattering.interp_ops import (
    QuadtreeToUniform,
    UniformToQuadtree,
)
from src.wave_scattering.exterior_solver import (
    forward_model_exterior,
)


class HPSScatteringSolver():
    def __init__(
        self,
        L: int, p: int, N_x: int,
        k: float,
        S_int, D_int,
        S_ext=None, D_ext=None,
        N_r=None, N_s=None,
        unif_domain_bounds=(-0.5, 0.5, -0.5, 0.5),
        quad_domain_bounds=(-0.5, 0.5, -0.5, 0.5),
        receiver_radius=100,
        UtQ=None,
        source_dirs=None,
        receiver_dirs=None,
        # QtU=None,
    ):
        """A wrapper for the HPS solver for the wave scattering problem (on the receiver ring)

        Parameters:
            L (int): number of levels in the HPS quadtree
            p (int): polynomial order of the chebyshev grids on the leaf-level of HPS grids
            N_x (int): number of points on the uniform grid (e.g., 192) corresponding to the
                domain given by unif_domain_bounds
            k (float): angular wavenumber used for the source waves
                i.e., already contains the factor of 2pi
            S_int (jax.Array): interior single-layer potential scattering matrix
                pre-computed in Matlab with ChunkIE and chebfun
            D_int (jax.Array): interior double-layer potential scattering matrix
                pre-computed in Matlab with ChunkIE and chebfun
            S_ext (jax.Array): exterior single-layer potential scattering matrix
                can be precomputed with gen_S_exterior or computed in this initialization phase
            D_ext (jax.Array): exterior double-layer potential scattering matrix
                can be precomputed with gen_D_exterior or computed in this initialization phase
            N_r (int): number of receivers
            N_s (int): number of sources
            unif_domain_bounds (jax.Array): the bounds of the uniformly-distributed grid
                meant for use as the space where the scattering potential lives
                Note: format is [xmin, xmax, ymin, max]
            quad_domain_bounds (jax.Array): the bounds of the HPS quadtree's computational grid
                meant for use computing the scattered wave u_scat
                Expanding the computational domain can prevent the solution from
                degrading when it would normally leave the computational domain
                Note: format is [xmin, xmax, ymin, max]
            receiver_radius (float): radius where the receiver locations are distributed
            UtQ (UniformToQuadtree): an interpolation object that helps map from the uniform grid
                to the quadtree discretization (and node ordering)
                Will be computed if not supplied
            source_dirs (jax.Array): the angles corresponding to source waves
                Note: should be set to np.pi/2-np.linspace(0,2*np.pi,N_s,endpoint=False)
                in order to match the Lippmann-Schwinger solver configuration
            receiver_dirs (jax.Array): the angles corresponding to receiver nodes on the receiver radius
                Note: should be set to np.pi/2-np.linspace(0,2*np.pi,N_s,endpoint=False)
                in order to match the Lippmann-Schwinger solver configuration

        Caution: the source_dirs and receiver_dirs should be different from the usual setup
        To agree with the LS solver with source_dirs=np.linspace(0,2*np.pi,N_s,endpoint=False)
        We instead need to take source_dirs=np.pi/2-np.linspace(0,2*np.pi,N_s,endpoint=False)
        I believe there is some sort of discrepancy in axis ordering that causes this.
        """
        self.L = L
        self.p = p
        self.N_x = N_x
        self.n_per_leaf = N_x // 2**L
        self.k = k
        self.receiver_radius = receiver_radius

        self.leaf_bounds = (-1., 1., -1., 1.)
        self.unif_domain_bounds = jnp.array(unif_domain_bounds)
        self.quad_domain_bounds = jnp.array(quad_domain_bounds)
        self.root = DiscretizationNode2D(*self.quad_domain_bounds)
        self.domain = Domain(p=p, q=p - 2, root=self.root, L=L)

        # Calculate the grids in case they'll be useful later...
        self.leaf_unif_x, self.leaf_unif_y, self.leaf_unif_xy = prep_grids_unif_2d(
            0, self.n_per_leaf, domain_bounds=self.leaf_bounds, rel_offset=0,
        )
        self.leaf_cheb_x, self.leaf_cheb_y, self.leaf_cheb_xy = prep_grids_cheb_2d(
            0, self.p, domain_bounds=self.leaf_bounds
        )
        self.tree_unif_x, self.tree_unif_y, self.tree_unif_xy = prep_grids_unif_2d(
            0, self.N_x, domain_bounds=self.unif_domain_bounds, rel_offset=0,
        )
        self.tree_cheb_x, self.tree_cheb_y, self.tree_cheb_xy = prep_grids_cheb_2d(
            self.L, self.p, domain_bounds=self.quad_domain_bounds
        )

        # Expect N_r or receiver_dirs and N_s or source_dirs
        # If both are given but conflict, the values from the {source,receiver}_dirs will be used
        assert not (N_r is None and receiver_dirs is None)
        assert not (N_s is None and source_dirs is None)
        self.N_r = N_r if N_r is not None else receiver_dirs.shape[0]
        self.N_s = N_s if N_s is not None else source_dirs.shape[0]
        self.receiver_dirs = receiver_dirs if receiver_dirs is not None else \
            jnp.pi/2-jnp.linspace(0, 2*jnp.pi, self.N_r, endpoint=False)
        self.source_dirs = source_dirs if source_dirs is not None else \
            jnp.pi/2-jnp.linspace(0, 2*jnp.pi, self.N_s, endpoint=False)

        # Save the interior S, D matrices
        self.S_int = S_int
        self.D_int = D_int

        # Compute the exterior S, D matrices in case they were not already passed
        self.S_ext = S_ext if S_ext is not None else \
            gen_S_exterior(
                domain=self.domain,
                k=k,
                rad=receiver_radius,
                source_dirs=self.receiver_dirs
            )
        self.D_ext = D_ext if D_ext is not None else \
            gen_D_exterior(
                domain=self.domain,
                k=k,
                rad=receiver_radius,
                source_dirs=self.receiver_dirs
            )

        # Set up the PDE Problem object
        self.d_xx_evals = jnp.ones_like(self.domain.interior_points[..., 0])
        self.d_yy_evals = jnp.ones_like(self.domain.interior_points[..., 0])
        self.pde_problem = PDEProblem(
            domain=self.domain,
            D_xx_coefficients=self.d_xx_evals,
            D_yy_coefficients=self.d_yy_evals,
            eta=self.k,
            use_ItI=True,
        )

        # Set up the Scattering Problem object
        self.scat_problem = ScatteringProblem(
            pde_problem=self.pde_problem,
            S_int=self.S_int,
            D_int=self.D_int,
            S_ext=self.S_ext,
            D_ext=self.D_ext,
            target_points_reg=None,
            source_dirs=self.source_dirs,
        )
        self.UtQ = UtQ if UtQ is not None else \
            UniformToQuadtree(
                self.L, self.p, self.N_x,
                self.unif_domain_bounds,
                self.quad_domain_bounds,
                rel_offset=0,
            )
        # # in case I need this later...
        # self.QtU = QtU if QtU is not None else \
        #     QuadtreeToUniform(
        #         self.L, self.p, self.n_per_leaf, self.N_x,
        #         quad_domain_bounds=self.quad_domain_bounds,
        #         clip_domain_bounds=self.clip_domain_bounds,
        #         rel_offset=0,
        #     )

    def solve_exterior(self, q: jax.Array, quadtree_in=False) -> Tuple[jax.Array]:
        """Solve for uscat on the exterior for the values at the receivers
        Note: returns usc_ext_hps and also d_rs, which has its axes flipped
        to match the format of the Lippmann-Schwinger solver's outputs
        Returns both to hopefully reduce the chance of confusion later on...
        """
        # logging.info(f"Converting q to quadtree if needed")
        q_quadtree = q if quadtree_in else self.UtQ.apply(q)
        # logging.info(f"Calling the forward model...")
        usc_ext_hps = forward_model_exterior(
            scattering_problem=self.scat_problem, q=q_quadtree,
        )
        d_rs_hps = usc_ext_hps.T
        # logging.info(f"Returning usc_ext_hps and d_rs_hps")
        return usc_ext_hps, d_rs_hps

    def solve_interior(self, q: jax.Array) -> Tuple[jax.Array]:
        """Solve for uscat on the interior of the domain
        Not implemented so far...
        """
        pass
