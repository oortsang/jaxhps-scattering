import jax.numpy as jnp
import numpy as np

from hps.src.quadrature.quad_2D.adaptive_merge_indexing import (
    get_quadmerge_blocks_a,
    get_quadmerge_blocks_b,
)


class Test_get_quadmerge_blocks_a:
    def test_0(self) -> None:
        q = 7
        T = np.random.normal(size=(4 * q, 4 * q))
        v_prime = np.random.normal(size=(4 * q,))
        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))
        need_interp_5 = jnp.array([False])
        need_interp_8 = jnp.array([False])
        (
            v_prime_a_1,
            v_prime_a_5,
            v_prime_a_8,
            T_a_11,
            T_a_15,
            T_a_18,
            T_a_51,
            T_a_55,
            T_a_58,
            T_a_81,
            T_a_85,
            T_a_88,
        ) = get_quadmerge_blocks_a(
            T, v_prime, L_2f1, L_1f2, need_interp_5, need_interp_8, q, q, q, q
        )
        assert v_prime_a_1.shape == (2 * q,)
        assert v_prime_a_5.shape == (q,)
        assert v_prime_a_8.shape == (q,)
        assert T_a_11.shape == (2 * q, 2 * q)
        assert T_a_15.shape == (2 * q, q)
        assert T_a_18.shape == (2 * q, q)
        assert T_a_51.shape == (q, 2 * q)
        assert T_a_55.shape == (q, q)
        assert T_a_58.shape == (q, q)
        assert T_a_81.shape == (q, 2 * q)
        assert T_a_85.shape == (q, q)
        assert T_a_88.shape == (q, q)

    def test_1(self) -> None:
        """Compress along edge 8"""
        q = 7
        T = np.random.normal(size=(8 * q, 8 * q))
        v_prime = np.random.normal(size=(8 * q,))
        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))

        need_interp_5 = jnp.array([False, False])
        need_interp_8 = jnp.array([True, True])
        (
            v_prime_a_1,
            v_prime_a_5,
            v_prime_a_8,
            T_a_11,
            T_a_15,
            T_a_18,
            T_a_51,
            T_a_55,
            T_a_58,
            T_a_81,
            T_a_85,
            T_a_88,
        ) = get_quadmerge_blocks_a(
            T,
            v_prime,
            L_2f1,
            L_1f2,
            need_interp_5,
            need_interp_8,
            2 * q,
            2 * q,
            2 * q,
            2 * q,
        )
        # Expect the side 5 lengths to be 2 * q and side 1 lengths to be 4 * q
        assert v_prime_a_1.shape == (4 * q,)
        assert v_prime_a_5.shape == (2 * q,)
        assert v_prime_a_8.shape == (q,)
        assert T_a_11.shape == (4 * q, 4 * q)
        assert T_a_15.shape == (4 * q, 2 * q)
        assert T_a_18.shape == (4 * q, q)
        assert T_a_51.shape == (2 * q, 4 * q)
        assert T_a_55.shape == (2 * q, 2 * q)
        assert T_a_58.shape == (2 * q, q)
        assert T_a_81.shape == (q, 4 * q)
        assert T_a_85.shape == (q, 2 * q)
        assert T_a_88.shape == (q, q)

    def test_2(self) -> None:
        """Try when interpolating only half of the boundary.
        Suppose the sides of a have this many Chebyshev points:
        South side: 2q
        East side: 2q
        North side: 3q
        West side: 3q

        Suppose we want to compress the first half of the north boundary.
        """

        q = 7
        n_pts_south = 2 * q
        n_pts_east = 2 * q
        n_pts_north = 3 * q
        n_pts_west = 3 * q

        T = np.random.normal(
            size=(n_pts_south + n_pts_east + n_pts_north + n_pts_west,) * 2
        )
        print("test_2: T.shape", T.shape)

        need_interp_8 = jnp.array([True, True, False])
        need_interp_5 = jnp.array([False, False, False])

        v_prime = np.random.normal(size=T.shape[0])
        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))
        (
            v_prime_a_1,
            v_prime_a_5,
            v_prime_a_8,
            T_a_11,
            T_a_15,
            T_a_18,
            T_a_51,
            T_a_55,
            T_a_58,
            T_a_81,
            T_a_85,
            T_a_88,
        ) = get_quadmerge_blocks_a(
            T,
            v_prime,
            L_2f1,
            L_1f2,
            need_interp_5,
            need_interp_8,
            n_pts_south,
            n_pts_east,
            n_pts_north,
            n_pts_west,
        )

        # Expect the side 8 lengths to be 2 * q and side 1 lengths to be 5 * q
        assert v_prime_a_1.shape == (5 * q,)
        assert v_prime_a_5.shape == (2 * q,)
        assert v_prime_a_8.shape == (2 * q,)
        assert T_a_11.shape == (5 * q, 5 * q)
        assert T_a_15.shape == (5 * q, 2 * q)
        assert T_a_18.shape == (5 * q, 2 * q)
        assert T_a_51.shape == (2 * q, 5 * q)
        assert T_a_55.shape == (2 * q, 2 * q)
        assert T_a_58.shape == (2 * q, 2 * q)
        assert T_a_81.shape == (2 * q, 5 * q)
        assert T_a_85.shape == (2 * q, 2 * q)
        assert T_a_88.shape == (2 * q, 2 * q)

    def test_3(self) -> None:
        """Try when interpolating only half of the boundary.
        Suppose the sides of a have this many Chebyshev points:
        South side: 2q
        East side: 2q
        North side: 2q
        West side: 2q

        Suppose we want to compress the north boundary.
        """

        q = 7
        n_pts_south = 2 * q
        n_pts_east = 2 * q
        n_pts_north = 2 * q
        n_pts_west = 2 * q

        T = np.random.normal(
            size=(n_pts_south + n_pts_east + n_pts_north + n_pts_west,) * 2
        )
        print("test_2: T.shape", T.shape)

        need_interp_5 = jnp.array([False, False])
        need_interp_8 = jnp.array([True, True])

        v_prime = np.random.normal(size=T.shape[0])
        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))
        (
            v_prime_a_1,
            v_prime_a_5,
            v_prime_a_8,
            T_a_11,
            T_a_15,
            T_a_18,
            T_a_51,
            T_a_55,
            T_a_58,
            T_a_81,
            T_a_85,
            T_a_88,
        ) = get_quadmerge_blocks_a(
            T,
            v_prime,
            L_2f1,
            L_1f2,
            need_interp_5,
            need_interp_8,
            n_pts_south,
            n_pts_east,
            n_pts_north,
            n_pts_west,
        )

        # Expect the side 8 lengths to be q and side 1 lengths to be 4 * q and side 5 lengths to be 2 * q
        n_1_out = 4 * q
        n_5_out = 2 * q
        n_8_out = q
        assert v_prime_a_1.shape == (n_1_out,)
        assert v_prime_a_5.shape == (n_5_out,)
        assert v_prime_a_8.shape == (n_8_out,)
        assert T_a_11.shape == (n_1_out, n_1_out)
        assert T_a_15.shape == (n_1_out, n_5_out)
        assert T_a_18.shape == (n_1_out, n_8_out)
        assert T_a_51.shape == (n_5_out, n_1_out)
        assert T_a_55.shape == (n_5_out, n_5_out)
        assert T_a_58.shape == (n_5_out, n_8_out)
        assert T_a_81.shape == (n_8_out, n_1_out)
        assert T_a_85.shape == (n_8_out, n_5_out)
        assert T_a_88.shape == (n_8_out, n_8_out)


class Test_get_quadmerge_blocks_b:
    def test_0(self) -> None:
        """Tests the case when b has this many Chebyshev points:
        south: 2 q
        east: 2 q
        north: 2 q
        west: 2 q

        and we want to compress the whole north boundary.
        """
        q = 7
        n_pts_side = 2 * q
        T = np.random.normal(size=(4 * n_pts_side, 4 * n_pts_side))
        print("test_0: T.shape", T.shape)
        v_prime = np.random.normal(size=(4 * n_pts_side,))
        L_2f1 = np.random.normal(size=(2 * q, q))
        L_1f2 = np.random.normal(size=(q, 2 * q))

        need_interp_6 = jnp.array([True, True])
        need_interp_5 = jnp.array([False, False])

        (
            v_prime_b_2,
            v_prime_b_6,
            v_prime_b_5,
            T_b_22,
            T_b_26,
            T_b_25,
            T_b_62,
            T_b_66,
            T_b_65,
            T_b_52,
            T_b_56,
            T_b_55,
        ) = get_quadmerge_blocks_b(
            T,
            v_prime,
            L_2f1,
            L_1f2,
            need_interp_6,
            need_interp_5,
            n_pts_side,
            n_pts_side,
            n_pts_side,
            n_pts_side,
        )

        # Expect the side 6 lengths to be  q, side 5 lengths to be 2 * q, and side 2 length to be 4  * q
        assert v_prime_b_2.shape == (4 * q,)
        assert v_prime_b_6.shape == (q,)
        assert v_prime_b_5.shape == (2 * q,)
        assert T_b_22.shape == (4 * q, 4 * q)
        assert T_b_26.shape == (4 * q, q)
        assert T_b_25.shape == (4 * q, 2 * q)
        assert T_b_62.shape == (q, 4 * q)
        assert T_b_66.shape == (q, q)
        assert T_b_65.shape == (q, 2 * q)
        assert T_b_52.shape == (2 * q, 4 * q)
        assert T_b_56.shape == (2 * q, q)
        assert T_b_55.shape == (2 * q, 2 * q)
