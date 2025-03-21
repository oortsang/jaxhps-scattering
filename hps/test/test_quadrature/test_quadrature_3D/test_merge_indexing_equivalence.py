import jax.numpy as jnp
import numpy as np
import pytest

from hps.src.quadrature.quad_3D.uniform_merge_indexing import (
    get_a_submatrices as get_a_submatrices_uniform,
    get_b_submatrices as get_b_submatrices_uniform,
    get_c_submatrices as get_c_submatrices_uniform,
    get_d_submatrices as get_d_submatrices_uniform,
    get_e_submatrices as get_e_submatrices_uniform,
    get_f_submatrices as get_f_submatrices_uniform,
    get_g_submatrices as get_g_submatrices_uniform,
    get_h_submatrices as get_h_submatrices_uniform,
    get_rearrange_indices as get_rearrange_indices_uniform,
)
from hps.src.quadrature.quad_3D.adaptive_merge_indexing import (
    get_a_submatrices as get_a_submatrices_adaptive,
    get_b_submatrices as get_b_submatrices_adaptive,
    get_c_submatrices as get_c_submatrices_adaptive,
    get_d_submatrices as get_d_submatrices_adaptive,
    get_e_submatrices as get_e_submatrices_adaptive,
    get_f_submatrices as get_f_submatrices_adaptive,
    get_g_submatrices as get_g_submatrices_adaptive,
    get_h_submatrices as get_h_submatrices_adaptive,
    get_rearrange_indices as get_rearrange_indices_adaptive,
)


class Test_a_submatrices_equivalence:
    def test_0(self) -> None:
        q = 2
        n_per_panel = q**2

        T = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
        v = np.random.normal(size=(6 * n_per_panel,))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))
        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        need_interp = jnp.array([False])

        a_submatrices_uniform = get_a_submatrices_uniform(T, v)
        a_submatrices_adaptive = get_a_submatrices_adaptive(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp,
            need_interp,
            need_interp,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
        )

        for i in range(len(a_submatrices_uniform)):
            print("test_0: i=", i)
            assert (
                a_submatrices_uniform[i].shape
                == a_submatrices_adaptive[i].shape
            )
            assert jnp.allclose(
                a_submatrices_uniform[i], a_submatrices_adaptive[i]
            )


class Test_b_submatrices_equivalence:
    def test_0(self) -> None:
        q = 2
        n_per_panel = q**2

        T = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
        v = np.random.normal(size=(6 * n_per_panel,))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))
        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        need_interp = jnp.array([False])

        b_submatrices_uniform = get_b_submatrices_uniform(T, v)
        b_submatrices_adaptive = get_b_submatrices_adaptive(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp,
            need_interp,
            need_interp,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
        )

        for i in range(len(b_submatrices_uniform)):
            print("test_0: i=", i)
            assert (
                b_submatrices_uniform[i].shape
                == b_submatrices_adaptive[i].shape
            )
            assert jnp.allclose(
                b_submatrices_uniform[i], b_submatrices_adaptive[i]
            )


class Test_c_submatrices_equivalence:
    def test_0(self) -> None:
        q = 2
        n_per_panel = q**2

        T = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
        v = np.random.normal(size=(6 * n_per_panel,))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))
        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        need_interp = jnp.array([False])

        c_submatrices_uniform = get_c_submatrices_uniform(T, v)
        c_submatrices_adaptive = get_c_submatrices_adaptive(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp,
            need_interp,
            need_interp,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
        )

        for i in range(len(c_submatrices_uniform)):
            print("test_0: i=", i)
            assert (
                c_submatrices_uniform[i].shape
                == c_submatrices_adaptive[i].shape
            )
            assert jnp.allclose(
                c_submatrices_uniform[i], c_submatrices_adaptive[i]
            )


class Test_d_submatrices_equivalence:
    def test_0(self) -> None:
        q = 2
        n_per_panel = q**2

        T = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
        v = np.random.normal(size=(6 * n_per_panel,))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))
        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        need_interp = jnp.array([False])

        d_submatrices_uniform = get_d_submatrices_uniform(T, v)
        d_submatrices_adaptive = get_d_submatrices_adaptive(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp,
            need_interp,
            need_interp,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
        )

        for i in range(len(d_submatrices_uniform)):
            print("test_0: i=", i)
            assert (
                d_submatrices_uniform[i].shape
                == d_submatrices_adaptive[i].shape
            )
            assert jnp.allclose(
                d_submatrices_uniform[i], d_submatrices_adaptive[i]
            )


class Test_e_submatrices_equivalence:
    def test_0(self) -> None:
        q = 2
        n_per_panel = q**2

        T = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
        v = np.random.normal(size=(6 * n_per_panel,))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))
        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        need_interp = jnp.array([False])

        e_submatrices_uniform = get_e_submatrices_uniform(T, v)
        e_submatrices_adaptive = get_e_submatrices_adaptive(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp,
            need_interp,
            need_interp,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
        )

        for i in range(len(e_submatrices_uniform)):
            print("test_0: i=", i)
            assert (
                e_submatrices_uniform[i].shape
                == e_submatrices_adaptive[i].shape
            )
            assert jnp.allclose(
                e_submatrices_uniform[i], e_submatrices_adaptive[i]
            )


class Test_f_submatrices_equivalence:
    def test_0(self) -> None:
        q = 2
        n_per_panel = q**2

        T = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
        v = np.random.normal(size=(6 * n_per_panel,))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))
        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        need_interp = jnp.array([False])

        f_submatrices_uniform = get_f_submatrices_uniform(T, v)
        f_submatrices_adaptive = get_f_submatrices_adaptive(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp,
            need_interp,
            need_interp,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
        )

        for i in range(len(f_submatrices_uniform)):
            print("test_0: i=", i)
            assert (
                f_submatrices_uniform[i].shape
                == f_submatrices_adaptive[i].shape
            )
            assert jnp.allclose(
                f_submatrices_uniform[i], f_submatrices_adaptive[i]
            )


class Test_g_submatrices_equivalence:
    def test_0(self) -> None:
        q = 2
        n_per_panel = q**2

        T = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
        v = np.random.normal(size=(6 * n_per_panel,))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))
        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        need_interp = jnp.array([False])

        g_submatrices_uniform = get_g_submatrices_uniform(T, v)
        g_submatrices_adaptive = get_g_submatrices_adaptive(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp,
            need_interp,
            need_interp,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
        )

        for i in range(len(g_submatrices_uniform)):
            print("test_0: i=", i)
            assert (
                g_submatrices_uniform[i].shape
                == g_submatrices_adaptive[i].shape
            )
            assert jnp.allclose(
                g_submatrices_uniform[i], g_submatrices_adaptive[i]
            )


class Test_h_submatrices_equivalence:
    def test_0(self) -> None:
        q = 2
        n_per_panel = q**2

        T = np.random.normal(size=(6 * n_per_panel, 6 * n_per_panel))
        v = np.random.normal(size=(6 * n_per_panel,))
        L_2f1 = np.random.normal(size=(4 * n_per_panel, n_per_panel))
        L_1f2 = np.random.normal(size=(n_per_panel, 4 * n_per_panel))
        need_interp = jnp.array([False])

        h_submatrices_uniform = get_h_submatrices_uniform(T, v)
        h_submatrices_adaptive = get_h_submatrices_adaptive(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp,
            need_interp,
            need_interp,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
        )

        for i in range(len(h_submatrices_uniform)):
            print("test_0: i=", i)
            assert (
                h_submatrices_uniform[i].shape
                == h_submatrices_adaptive[i].shape
            )
            assert jnp.allclose(
                h_submatrices_uniform[i], h_submatrices_adaptive[i]
            )


class Test_rearrange_indices_equivalence:
    def test_0(self) -> None:
        q = 4
        n_per_panel = q**2
        n_panels = 24

        idxes = jnp.arange(n_per_panel * n_panels)
        r_uniform = get_rearrange_indices_uniform(idxes, None)

        r_adaptive = get_rearrange_indices_adaptive(
            idxes,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
            n_per_panel,
        )
        assert r_uniform.shape == r_adaptive.shape
        print("test_0: r_uniform=", r_uniform)
        print("test_0: r_adaptive=", r_adaptive)
        print("test_0: r_uniform - r_adaptive=", r_uniform - r_adaptive)
        assert jnp.all(r_uniform == r_adaptive)


if __name__ == "__main__":
    pytest.main()
