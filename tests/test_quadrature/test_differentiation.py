from jaxhps.quadrature import differentiation_matrix_1D, chebyshev_points
import jax.numpy as jnp


class Test_differentiation_matrix_1D:
    def test_0(self) -> None:
        p = jnp.arange(10)
        o = differentiation_matrix_1D(p)
        assert o.shape == (10, 10), o.shape

    def test_1(self) -> None:
        """Check n=2 against
        example from MATLAB textbook (T00)
        """

        p = chebyshev_points(2)

        d = differentiation_matrix_1D(p)

        expected_d = jnp.array([[-1 / 2, 1 / 2], [-1 / 2, 1 / 2]])

        print(d)
        print(expected_d)

        assert jnp.allclose(d, expected_d)

    def test_2(self) -> None:
        """Check n=3 against
        example from MATLAB textbook (T00)
        """

        p = chebyshev_points(3)

        d = differentiation_matrix_1D(p)

        expected_d = jnp.array(
            [[-3 / 2, 2, -1 / 2], [-1 / 2, 0, 1 / 2], [1 / 2, -2, 3 / 2]]
        )

        print(d)
        print(expected_d)
        print(d - expected_d)

        assert jnp.allclose(d, expected_d)

    def test_3(self) -> None:
        """Check n=4 against example from MATLAB textbook (T00)"""

        p = chebyshev_points(4)

        d = differentiation_matrix_1D(p)

        expected_d = jnp.array(
            [
                [19 / 6, -4, 4 / 3, -1 / 2],
                [1, -1 / 3, -1, 1 / 3],
                [-1 / 3, 1, 1 / 3, -1],
                [1 / 2, -4 / 3, 4, -19 / 6],
            ]
        )
        expected_d = -1 * expected_d

        print(d)
        print(expected_d)
        print(d - expected_d)

        assert jnp.allclose(d, expected_d)

    def test_4(self) -> None:
        """Checks the matrix differentiates f(x) = x^2 correctly."""

        n = 7
        p = chebyshev_points(n)
        print("test_4: p: ", p)
        d = differentiation_matrix_1D(p)
        f = p**2
        df = d @ f
        expected = 2 * p
        print("test_4: df: ", df)
        print("test_4: expected: ", expected)
        assert jnp.allclose(df, expected)
