import numpy as np
import pytest

from hps.src.utils import (
    lst_of_points_to_meshgrid,
    points_to_2d_lst_of_points,
    meshgrid_to_lst_of_pts,
)
from hps.src.test_utils import check_arrays_close


class Test_lst_of_points_to_meshgrid:
    def test_0(self) -> None:
        """Checks the function returns without error"""
        n = 5
        x = np.random.normal(size=(n**2, 2))
        xx, yy = lst_of_points_to_meshgrid(x)
        assert xx.shape == (n, n)
        assert yy.shape == (n, n)

    def test_1(self) -> None:

        x = np.linspace(0, 1, 4)
        X, Y = np.meshgrid(x, np.flipud(x), indexing="ij")
        pts_lst = points_to_2d_lst_of_points(x)
        xx, yy = lst_of_points_to_meshgrid(pts_lst)
        check_arrays_close(xx, X)
        check_arrays_close(yy, Y)


class Test_meshgrid_to_lst_of_pts:
    def test_0(self) -> None:
        """Checks that things return the correct shape and X and Y get put into the correct dimsnsions."""

        n = 5

        x_flat = np.arange(n**2)
        y_flat = np.arange(n**2) + 101

        X = x_flat.reshape(n, n)
        Y = y_flat.reshape(n, n)

        out = meshgrid_to_lst_of_pts(X, Y)
        assert out.shape == (n**2, 2)

        print(out[:, 0])

        assert np.all(out[:, 0] < 100)
        assert np.all(out[:, 1] > 100)


if __name__ == "__main__":
    pytest.main()
