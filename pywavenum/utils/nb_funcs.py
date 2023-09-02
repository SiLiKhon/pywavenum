from numba import njit
import numpy as np

@njit
def low_pass_filter(
    array: np.ndarray,
    alphas: np.ndarray,
) -> None:
    assert array.ndim == 1
    assert array.shape == alphas.shape

    for i in range(1, array.shape[0]):
        array[i] = array[i - 1] + alphas[i] * (array[i] - array[i - 1])

@njit
def high_pass_filter(
    array: np.ndarray,
    alphas: np.ndarray,
) -> None:
    assert array.ndim == 1
    assert array.shape == alphas.shape

    pre_upd_val = array[0]
    for i in range(1, array.shape[0]):
        (array[i], pre_upd_val) = (
            alphas[i] * (array[i - 1] + array[i] - pre_upd_val),
            array[i],
        )
