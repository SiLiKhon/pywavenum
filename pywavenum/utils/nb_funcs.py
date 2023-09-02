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

@njit
def smooth_envelope(
    envelope: np.ndarray,
    alphas_attack: np.ndarray,
    alphas_release: np.ndarray,
) -> None:
    assert envelope.ndim == 1
    assert envelope.shape == alphas_attack.shape
    assert envelope.shape == alphas_release.shape

    result_attack = np.empty_like(envelope)
    result_release = np.empty_like(envelope)

    N = envelope.shape[0]

    result_attack[N - 1] = envelope[N - 1]
    result_release[0] = envelope[0]

    for i in range(1, N):
        result_release[i] = max(
            envelope[i], (1 - alphas_release[i]) * envelope[i] + alphas_release[i] * result_release[i - 1]
        )
        j = N - 1 - i
        result_attack[j] = max(
            envelope[j], (1 - alphas_attack[j]) * envelope[j] + alphas_attack[j] * result_attack[j + 1]
        )
    envelope[:] = np.maximum(result_release, result_attack)
