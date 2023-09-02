from typing import Union

import numpy as np

from .signal import Signal, Number
from ..utils import nb_funcs

class DiscreteSignal(Signal):
    @staticmethod
    def _deduce_sample_rate(t: np.ndarray) -> float:
        dt = np.diff(t)
        if not (dt > 0).all():
            raise ValueError("All values of t must be consecutive")
        dt_mean = dt.mean()

        if not np.allclose(dt_mean, dt):
            raise ValueError("All values of t must be equally spaced")

        return 1.0 / dt_mean

    def __init__(self, signal: Signal):
        super().__init__(signal.time_range)
        self.signal = signal

    def render(self, t: np.ndarray) -> np.ndarray:
        self.sample_rate = self._deduce_sample_rate(t)
        return self.signal(t)


class LPF(DiscreteSignal):
    def __init__(self, signal: Signal, freq: Union[Number, Signal]):
        super().__init__(signal)
        self.freq = self._as_signal(freq)

    def render(self, t: np.ndarray) -> np.ndarray:
        samples = super().render(t)
        alpha = (1.0 / (1.0 + self.sample_rate / self.freq))(t)
        nb_funcs.low_pass_filter(samples, alpha)
        return samples


class HPF(DiscreteSignal):
    def __init__(self, signal: Signal, freq: Union[Number, Signal]):
        super().__init__(signal)
        self.freq = self._as_signal(freq)

    def render(self, t: np.ndarray) -> np.ndarray:
        samples = super().render(t)
        alpha = (1.0 / (1.0 + self.freq / self.sample_rate))(t)
        nb_funcs.high_pass_filter(samples, alpha)
        return samples
