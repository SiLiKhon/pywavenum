from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pydub

Number = Union[int, float, np.int_, np.float_]

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def _soft_softplus(x: np.ndarray, softness: np.ndarray) -> np.ndarray:
    return _softplus(x / softness) * softness


class Signal(ABC):
    @abstractmethod
    def __call__(self, t: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def _as_signal(v: Union[Number, "Signal"]) -> "Signal":
        if not isinstance(v, Signal):
            v = Constant(v)
        return v

    def __add__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a + b, self, Signal._as_signal(other))

    def __radd__(self, other: Union[Number, "Signal"]) -> "Signal":
        return self + other

    def __sub__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a - b, self, Signal._as_signal(other))

    def __mul__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a * b, self, Signal._as_signal(other))

    def __rmul__(self, other: Union[Number, "Signal"]) -> "Signal":
        return self * other

    def __truediv__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a / b, self, Signal._as_signal(other))

    def __pow__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a**b, self, Signal._as_signal(other))

    def __rpow__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a**b, Signal._as_signal(other), self)

    def gain(self, dB: Union[Number, "Signal"]) -> "Signal":
        factor = 10**(Signal._as_signal(dB) / 20)
        return self * factor

    def get_samples(
        self,
        tmin: float = 0.0,
        tmax: float = 1.0,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, np.ndarray]:
        tt = np.arange((tmax - tmin) * sample_rate) / sample_rate + tmin
        yy = (np.clip(self(tt), -1, 1) * 2**15).astype(np.int16)
        return tt, yy
    
    def visualize(
        self,
        tmin: float = 0.0,
        tmax: float = 1.0,
        sample_rate: int = 44100,
        ylims: Optional[Tuple[float, float]]=(-1, 1)
    ) -> mpl.figure.Figure:
        tt, yy = self.get_samples(tmin, tmax, sample_rate)
        yy = yy / 2**15
        fig = plt.figure(figsize=(16, 8), tight_layout=True)
        MAX_SAMPLES = 10000
        MAX_LINES = 12
        MIN_LINES = 4
        nlines = min(MAX_LINES, int(np.ceil(len(tt) / MAX_SAMPLES)))
        nlines = max(MIN_LINES, nlines)
        samples_per_line = int(np.ceil(len(tt) / nlines))
        gs = mpl.gridspec.GridSpec(nlines, 2)
    
        for i_line in range(nlines):
            first_sample = samples_per_line * i_line
            last_sample = first_sample + samples_per_line
            plt.sca(fig.add_subplot(gs[i_line, 0]))
            plt.plot(tt[first_sample: last_sample], yy[first_sample: last_sample])
            if ylims is not None:
                plt.ylim(*ylims)

        plt.sca(fig.add_subplot(gs[:, 1]))
        plt.specgram(yy, NFFT=2048, Fs=sample_rate, noverlap=1792)
        plt.yscale("log")
        plt.ylim(10, 40000)
        plt.colorbar()

        return fig

    def to_audio(
        self,
        tmin: float = 0.0,
        tmax: float = 1.0,
        sample_rate: int = 44100
    ) -> pydub.AudioSegment:
        _, yy = self.get_samples(tmin, tmax, sample_rate)
        yy = np.c_[yy, yy]

        return pydub.AudioSegment(
            yy.tobytes(), sample_width=2, frame_rate=sample_rate, channels=2,
        )


class Time(Signal):
    def __call__(self, t: np.ndarray) -> np.ndarray:
        return t

class Composite(Signal):
    def __init__(self, func: Callable, *signals: Signal):
        self.func = func
        self.signals = signals

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return self.func(*[s(t) for s in self.signals])

class Constant(Signal):
    def __init__(self, value: Union[Number, Signal]):
        self.val = value

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return self.val * np.ones_like(t)

class Sine(Signal):
    def __init__(
        self,
        freq: Union[Number, Signal],
        phase: Union[Number, Signal] = 0.0,
        phase_correction: bool = True,
    ):
        self.freq = self._as_signal(freq)
        self.phase = self._as_signal(phase)
        self.phase_correction = phase_correction

    def __call__(self, t: np.ndarray) -> np.ndarray:
        freq = self.freq(t)
        phase = self.phase(t)
        pre_arg = t * freq + phase
        if self.phase_correction:
            df = np.diff(freq, prepend=freq[0])
            corr_phase = (df * t).cumsum()
            pre_arg -= corr_phase
        arg = 2 * np.pi * pre_arg
        return np.sin(arg)

class Step(Signal):
    def __init__(self, threshold: Number, reverse: bool = False):
        self.thr = threshold
        self.rev = reverse

    def __call__(self, t: np.ndarray) -> np.ndarray:
        cond = t >= self.thr
        if self.rev:
            cond = ~cond
        return np.where(cond, 1.0, 0.0)

class Pulse(Signal):
    def __init__(self, start: Number, stop: Number):
        self.start = start
        self.stop = stop

    def __call__(self, t: np.ndarray) -> np.ndarray:
        out = np.ones_like(t)
        return np.where((t >= self.start) & (t <= self.stop), out, 0)

class Delay(Signal):
    def __init__(self, signal: Signal, amount: Number):
        self.signal = signal
        self.amount = amount

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return self.signal(t - self.amount)

class Clip(Signal):
    def __init__(
        self,
        signal: Signal,
        threshold_dB: Union[Number, Signal],
        softness: Union[Number, Signal] = 0.01
    ):
        self.signal = signal
        self.softness = self._as_signal(softness)
        self.thr = 10**(self._as_signal(threshold_dB) / 20)

    @staticmethod
    def _soft_clip(val: np.ndarray, thr: np.ndarray, softness: np.ndarray):
        clipped_above = thr - _soft_softplus(thr - val, softness)
        return _soft_softplus(thr + clipped_above, softness) - thr

    def __call__(self, t: np.ndarray) -> np.ndarray:
        sig_vals = self.signal(t)
        return 0.5 * (
            self._soft_clip(sig_vals, self.thr(t), self.softness(t))
            -self._soft_clip(-sig_vals, self.thr(t), self.softness(t))
        )

class DiscreteSignal(Signal):
    def __init__(
        self,
        signal: Signal,
        tmin: float = 0.0,
        tmax: float = 1.0,
        sample_rate: int = 44100,
    ):
        self.sample_rate = sample_rate
        self.tmin = tmin
        self.tmax = tmax
        self.tt, self.samples = signal.get_samples(tmin, tmax, sample_rate)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        dt = np.diff(t)
        if not (dt > 0).all():
            raise ValueError("All values of t must be consecutive")
        if not np.allclose(dt[0], dt):
            raise ValueError("All values of t must be equally spaced")

        if not np.isclose(dt[0], 1.0 / self.sample_rate):
            raise ValueError("Sample rate doesn't match")

        dt = dt[0]
        tmin = t.min()
        time_offset = self.tmin - tmin
        if (np.abs(((time_offset + dt / 2) % dt) - dt / 2) / dt) > 1e-6:
            raise ValueError("Grid not aligned")

        samples = self.samples.copy()

        n_leading_zeros = int(np.round(time_offset / dt))
        if n_leading_zeros <= 0:
            samples = samples[-n_leading_zeros:]
        else:
            samples = np.concatenate([np.zeros(n_leading_zeros, dtype=samples.dtype), samples])

        n_trailing_zeros = len(t) - len(samples)
        if n_trailing_zeros <= 0:
            samples = samples[:len(samples) + n_trailing_zeros]
        else:
            samples = np.concatenate([samples, np.zeros(n_trailing_zeros, dtype=samples.dtype)])

        return samples
