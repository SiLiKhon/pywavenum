from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pydub

from . import nb_funcs

Number = Union[int, float, np.int_, np.float_]

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def _soft_softplus(x: np.ndarray, softness: np.ndarray) -> np.ndarray:
    return _softplus(x / softness) * softness


class Signal(ABC):
    def __init__(self, time_range: Tuple[Number, Number] = (-np.inf, np.inf)):
        if time_range[0] >= time_range[1]:
            raise ValueError(f"Invalid time range {time_range}")
        self.time_range = time_range

    def __call__(self, t: np.ndarray) -> np.ndarray:
        result = np.empty_like(t)
        selection = (t >= self.time_range[0]) & (t <= self.time_range[1])
        result[~selection] = 0.0
        result[selection] = self.render(t[selection])
        return result

    @abstractmethod
    def render(self, t: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def _as_signal(v: Union[Number, "Signal"]) -> "Signal":
        if not isinstance(v, Signal):
            v = Constant(v)
        return v

    def __add__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a + b, self, Signal._as_signal(other), time_range_mode="join")

    def __radd__(self, other: Union[Number, "Signal"]) -> "Signal":
        return self + other

    def __sub__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a - b, self, Signal._as_signal(other), time_range_mode="join")

    def __mul__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a * b, self, Signal._as_signal(other), time_range_mode="intersect")

    def __rmul__(self, other: Union[Number, "Signal"]) -> "Signal":
        return self * other

    def __truediv__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a / b, self, Signal._as_signal(other), time_range_mode="intersect")

    def __rtruediv__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a / b, Signal._as_signal(other), self, time_range_mode="intersect")

    def __pow__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a**b, self, Signal._as_signal(other), time_range_mode="join")

    def __rpow__(self, other: Union[Number, "Signal"]) -> "Signal":
        return Composite(lambda a, b: a**b, Signal._as_signal(other), self, time_range_mode="join")

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
    def render(self, t: np.ndarray) -> np.ndarray:
        return t


class Composite(Signal):
    def __init__(self, func: Callable, *signals: Signal, time_range_mode: str):
        self.func = func
        self.signals = signals
        if time_range_mode == "intersect":
            tmin = max(s.time_range[0] for s in signals)
            tmax = min(s.time_range[1] for s in signals)
            if tmin >= tmax:
                tmax = tmin + 1e-7
        elif time_range_mode == "join":
            tmin = min(s.time_range[0] for s in signals)
            tmax = max(s.time_range[1] for s in signals)
        else:
            raise NotImplementedError(time_range_mode)
        super().__init__((tmin, tmax))

    def render(self, t: np.ndarray) -> np.ndarray:
        return self.func(*[s(t) for s in self.signals])


class Constant(Signal):
    def __init__(self, value: Union[Number, Signal], **kwargs):
        super().__init__(**kwargs)
        self.val = value

    def render(self, t: np.ndarray) -> np.ndarray:
        return self.val * np.ones_like(t)


class Sine(Signal):
    def __init__(
        self,
        freq: Union[Number, Signal],
        phase: Union[Number, Signal] = 0.0,
        phase_correction: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freq = self._as_signal(freq)
        self.phase = self._as_signal(phase)
        self.phase_correction = phase_correction

    def render(self, t: np.ndarray) -> np.ndarray:
        freq = self.freq(t)
        phase = self.phase(t)
        pre_arg = t * freq + phase
        if self.phase_correction:
            df = np.diff(freq, prepend=freq[0])
            corr_phase = (df * t).cumsum()
            pre_arg -= corr_phase
        arg = 2 * np.pi * pre_arg
        return np.sin(arg)


class Periodic(Signal):
    def __init__(
        self,
        waveform: Callable,
        freq: Union[Number, Signal],
        phase: Union[Number, Signal] = 0.0,
        phase_correction: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wf = waveform
        self.freq = self._as_signal(freq)
        self.phase = self._as_signal(phase)
        self.phase_correction = phase_correction

    def render(self, t: np.ndarray) -> np.ndarray:
        freq = self.freq(t)
        phase = self.phase(t)
        arg = t * freq + phase
        if self.phase_correction:
            df = np.diff(freq, prepend=freq[0])
            corr_phase = (df * t).cumsum()
            arg -= corr_phase
        return self.wf(arg % 1.0, t)


class Step(Signal):
    def __init__(self, threshold: Number, reverse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.thr = threshold
        self.rev = reverse

    def render(self, t: np.ndarray) -> np.ndarray:
        cond = t >= self.thr
        if self.rev:
            cond = ~cond
        return np.where(cond, 1.0, 0.0)


class Pulse(Signal):
    def __init__(self, start: Number, stop: Number):
        super().__init__((start, stop))

    def render(self, t: np.ndarray) -> np.ndarray:
        return np.ones_like(t)


class Delay(Signal):
    def __init__(self, signal: Signal, amount: Number):
        super().__init__((signal.time_range[0] + amount, signal.time_range[1] + amount,))
        self.signal = signal
        self.amount = amount

    def render(self, t: np.ndarray) -> np.ndarray:
        return self.signal.render(t - self.amount)


class Clip(Signal):
    def __init__(
        self,
        signal: Signal,
        threshold_dB: Union[Number, Signal],
        softness: Union[Number, Signal] = 0.01
    ):
        super().__init__(signal.time_range)
        self.signal = signal
        self.softness = self._as_signal(softness)
        self.thr = 10**(self._as_signal(threshold_dB) / 20)

    @staticmethod
    def _soft_clip(val: np.ndarray, thr: np.ndarray, softness: np.ndarray):
        clipped_above = thr - _soft_softplus(thr - val, softness)
        return _soft_softplus(thr + clipped_above, softness) - thr

    def render(self, t: np.ndarray) -> np.ndarray:
        sig_vals = self.signal(t)
        return 0.5 * (
            self._soft_clip(sig_vals, self.thr(t), self.softness(t))
            -self._soft_clip(-sig_vals, self.thr(t), self.softness(t))
        )


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


class PiecewiseLinear(Signal):
    def __init__(self, tt: np.ndarray, yy: np.ndarray, extend: bool = False):
        if not (np.diff(tt) > 0).all():
            raise ValueError("Times must be consecutive")

        if len(tt) != len(yy):
            raise ValueError("Arrays must be of same size")

        if extend:
            tmin = -np.inf
            tmax = np.inf
        else:
            tmin = min(tt)
            tmax = max(tt)
        super().__init__((tmin, tmax))

        if extend:
            self.signal = (
                yy[0] * Pulse(tmin, tt[0] - 1e-7)
                + yy[-1] * Pulse(tt[-1], tmax)
            )
        else:
            self.signal = Constant(0, time_range=self.time_range)
        for t0, t1, y0, y1 in zip(tt[:-1], tt[1:], yy[:-1], yy[1:]):
            self.signal = self.signal + Pulse(t0, t1 - 1e-7) * (
                y0 + (y1 - y0) * (Time() - t0) / (t1 - t0)
            )

    def render(self, t):
        return self.signal.render(t)


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
