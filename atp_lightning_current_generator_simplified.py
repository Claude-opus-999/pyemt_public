"""
ATP TWOEXPF / HEIDLERF compatible lightning-current source generator.

v3 final changes:
- Lightning current source only: ATP U/I is fixed to -1.
- No source_type argument.
- No waveform_type argument.
- Use T1 and T2 directly.
- Evaluation, sampling, table export, and ATP-format output methods removed.
- Default PERC is 30.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np

try:
    from scipy.optimize import least_squares, minimize_scalar
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False


ModelName = Literal["twoexpf", "heidlerf"]
ParameterSource = Literal["direct", "fitted", "standard"]
StandardWaveformName = str


ATP_CURRENT_UI = -1
VALID_PERC = {0, 10, 30, 50}

TWOEXPF_RATIO_MIN_BY_PERC: Dict[int, float] = {
    0: 2.8,
    10: 3.9,
    30: 3.5,
    50: 4.4,
}

HEIDLERF_RATIO_MIN = 2.0
HEIDLERF_RATIO_MAX = 100.0

# Standard double-exponential waveform library.
#
# Values are tau-form parameters:
#     raw(t) = exp(-t/tau1) - exp(-t/tau2), tau1 > tau2 > 0
#
# They are intentionally stored as direct tau1/tau2 parameters so that standard
# TWOEXPF waveforms can skip numerical fitting.  A/B are derived as:
#     A = -1/tau1, B = -1/tau2
#
# The keys follow common impulse notation in microseconds.  They include both
# current and voltage impulse shapes; this module still creates ATP current
# sources because U/I is fixed to -1.
STANDARD_DOUBLE_EXPONENTIAL_PARAMS: Dict[str, Tuple[float, float, str]] = {
    "1.2/50":   (68.22e-6,    0.4074e-6,  "standard lightning voltage impulse"),
    "2/20":     (2.301907e-5, 8.285809e-7, "subsequent stroke current impulse"),
    "8/20":     (20.37e-6,    3.91e-6,    "standard lightning current impulse"),
    "4/10":     (10.48e-6,    1.93e-6,    "fast lightning current impulse"),
    "10/350":   (485.0e-6,    19.0e-6,    "first stroke current impulse"),
    "0.25/100": (143.0e-6,    0.454e-6,   "subsequent stroke current impulse"),
    "10/700":   (1000.0e-6,   19.0e-6,    "telecommunication line lightning impulse"),
    "30/80":    (111.0e-6,    18.5e-6,    "switching current impulse"),
    "250/2500": (3470.0e-6,   363.0e-6,   "long-front voltage impulse"),
    "1/200":    (280.0e-6,    1.82e-6,    "subsequent stroke current impulse"),
}


def normalize_standard_waveform_name(waveform_type: str) -> str:
    """Normalize a standard waveform name such as '8/20 us' to '8/20'."""
    key = waveform_type.strip().lower()
    key = key.replace("μs", "").replace("us", "").replace("microsecond", "")
    key = key.replace("microseconds", "").replace(" ", "")
    return key


def get_standard_waveform_params(
    waveform_type: str,
) -> Tuple[float, float, str]:
    """Return (tau1, tau2, description) for a standard double-exponential waveform."""
    key = normalize_standard_waveform_name(waveform_type)
    if key not in STANDARD_DOUBLE_EXPONENTIAL_PARAMS:
        raise ValueError(
            f"Unknown standard waveform_type={waveform_type!r}; "
            f"available: {sorted(STANDARD_DOUBLE_EXPONENTIAL_PARAMS)}"
        )
    return STANDARD_DOUBLE_EXPONENTIAL_PARAMS[key]


def parse_standard_waveform_T1_T2(waveform_type: str) -> Tuple[float, float]:
    """Parse nominal T1/T2 in seconds from a standard name such as '8/20'."""
    key = normalize_standard_waveform_name(waveform_type)
    try:
        t1_str, t2_str = key.split("/")
        return float(t1_str) * 1e-6, float(t2_str) * 1e-6
    except Exception as exc:
        raise ValueError(f"Could not parse T1/T2 from waveform_type={waveform_type!r}") from exc


def list_standard_waveforms() -> Tuple[str, ...]:
    """Return available standard waveform names."""
    return tuple(STANDARD_DOUBLE_EXPONENTIAL_PARAMS.keys())


@dataclass(frozen=True)
class WaveformTimes:
    """Characteristic times of a normalized impulse waveform."""

    peak_value: float
    t_peak: float
    t10: float
    t30: float
    t90: float
    t50_tail: float
    virtual_zero_10: float
    virtual_zero_30: float

    def T1_T2_for_perc(self, perc: int) -> Tuple[float, float]:
        """Return ATP-style T1/T2 according to PERC definition."""
        perc = validate_perc(perc)

        if perc == 0:
            return self.t_peak, self.t50_tail

        if perc == 10:
            T1 = (self.t90 - self.t10) / 0.8
            T2 = self.t50_tail - self.virtual_zero_10
            return T1, T2

        if perc == 30:
            T1 = (self.t90 - self.t30) / 0.6
            T2 = self.t50_tail - self.virtual_zero_30
            return T1, T2

        if perc == 50:
            T1 = self.t90 - self.t10
            T2 = self.t50_tail
            return T1, T2

        raise ValueError(f"Unsupported PERC={perc}")


@dataclass
class FitResult:
    """Result returned by automatic parameter fitting."""

    model: ModelName
    parameter_source: ParameterSource
    T1_target: float
    T2_target: float
    PERC: int
    n: Optional[float] = None

    # TWOEXPF parameters
    tau1: Optional[float] = None
    tau2: Optional[float] = None
    A: Optional[float] = None
    B: Optional[float] = None

    # HEIDLERF parameters
    Tf: Optional[float] = None
    tau: Optional[float] = None

    fit_error: Optional[float] = None
    message: str = ""


def validate_perc(PERC: int) -> int:
    PERC = int(PERC)
    if PERC not in VALID_PERC:
        raise ValueError(f"PERC must be one of {sorted(VALID_PERC)}, got {PERC}")
    return PERC


def validate_T1_T2(T1: float, T2: float) -> None:
    if not (isfinite(T1) and isfinite(T2)):
        raise ValueError("T1 and T2 must be finite")
    if T1 <= 0 or T2 <= 0:
        raise ValueError(f"T1 and T2 must be positive, got T1={T1}, T2={T2}")
    if T2 <= T1:
        raise ValueError(f"T2 must be greater than T1, got T1={T1}, T2={T2}")


def normalize_model_name(model: str) -> ModelName:
    key = model.strip().lower().replace("-", "_")
    if key in {"twoexpf", "two_exp", "double_exp", "double", "double_exponential", "de"}:
        return "twoexpf"
    if key in {"heidlerf", "heidler", "h"}:
        return "heidlerf"
    raise ValueError("model must be 'twoexpf' or 'heidlerf'")


def validate_atp_restrictions(
    model: Union[str, ModelName],
    T1: float,
    T2: float,
    PERC: int = 30,
    n: float = 10.0,
) -> None:
    """Check ATP TYPE-15 fitting restrictions."""
    model = normalize_model_name(str(model))
    PERC = validate_perc(PERC)
    validate_T1_T2(T1, T2)
    ratio = T2 / T1

    if model == "twoexpf":
        min_ratio = TWOEXPF_RATIO_MIN_BY_PERC[PERC]
        if ratio < min_ratio:
            raise ValueError(
                f"TWOEXPF ATP restriction failed for PERC={PERC}: "
                f"T2/T1={ratio:.6g} must be >= {min_ratio}"
            )
        return

    if model == "heidlerf":
        if not (HEIDLERF_RATIO_MIN <= ratio <= HEIDLERF_RATIO_MAX):
            raise ValueError(
                f"HEIDLERF ATP restriction failed: "
                f"{HEIDLERF_RATIO_MIN} <= T2/T1 <= {HEIDLERF_RATIO_MAX}, "
                f"got {ratio:.6g}"
            )
        if n <= 0:
            raise ValueError(f"HEIDLERF n must be positive, got {n}")
        return


def _find_crossing_time(
    t: np.ndarray,
    y: np.ndarray,
    level: float,
    start_index: int,
    stop_index: int,
    rising: bool,
) -> float:
    """Find one crossing time by linear interpolation."""
    if rising:
        indices = range(start_index, stop_index)
        for i in indices:
            if y[i] <= level <= y[i + 1]:
                y0, y1 = y[i], y[i + 1]
                if y1 == y0:
                    return float(t[i])
                frac = (level - y0) / (y1 - y0)
                return float(t[i] + frac * (t[i + 1] - t[i]))
    else:
        indices = range(start_index, stop_index)
        for i in indices:
            if y[i] >= level >= y[i + 1]:
                y0, y1 = y[i], y[i + 1]
                if y1 == y0:
                    return float(t[i])
                frac = (level - y0) / (y1 - y0)
                return float(t[i] + frac * (t[i + 1] - t[i]))

    raise ValueError(f"Could not find crossing at level={level}")


def characteristic_times_from_function(
    raw_func: Callable[[np.ndarray], np.ndarray],
    t_max: float,
    n_points: int = 12000,
) -> WaveformTimes:
    """Extract peak, 10%, 30%, 90%, tail 50%, and virtual-zero times."""
    if t_max <= 0:
        raise ValueError("t_max must be positive")

    t = np.linspace(0.0, t_max, n_points)
    y = np.asarray(raw_func(t), dtype=float)
    y[~np.isfinite(y)] = 0.0
    y[y < 0] = 0.0

    peak_index = int(np.argmax(y))
    peak_value = float(y[peak_index])

    if peak_value <= 0:
        raise ValueError("Waveform peak is not positive")

    if peak_index <= 1 or peak_index >= len(t) - 3:
        raise ValueError("Peak is outside usable time range; increase t_max")

    t_peak = float(t[peak_index])
    t10 = _find_crossing_time(t, y, 0.10 * peak_value, 0, peak_index, rising=True)
    t30 = _find_crossing_time(t, y, 0.30 * peak_value, 0, peak_index, rising=True)
    t90 = _find_crossing_time(t, y, 0.90 * peak_value, 0, peak_index, rising=True)
    t50_tail = _find_crossing_time(
        t, y, 0.50 * peak_value, peak_index, len(t) - 2, rising=False
    )

    # Virtual zero from the straight line through 10%-90% and 30%-90%.
    virtual_zero_10 = t10 - 0.10 * (t90 - t10) / 0.80
    virtual_zero_30 = t30 - 0.30 * (t90 - t30) / 0.60

    return WaveformTimes(
        peak_value=peak_value,
        t_peak=t_peak,
        t10=t10,
        t30=t30,
        t90=t90,
        t50_tail=t50_tail,
        virtual_zero_10=virtual_zero_10,
        virtual_zero_30=virtual_zero_30,
    )


def raw_twoexp_tau(t: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    """Double exponential in tau form: exp(-t/tau1) - exp(-t/tau2)."""
    return np.exp(-t / tau1) - np.exp(-t / tau2)


def raw_twoexp_AB(t: np.ndarray, A: float, B: float) -> np.ndarray:
    """ATP double exponential form: exp(A*t) - exp(B*t)."""
    return np.exp(A * t) - np.exp(B * t)


def raw_heidler(t: np.ndarray, Tf: float, tau: float, n: float) -> np.ndarray:
    """Heidler raw function before peak scaling."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = t >= 0.0
    if not np.any(mask):
        return out

    x = np.zeros_like(t)
    x[mask] = t[mask] / Tf
    # Avoid overflow for large n by clipping x.
    x = np.clip(x, 0.0, 1e30)
    xn = np.power(x, n, where=mask, out=np.zeros_like(t))
    out[mask] = (xn[mask] / (1.0 + xn[mask])) * np.exp(-t[mask] / tau)
    out[~np.isfinite(out)] = 0.0
    return out


def twoexp_peak_time_tau(tau1: float, tau2: float) -> float:
    """Analytical peak time for tau-form double exponential."""
    if tau1 <= tau2 or tau2 <= 0:
        raise ValueError("Require tau1 > tau2 > 0")
    return (tau1 * tau2) / (tau1 - tau2) * log(tau1 / tau2)


def twoexp_peak_time_AB(A: float, B: float) -> float:
    """Analytical peak time for ATP A/B-form double exponential."""
    if not (A < 0 and B < 0 and A > B):
        raise ValueError("Require A < 0, B < 0, and A > B")
    return log(A / B) / (B - A)


def _default_tmax(T1: float, T2: float) -> float:
    return max(8.0 * T2, 20.0 * T1, T2 + 10.0 * T1)


def _require_scipy() -> None:
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for automatic fitting. "
            "Install scipy, or provide direct parameters such as tau1/tau2 or Tf/tau."
        )


def fit_twoexp_params(
    T1: float,
    T2: float,
    PERC: int = 30,
    atp_compatible: bool = True,
) -> FitResult:
    """Fit TWOEXPF tau1/tau2 and ATP A/B from target T1/T2/PERC."""
    _require_scipy()
    PERC = validate_perc(PERC)
    validate_T1_T2(T1, T2)
    if atp_compatible:
        validate_atp_restrictions("twoexpf", T1, T2, PERC)

    t_max = _default_tmax(T1, T2)

    def unpack(p: np.ndarray) -> Tuple[float, float]:
        tau2 = float(np.exp(p[0]))
        tau1 = tau2 * (1.0 + float(np.exp(p[1])))
        return tau1, tau2

    def residual(p: np.ndarray) -> np.ndarray:
        tau1, tau2 = unpack(p)
        try:
            times = characteristic_times_from_function(
                lambda tt: raw_twoexp_tau(tt, tau1, tau2),
                t_max=t_max,
            )
            T1_actual, T2_actual = times.T1_T2_for_perc(PERC)
            return np.array([
                (T1_actual - T1) / T1,
                (T2_actual - T2) / T2,
            ])
        except Exception:
            return np.array([1e3, 1e3])

    # Initial guess: long tau near T2, short tau near T1.
    tau2_guess = max(T1 / 4.0, T1 * 0.05)
    tau1_guess = max(T2, tau2_guess * 2.0)
    p0 = np.array([
        np.log(tau2_guess),
        np.log(max(tau1_guess / tau2_guess - 1.0, 1e-3)),
    ])

    res = least_squares(
        residual,
        p0,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2500,
    )

    tau1, tau2 = unpack(res.x)
    A = -1.0 / tau1
    B = -1.0 / tau2
    err = float(np.linalg.norm(res.fun))

    return FitResult(
        model="twoexpf",
        parameter_source="fitted",
        T1_target=T1,
        T2_target=T2,
        PERC=PERC,
        n=None,
        tau1=tau1,
        tau2=tau2,
        A=A,
        B=B,
        fit_error=err,
        message=res.message,
    )


def fit_heidler_params(
    T1: float,
    T2: float,
    n: float = 10.0,
    PERC: int = 30,
    atp_compatible: bool = True,
) -> FitResult:
    """Fit HEIDLERF Tf/tau from target T1/T2/PERC/n."""
    _require_scipy()
    PERC = validate_perc(PERC)
    validate_T1_T2(T1, T2)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if atp_compatible:
        validate_atp_restrictions("heidlerf", T1, T2, PERC, n=n)

    t_max = _default_tmax(T1, T2)

    def unpack(p: np.ndarray) -> Tuple[float, float]:
        Tf = float(np.exp(p[0]))
        tau = float(np.exp(p[1]))
        return Tf, tau

    def residual(p: np.ndarray) -> np.ndarray:
        Tf, tau = unpack(p)
        try:
            times = characteristic_times_from_function(
                lambda tt: raw_heidler(tt, Tf, tau, n),
                t_max=t_max,
            )
            T1_actual, T2_actual = times.T1_T2_for_perc(PERC)
            return np.array([
                (T1_actual - T1) / T1,
                (T2_actual - T2) / T2,
            ])
        except Exception:
            return np.array([1e3, 1e3])

    p0 = np.array([
        np.log(max(T1 / 2.0, 1e-12)),
        np.log(max(T2, T1 * 2.0)),
    ])

    res = least_squares(
        residual,
        p0,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=3000,
    )

    Tf, tau = unpack(res.x)
    err = float(np.linalg.norm(res.fun))

    return FitResult(
        model="heidlerf",
        parameter_source="fitted",
        T1_target=T1,
        T2_target=T2,
        PERC=PERC,
        n=n,
        Tf=Tf,
        tau=tau,
        fit_error=err,
        message=res.message,
    )


class BaseLightningCurrentSource:
    """Base class for ATP lightning current sources."""

    model: ModelName = "twoexpf"

    def __init__(
        self,
        *,
        peak: float,
        T1: float,
        T2: float,
        PERC: int = 30,
        n: Optional[float] = None,
        Tstart: float = 0.0,
        Tstop: Optional[float] = None,
        description: str = "",
        parameter_source: ParameterSource = "fitted",
        fit_error: Optional[float] = None,
    ) -> None:
        if peak == 0 or not isfinite(peak):
            raise ValueError(f"peak must be finite and non-zero, got {peak}")
        validate_T1_T2(T1, T2)
        self.PERC = validate_perc(PERC)

        if Tstop is not None and Tstop <= Tstart:
            raise ValueError(f"Tstop must be greater than Tstart, got {Tstop} <= {Tstart}")

        self.peak = float(peak)
        self.T1 = float(T1)
        self.T2 = float(T2)
        self.n = None if n is None else float(n)
        self.Tstart = float(Tstart)
        self.Tstop = None if Tstop is None else float(Tstop)
        self.description = description
        self.parameter_source = parameter_source
        self.fit_error = fit_error

        self.ui = ATP_CURRENT_UI

        self.t_peak_raw, self.raw_peak = self._find_raw_peak()
        if self.raw_peak <= 0 or not isfinite(self.raw_peak):
            raise ValueError("Raw waveform peak is not positive/finite")

        self.k_factor = 1.0 / self.raw_peak
        self.times_raw = characteristic_times_from_function(
            lambda tt: self._raw(tt),
            t_max=self._default_characteristic_tmax(),
        )

    def _raw(self, t_rel: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _analytical_peak_time(self) -> Optional[float]:
        return None

    def _default_characteristic_tmax(self) -> float:
        return _default_tmax(self.T1, self.T2)

    def _find_raw_peak(self) -> Tuple[float, float]:
        analytical = self._analytical_peak_time()
        if analytical is not None:
            value = float(self._raw(np.array([analytical]))[0])
            return analytical, value

        _require_scipy()
        t_max = self._default_characteristic_tmax()

        def objective(t: float) -> float:
            return -float(self._raw(np.array([t]))[0])

        res = minimize_scalar(objective, bounds=(0.0, t_max), method="bounded")
        if not res.success:
            raise RuntimeError("Could not locate raw waveform peak")
        t_peak = float(res.x)
        value = float(self._raw(np.array([t_peak]))[0])
        return t_peak, value

    @property
    def t_peak(self) -> float:
        return self.Tstart + self.t_peak_raw

    def current_at(self, t: float) -> float:
        """Return scalar source current at absolute simulation time ``t``."""
        t_rel = float(t) - self.Tstart
        if t_rel < 0.0:
            return 0.0
        if self.Tstop is not None and float(t) > self.Tstop:
            return 0.0
        raw = float(self._raw(np.array([t_rel], dtype=float))[0])
        return float(self.peak) * float(self.k_factor) * raw

    def to_callable(self) -> Callable[[float], float]:
        """Return a scalar callable suitable for EMTPSolver current sources."""
        return lambda t: self.current_at(t)

    def verify_peak(self) -> Tuple[float, float, float]:
        actual = self.peak * self.k_factor * float(
            self._raw(np.array([self.t_peak_raw], dtype=float))[0]
        )
        error = abs(actual - self.peak) / abs(self.peak)
        return self.peak, actual, error

    def get_info(self) -> dict:
        target_peak, actual_peak, peak_error = self.verify_peak()
        T1_actual, T2_actual = self.times_raw.T1_T2_for_perc(self.PERC)

        return {
            "model": self.model.upper(),
            "U/I": self.ui,
            "waveform_type": getattr(self, "waveform_type", None),
            "description": self.description,
            "peak_specified": target_peak,
            "peak_actual": actual_peak,
            "peak_error": peak_error,
            "T1_target": self.T1,
            "T2_target": self.T2,
            "T1_actual_by_PERC": T1_actual,
            "T2_actual_by_PERC": T2_actual,
            "PERC": self.PERC,
            "n": self.n,
            "Tstart": self.Tstart,
            "Tstop": self.Tstop,
            "t_peak": self.t_peak,
            "k_factor": self.k_factor,
            "parameter_source": self.parameter_source,
            "fit_error": self.fit_error,
        }

    def print_info(self) -> None:
        info = self.get_info()
        sep = "-" * 72
        print(sep)
        print(f"ATP lightning current source: {info['model']}   U/I={info['U/I']}")
        if info["description"]:
            print(f"Description: {info['description']}")
        print(sep)
        print(
            f"Peak specified = {info['peak_specified']:.8g} A, "
            f"actual = {info['peak_actual']:.8g} A, "
            f"error = {info['peak_error']:.3e}"
        )
        print(
            f"T1 target = {_fmt_time(info['T1_target'])}, "
            f"actual = {_fmt_time(info['T1_actual_by_PERC'])}"
        )
        print(
            f"T2 target = {_fmt_time(info['T2_target'])}, "
            f"actual = {_fmt_time(info['T2_actual_by_PERC'])}"
        )
        if info["waveform_type"] is not None:
            print(f"waveform_type = {info['waveform_type']}")
        print(f"PERC = {info['PERC']}, n = {info['n']}")
        print(f"Tstart = {_fmt_time(info['Tstart'])}, Tstop = {info['Tstop']}")
        print(f"t_peak = {_fmt_time(info['t_peak'])}, K = {info['k_factor']:.10g}")
        print(f"parameter_source = {info['parameter_source']}, fit_error = {info['fit_error']}")
        self._print_model_parameters()
        print(sep)

    def _print_model_parameters(self) -> None:
        pass


class TWOEXPFCurrentSource(BaseLightningCurrentSource):
    """ATP TWOEXPF double-exponential lightning current source."""

    model: ModelName = "twoexpf"

    def __init__(
        self,
        *,
        peak: float,
        T1: float,
        T2: float,
        PERC: int = 30,
        Tstart: float = 0.0,
        Tstop: Optional[float] = None,
        waveform_type: Optional[str] = None,
        tau1: Optional[float] = None,
        tau2: Optional[float] = None,
        A: Optional[float] = None,
        B: Optional[float] = None,
        atp_compatible: bool = True,
        description: str = "",
    ) -> None:
        validate_T1_T2(T1, T2)
        PERC = validate_perc(PERC)

        if waveform_type is not None:
            if any(value is not None for value in (tau1, tau2, A, B)):
                raise ValueError(
                    "waveform_type cannot be combined with direct tau1/tau2 or A/B parameters"
                )
            tau1_std, tau2_std, std_description = get_standard_waveform_params(waveform_type)
            self.waveform_type = normalize_standard_waveform_name(waveform_type)
            self.tau1 = float(tau1_std)
            self.tau2 = float(tau2_std)
            self.A = -1.0 / self.tau1
            self.B = -1.0 / self.tau2
            parameter_source: ParameterSource = "standard"
            fit_error = None
            if not description:
                description = std_description

        elif (A is not None) or (B is not None):
            if A is None or B is None:
                raise ValueError("A and B must be provided together")
            if not (A < 0 and B < 0 and A > B):
                raise ValueError("Require A < 0, B < 0, and A > B")
            self.waveform_type = "custom"
            self.A = float(A)
            self.B = float(B)
            self.tau1 = -1.0 / self.A
            self.tau2 = -1.0 / self.B
            parameter_source: ParameterSource = "direct"
            fit_error = None

        elif (tau1 is not None) or (tau2 is not None):
            if tau1 is None or tau2 is None:
                raise ValueError("tau1 and tau2 must be provided together")
            if not (tau1 > tau2 > 0):
                raise ValueError("Require tau1 > tau2 > 0")
            self.waveform_type = "custom"
            self.tau1 = float(tau1)
            self.tau2 = float(tau2)
            self.A = -1.0 / self.tau1
            self.B = -1.0 / self.tau2
            parameter_source = "direct"
            fit_error = None

        else:
            fit = fit_twoexp_params(
                T1=T1,
                T2=T2,
                PERC=PERC,
                atp_compatible=atp_compatible,
            )
            self.waveform_type = "fitted"
            self.tau1 = float(fit.tau1)
            self.tau2 = float(fit.tau2)
            self.A = float(fit.A)
            self.B = float(fit.B)
            parameter_source = "fitted"
            fit_error = fit.fit_error

        super().__init__(
            peak=peak,
            T1=T1,
            T2=T2,
            PERC=PERC,
            n=None,
            Tstart=Tstart,
            Tstop=Tstop,
            description=description,
            parameter_source=parameter_source,
            fit_error=fit_error,
        )

    def _raw(self, t_rel: np.ndarray) -> np.ndarray:
        return raw_twoexp_AB(t_rel, self.A, self.B)

    def _analytical_peak_time(self) -> Optional[float]:
        return twoexp_peak_time_AB(self.A, self.B)

    def _print_model_parameters(self) -> None:
        print(f"A = {self.A:.10g} 1/s, B = {self.B:.10g} 1/s")
        print(f"tau1 = {_fmt_time(self.tau1)}, tau2 = {_fmt_time(self.tau2)}")



class HEIDLERFCurrentSource(BaseLightningCurrentSource):
    """ATP HEIDLERF lightning current source."""

    model: ModelName = "heidlerf"

    def __init__(
        self,
        *,
        peak: float,
        T1: float,
        T2: float,
        n: float = 10.0,
        PERC: int = 30,
        Tstart: float = 0.0,
        Tstop: Optional[float] = None,
        Tf: Optional[float] = None,
        tau: Optional[float] = None,
        atp_compatible: bool = True,
        description: str = "",
    ) -> None:
        validate_T1_T2(T1, T2)
        PERC = validate_perc(PERC)
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        if (Tf is not None) or (tau is not None):
            if Tf is None or tau is None:
                raise ValueError("Tf and tau must be provided together")
            if Tf <= 0 or tau <= 0:
                raise ValueError("Tf and tau must be positive")
            self.Tf = float(Tf)
            self.tau = float(tau)
            parameter_source: ParameterSource = "direct"
            fit_error = None
        else:
            fit = fit_heidler_params(
                T1=T1,
                T2=T2,
                n=n,
                PERC=PERC,
                atp_compatible=atp_compatible,
            )
            self.Tf = float(fit.Tf)
            self.tau = float(fit.tau)
            parameter_source = "fitted"
            fit_error = fit.fit_error

        super().__init__(
            peak=peak,
            T1=T1,
            T2=T2,
            PERC=PERC,
            n=n,
            Tstart=Tstart,
            Tstop=Tstop,
            description=description,
            parameter_source=parameter_source,
            fit_error=fit_error,
        )

    def _raw(self, t_rel: np.ndarray) -> np.ndarray:
        return raw_heidler(t_rel, self.Tf, self.tau, float(self.n))

    def _print_model_parameters(self) -> None:
        print(f"Tf = {_fmt_time(self.Tf)}, tau = {_fmt_time(self.tau)}")
        print(f"K = {self.k_factor:.10g}, eta = {1.0 / self.k_factor:.10g}")




def create_twoexpf_current_source(
    *,
    peak: float,
    T1: float,
    T2: float,
    PERC: int = 30,
    Tstart: float = 0.0,
    Tstop: Optional[float] = None,
    waveform_type: Optional[str] = None,
    tau1: Optional[float] = None,
    tau2: Optional[float] = None,
    A: Optional[float] = None,
    B: Optional[float] = None,
    atp_compatible: bool = True,
    description: str = "",
) -> TWOEXPFCurrentSource:
    """Create an ATP TWOEXPF lightning current source."""
    return TWOEXPFCurrentSource(
        peak=peak,
        T1=T1,
        T2=T2,
        PERC=PERC,
        Tstart=Tstart,
        Tstop=Tstop,
        waveform_type=waveform_type,
        tau1=tau1,
        tau2=tau2,
        A=A,
        B=B,
        atp_compatible=atp_compatible,
        description=description,
    )


def create_heidlerf_current_source(
    *,
    peak: float,
    T1: float,
    T2: float,
    n: float = 10.0,
    PERC: int = 30,
    Tstart: float = 0.0,
    Tstop: Optional[float] = None,
    Tf: Optional[float] = None,
    tau: Optional[float] = None,
    atp_compatible: bool = True,
    description: str = "",
) -> HEIDLERFCurrentSource:
    """Create an ATP HEIDLERF lightning current source."""
    return HEIDLERFCurrentSource(
        peak=peak,
        T1=T1,
        T2=T2,
        n=n,
        PERC=PERC,
        Tstart=Tstart,
        Tstop=Tstop,
        Tf=Tf,
        tau=tau,
        atp_compatible=atp_compatible,
        description=description,
    )


def create_lightning_current_source(
    *,
    model: Union[str, ModelName],
    peak: float,
    T1: float,
    T2: float,
    n: float = 10.0,
    PERC: int = 30,
    Tstart: float = 0.0,
    Tstop: Optional[float] = None,
    atp_compatible: bool = True,
    description: str = "",
    # Standard TWOEXPF waveform library
    waveform_type: Optional[str] = None,
    # TWOEXPF direct parameters
    tau1: Optional[float] = None,
    tau2: Optional[float] = None,
    A: Optional[float] = None,
    B: Optional[float] = None,
    # HEIDLERF direct parameters
    Tf: Optional[float] = None,
    tau: Optional[float] = None,
) -> Union[TWOEXPFCurrentSource, HEIDLERFCurrentSource]:
    """Unified factory for lightning current sources."""
    model_name = normalize_model_name(str(model))

    if model_name == "twoexpf":
        return create_twoexpf_current_source(
            peak=peak,
            T1=T1,
            T2=T2,
            PERC=PERC,
            Tstart=Tstart,
            Tstop=Tstop,
            waveform_type=waveform_type,
            tau1=tau1,
            tau2=tau2,
            A=A,
            B=B,
            atp_compatible=atp_compatible,
            description=description,
        )

    return create_heidlerf_current_source(
        peak=peak,
        T1=T1,
        T2=T2,
        n=n,
        PERC=PERC,
        Tstart=Tstart,
        Tstop=Tstop,
        Tf=Tf,
        tau=tau,
        atp_compatible=atp_compatible,
        description=description,
    )



def create_standard_twoexpf_current_source(
    *,
    waveform_type: str,
    peak: float,
    PERC: int = 30,
    Tstart: float = 0.0,
    Tstop: Optional[float] = None,
    atp_compatible: bool = True,
    description: str = "",
) -> TWOEXPFCurrentSource:
    """Create a TWOEXPF current source from the standard waveform library.

    This skips numerical fitting by using stored tau1/tau2 parameters directly.
    Nominal T1/T2 are parsed from waveform_type, for example "8/20" -> 8 us / 20 us.
    """
    T1, T2 = parse_standard_waveform_T1_T2(waveform_type)
    return create_twoexpf_current_source(
        peak=peak,
        T1=T1,
        T2=T2,
        PERC=PERC,
        Tstart=Tstart,
        Tstop=Tstop,
        waveform_type=waveform_type,
        atp_compatible=atp_compatible,
        description=description,
    )


# Backward-compatible aliases.
create_atp_surge_source = create_lightning_current_source
create_lightning_source = create_lightning_current_source
LightningWaveform = TWOEXPFCurrentSource


def _fmt_time(value: Optional[float]) -> str:
    if value is None:
        return "None"
    v = float(value)
    av = abs(v)
    if av == 0:
        return "0 s"
    if av < 1e-6:
        return f"{v * 1e9:.6g} ns"
    if av < 1e-3:
        return f"{v * 1e6:.6g} us"
    if av < 1:
        return f"{v * 1e3:.6g} ms"
    return f"{v:.6g} s"


__all__ = [
    "ATP_CURRENT_UI",
    "VALID_PERC",
    "STANDARD_DOUBLE_EXPONENTIAL_PARAMS",
    "StandardWaveformName",
    "TWOEXPF_RATIO_MIN_BY_PERC",
    "HEIDLERF_RATIO_MIN",
    "HEIDLERF_RATIO_MAX",
    "WaveformTimes",
    "FitResult",
    "TWOEXPFCurrentSource",
    "HEIDLERFCurrentSource",
    "create_twoexpf_current_source",
    "create_heidlerf_current_source",
    "create_lightning_current_source",
    "create_standard_twoexpf_current_source",
    "list_standard_waveforms",
    "get_standard_waveform_params",
    "parse_standard_waveform_T1_T2",
    "create_atp_surge_source",
    "create_lightning_source",
    "LightningWaveform",
    "fit_twoexp_params",
    "fit_heidler_params",
    "validate_atp_restrictions",
]


if __name__ == "__main__":
    print("Smoke test: HEIDLERF 10/350 us, Peak=200 kA, default PERC=30")
    src_h = create_heidlerf_current_source(
        peak=200e3,
        T1=10e-6,
        T2=350e-6,
        n=10,
        # PERC omitted: default is 30
        atp_compatible=True,
        description="IEC positive first stroke",
    )
    src_h.print_info()

    print("\nSmoke test: TWOEXPF 10/350 us, Peak=30 kA, default PERC=30")
    src_t = create_twoexpf_current_source(
        peak=30e3,
        T1=10e-6,
        T2=350e-6,
        # PERC omitted: default is 30
        atp_compatible=True,
        description="Double exponential example",
    )
    src_t.print_info()

    print("\nSmoke test: standard TWOEXPF 8/20 us, Peak=30 kA, fitting skipped")
    src_std = create_standard_twoexpf_current_source(
        waveform_type="8/20",
        peak=30e3,
        # PERC omitted: default is 30
        atp_compatible=False,
    )
    src_std.print_info()
