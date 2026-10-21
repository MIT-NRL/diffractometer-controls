"""Qt-free, explicit peak fitting. No acquisition, catalog or site dependencies."""

from dataclasses import asdict, dataclass

import numpy as np
from lmfit.models import ConstantModel, GaussianModel, LinearModel, LorentzianModel, PseudoVoigtModel, VoigtModel
from scipy.signal import find_peaks

PEAK_MODELS = {'Gaussian': GaussianModel, 'Lorentzian': LorentzianModel,
               'Voigt': VoigtModel, 'Pseudo-Voigt': PseudoVoigtModel}
BACKGROUNDS = ('None', 'Constant', 'Linear')


@dataclass(frozen=True)
class FitConfig:
    peak: str = 'Gaussian'
    background: str = 'Constant'
    peaks: int = 1
    xmin: float | None = None
    xmax: float | None = None
    weighting: str = 'Equal'
    centers: tuple[float, ...] = ()


def fit_curve(x, y, config=FitConfig()):
    """Fit positive peaks in an inclusive window, returning arrays and diagnostics.

    Amplitude is integrated peak area; sigma depends on the selected profile.
    Residuals are unweighted data minus fit. Poisson weights are 1/sqrt(max(y,1))
    and are available only for nonnegative count data. Uncertainties use lmfit's
    covariance estimate and may be unavailable for poorly constrained fits.
    """
    if config.peak not in PEAK_MODELS or config.background not in BACKGROUNDS:
        raise ValueError('Choose a supported peak and background model.')
    if not 1 <= config.peaks <= 6:
        raise ValueError('Choose between one and six peaks.')
    if config.weighting not in {'Equal', 'Poisson (counts)'}:
        raise ValueError('Unsupported weighting.')
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError('x and y must be matching one-dimensional arrays.')
    finite = np.isfinite(x) & np.isfinite(y)
    lo = float(config.xmin) if config.xmin is not None else (float(np.min(x[finite])) if finite.any() else 0)
    hi = float(config.xmax) if config.xmax is not None else (float(np.max(x[finite])) if finite.any() else 0)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError('The fit window minimum must be below its maximum.')
    mask = finite & (x >= lo) & (x <= hi)
    x, y = x[mask], y[mask]
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    if x.size < 7 or np.unique(x).size < 7:
        raise ValueError('At least seven distinct finite x values are required in the fit window.')
    if np.ptp(y) <= np.finfo(float).eps * max(1, abs(float(np.mean(y)))):
        raise ValueError('The selected data are constant; there is no peak to fit.')
    if config.weighting == 'Poisson (counts)' and np.any(y < 0):
        raise ValueError('Poisson weighting requires nonnegative raw counts.')
    span = float(np.ptp(x))
    dx = float(np.min(np.diff(np.unique(x))))
    edge = max(2, min(len(x) // 5, 20))
    ex, ey = np.r_[x[:edge], x[-edge:]], np.r_[y[:edge], y[-edge:]]
    slope, intercept = np.polyfit(ex, ey, 1)
    if config.background == 'None':
        baseline = np.zeros_like(y)
        model = None
        params = None
    elif config.background == 'Constant':
        model = ConstantModel(prefix='bg_')
        value = float(np.median(ey))
        params = model.make_params(c=value)
        baseline = np.full_like(y, value)
    else:
        model = LinearModel(prefix='bg_')
        params = model.make_params(slope=float(slope), intercept=float(intercept))
        baseline = slope * x + intercept
    signal = y - baseline
    if config.centers:
        centers = sorted(float(value) for value in config.centers)
        if len(centers) != config.peaks or not np.isfinite(centers).all():
            raise ValueError('Supply one finite starting center for each peak.')
        if len(set(centers)) != len(centers) or centers[0] <= lo or centers[-1] >= hi:
            raise ValueError('Starting centers must be distinct and strictly inside the fit window.')
    else:
        candidates, _ = find_peaks(signal, distance=max(1, len(x) // (config.peaks * 4)))
        candidates = sorted(candidates, key=lambda i: signal[i], reverse=True)
        if len(candidates) < config.peaks:
            if config.peaks > 1:
                raise ValueError('Not enough separated peaks for an initial estimate; supply starting centers.')
            candidates = [int(np.argmax(signal))]
        centers = sorted(float(x[i]) for i in candidates[:config.peaks])
    boundaries = [lo, *[(a + b) / 2 for a, b in zip(centers, centers[1:])], hi]
    for index, center in enumerate(centers):
        peak = PEAK_MODELS[config.peak](prefix=f'p{index + 1}_')
        guess = peak.guess(np.maximum(signal, 0), x=x)
        width = min(max(float(guess[f'p{index + 1}_sigma'].value), dx), span / (2 * config.peaks))
        guess[f'p{index + 1}_center'].set(value=center, min=boundaries[index], max=boundaries[index + 1])
        guess[f'p{index + 1}_sigma'].set(value=width, min=max(dx * 0.1, span * 1e-8), max=span)
        guess[f'p{index + 1}_amplitude'].set(value=max(float(guess[f'p{index + 1}_amplitude'].value) / config.peaks, 1e-12), min=0)
        if config.peak == 'Voigt':
            guess[f'p{index + 1}_gamma'].set(value=width, expr='', vary=True, min=0, max=span)
        if model is None:
            model, params = peak, guess
        else:
            model += peak
            params.update(guess)
    nvarys = sum(parameter.vary and not parameter.expr for parameter in params.values())
    if len(x) <= nvarys + 1:
        raise ValueError(f'The window needs more than {nvarys + 1} data points for this model.')
    weights = None if config.weighting == 'Equal' else 1 / np.sqrt(np.maximum(y, 1))
    result = model.fit(y, params, x=x, weights=weights, nan_policy='raise', max_nfev=6000)
    if not result.success or not np.isfinite(result.best_fit).all():
        raise RuntimeError(f'Fit did not converge: {result.message}')
    parameters = {name: {'value': float(p.value), 'stderr': float(p.stderr) if p.stderr is not None and np.isfinite(p.stderr) else None,
                         'vary': bool(p.vary and not p.expr)} for name, p in result.params.items()}
    return {'config': asdict(config), 'x': x, 'y': y, 'fit': result.best_fit,
            'residual': y - result.best_fit, 'components': result.eval_components(x=x),
            'parameters': parameters, 'redchi': float(result.redchi), 'aic': float(result.aic),
            'bic': float(result.bic), 'ndata': int(result.ndata), 'nvarys': int(result.nvarys),
            'message': str(result.message), 'uncertainties_available': bool(result.errorbars),
            'report': result.fit_report()}
