"""
drlib.math_tools
================

Numerical helpers shared by the FMR analysis pipeline:

* :func:`derivative_divide` – background-correcting numerical derivative
  used to clean broadband FMR spectra (Maier-Flaig et al., arXiv:1705.05694).
* :func:`derivative`        – plain central-difference derivative along an axis.
* :func:`FFT_1D`            – convenience 1-D FFT with optional zero-padding
  and a shifted, angular-wavenumber ``k`` axis.

All functions operate on plain NumPy arrays and have no side effects.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Derivative-divide background correction
# ----------------------------------------------------------------------
def derivative_divide(
    X: np.ndarray = None,
    Y: np.ndarray = None,
    Z: np.ndarray = None,
    axis: int = 0,
    modulation_amp: int = 1,
    average: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Derivative divided by the non-derivative value (a background correction).

    Performs a numerical derivative of ``Z`` along ``axis`` and divides each
    row by the *non-derivative* value, suppressing slowly-varying multiplicative
    backgrounds. This is the standard "dd" method used in broadband FMR
    spectroscopy: if the raw signal has the form

    .. math::

        S(f, H) = \frac{V_o(f) + Z\, V_o(f)\, \chi(H)}{V_i}\, e^{i\phi},

    then

    .. math::

        \mathrm{dd}(S) = Z\,
        \frac{\chi(H+\Delta H) - \chi(H-\Delta H)}{\Delta H}
        + \mathcal{O}(Z^2),

    so the (frequency-dependent) phase :math:`\phi` and the cable transmission
    :math:`V_o(f)` drop out.

    A simple boxcar smoothing along ``axis`` can additionally be applied by
    setting ``modulation_amp > 1`` together with ``average=True``.

    Parameters
    ----------
    X : ndarray, shape (N, M)
        Independent variable on the M-axis (e.g. frequency).
    Y : ndarray, shape (N, M)
        Independent variable on the N-axis (e.g. magnetic field).
    Z : ndarray, shape (N, M)
        The 2-D signal to be processed.
    axis : {0, 1}, optional
        Axis along which the derivative is computed. Default ``0``.
    modulation_amp : int, optional
        Step width of the central difference. Default ``1``.
    average : bool, optional
        If ``True`` (default), Z is also averaged over
        ``modulation_amp`` neighbouring rows / columns to reduce noise.

    Returns
    -------
    X, Y : ndarray
        The original X and Y arrays, returned unchanged so chained
        operations can keep their context.
    G : ndarray, shape (N, M)
        The derivative-divide-processed signal, same shape as ``Z``.
        Rows within ``modulation_amp`` of the edge are zero.
    d : ndarray
        The last-used step-width vector (average of ``delta`` over the
        modulation window). Returned for diagnostics.

    References
    ----------
    .. [1] L. Maier-Flaig *et al.*,
       "Analysis of broadband ferromagnetic resonance in the frequency domain",
       arXiv:1705.05694.
    """
    if axis == 0:
        delta = np.diff(X, axis=axis)
    elif axis == 1:
        Z = Z.T
        delta = np.diff(Y, axis=axis).T
    else:
        raise ValueError("Only two-dimensional datasets are supported (axis must be 0 or 1).")

    G = np.zeros_like(Z)
    for row in np.arange(modulation_amp, np.shape(Z)[0] - modulation_amp):
        if average:
            zl = np.mean(Z[row - modulation_amp:row, :], axis=0)
            zh = np.mean(Z[row:row + modulation_amp + 1, :], axis=0)
        else:
            zl = Z[row - modulation_amp, :]
            zh = Z[row + modulation_amp, :]
        zm = Z[row, :]
        d = np.mean(delta[row - modulation_amp:row + modulation_amp, :], axis=0)
        G[row, :] = (zh - zl) / zm / d

    if axis == 1:
        G = G.T

    return X, Y, G, d


# ----------------------------------------------------------------------
# Plain numerical derivative
# ----------------------------------------------------------------------
def derivative(
    X: np.ndarray = None,
    Y: np.ndarray = None,
    Z: np.ndarray = None,
    axis: int = 0,
    modulation_amp: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plain central-difference derivative of a 2-D array along ``axis``.

    Companion to :func:`derivative_divide` for use cases that do not need the
    "divide" step (e.g. when the multiplicative background has already been
    removed by other means).

    Parameters
    ----------
    X, Y, Z : ndarray, shape (N, M)
        Same convention as :func:`derivative_divide`.
    axis : {0, 1}, optional
        Axis along which to differentiate. Default ``0``.
    modulation_amp : int, optional
        Step width of the central difference. Default ``1``.

    Returns
    -------
    X, Y : ndarray
        Returned unchanged.
    G : ndarray
        The derivative of ``Z`` along ``axis``.
    d : ndarray
        Step width used for the last row.
    """
    if axis == 0:
        delta = np.diff(X, axis=axis)
    elif axis == 1:
        Z = Z.T
        delta = np.diff(Y, axis=axis).T
    else:
        raise ValueError("Only two-dimensional datasets are supported (axis must be 0 or 1).")

    G = np.zeros_like(Z)
    for row in np.arange(modulation_amp, np.shape(Z)[0] - modulation_amp):
        zl = Z[row - modulation_amp, :]
        zh = Z[row, :]
        d = np.mean(delta[row - modulation_amp:row + modulation_amp, :], axis=0)
        G[row, :] = (zh - zl) / d

    if axis == 1:
        G = G.T

    return X, Y, G, d


# ----------------------------------------------------------------------
# 1-D FFT with zero-padding and angular-wavenumber axis
# ----------------------------------------------------------------------
def FFT_1D(
    array: np.ndarray,
    dx: float,
    zero_pad: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""1-D FFT with FFT-shifted output and an angular-wavenumber axis.

    Parameters
    ----------
    array : ndarray, shape (N,)
        Input signal (real or complex).
    dx : float
        Sample spacing in **micrometres** (µm).  Internally converted to
        metres for the wavenumber axis.
    zero_pad : int, optional
        If given, the input is zero-padded up to ``zero_pad`` samples
        before the FFT. Useful for finer spectral resolution / smoother
        plots.

    Returns
    -------
    spectrum : ndarray, complex
        FFT-shifted spectrum (DC at the centre).
    k : ndarray
        Angular wavenumber axis (rad / m). DC at the centre.
    dk : float
        Step size of ``k`` (rad / m).
    """
    if zero_pad:
        spectrum = np.fft.fftshift(np.fft.fft(array, n=zero_pad))
    else:
        spectrum = np.fft.fftshift(np.fft.fft(array))

    spectrum_abs = np.abs(spectrum)
    stepx = dx * 1e-6  # µm -> m
    k = np.fft.fftshift(np.fft.fftfreq(len(spectrum_abs), stepx)) * 2 * np.pi
    dk = np.abs(k[0] - k[1])

    return spectrum, k, dk
