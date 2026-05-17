"""Tests for ``drlib.math_tools`` (derivative, derivative_divide, FFT_1D)."""

from __future__ import annotations

import numpy as np
import pytest

from drlib.math_tools import FFT_1D, derivative, derivative_divide


# ----------------------------------------------------------------------
# derivative — central-difference of a known function
# ----------------------------------------------------------------------
def test_derivative_recovers_linear_slope():
    """Central-difference derivative of a linear ramp is the slope."""
    N_row, N_col = 21, 5
    X = np.tile(np.arange(N_row)[:, None], (1, N_col)).astype(float)
    Y = np.tile(np.arange(N_col)[None, :], (N_row, 1)).astype(float)
    Z = 3.0 * X + 7.0  # slope = 3 along axis 0
    _, _, G, _ = derivative(X=X, Y=Y, Z=Z, axis=0, modulation_amp=1)
    # Interior rows should reproduce slope = 3
    assert np.allclose(G[1:-1, :], 3.0, atol=1e-9)
    # Boundary rows are left at zero (documented behaviour)
    assert np.allclose(G[0, :], 0.0) and np.allclose(G[-1, :], 0.0)


def test_derivative_axis_validation():
    X = np.zeros((4, 4)); Y = np.zeros((4, 4)); Z = np.zeros((4, 4))
    with pytest.raises(ValueError):
        derivative(X=X, Y=Y, Z=Z, axis=2)


# ----------------------------------------------------------------------
# derivative_divide
# ----------------------------------------------------------------------
def test_derivative_divide_removes_multiplicative_background():
    """If Z = bg(f) * (1 + small chi(H)), dd suppresses bg."""
    N_field, N_freq = 41, 200
    Field = np.tile(np.arange(N_field)[:, None], (1, N_freq)).astype(float)
    Freq = np.tile(np.linspace(1e9, 5e9, N_freq)[None, :], (N_field, 1))
    bg = 0.5 + 0.4 * (Freq / Freq.max())                   # smooth in f, constant in H
    H_axis = np.arange(N_field)[:, None]
    chi = 0.01 * np.sin(2 * np.pi * H_axis / N_field)       # purely field-modulated
    Z = bg * (1.0 + chi)
    _, _, G, _ = derivative_divide(X=Field, Y=Freq, Z=Z, axis=0, modulation_amp=1)
    # Interior rows: dd ≈ d(chi)/dH (frequency-independent) since bg drops out.
    interior = G[5:-5, :]
    rel_spread = (interior.max(axis=1) - interior.min(axis=1)) / (
        np.abs(interior).max(axis=1) + 1e-12
    )
    # Frequency-axis spread of dd should be tiny — bg has been suppressed.
    assert np.median(rel_spread) < 0.05


def test_derivative_divide_is_zero_for_constant_Z():
    N_row, N_col = 10, 5
    X = np.tile(np.arange(N_row)[:, None], (1, N_col)).astype(float)
    Y = np.zeros((N_row, N_col))
    Z = np.full((N_row, N_col), 0.42)
    _, _, G, _ = derivative_divide(X=X, Y=Y, Z=Z, axis=0, modulation_amp=1)
    assert np.allclose(G, 0.0, atol=1e-12)


# ----------------------------------------------------------------------
# FFT_1D
# ----------------------------------------------------------------------
def test_fft_1d_recovers_single_tone():
    """A sinusoid produces a delta-like peak at its wavenumber."""
    N = 4096
    L_um = 100.0                  # 100 µm window
    dx_um = L_um / N              # µm per sample
    k_um1 = 0.3                   # 0.3 rad/µm
    x_um = np.arange(N) * dx_um
    sig = np.cos(k_um1 * x_um)
    spec, k_rad_per_m, _ = FFT_1D(sig, dx_um, zero_pad=4 * N)
    # k is returned in rad/m → convert to rad/µm for comparison
    k_um = k_rad_per_m * 1e-6
    peak_k = k_um[np.argmax(np.abs(spec))]
    assert abs(abs(peak_k) - k_um1) < 0.01


def test_fft_1d_zero_pad_changes_resolution():
    sig = np.cos(2 * np.pi * np.arange(64) / 64)
    s_short, k_short, _ = FFT_1D(sig, dx=1.0)
    s_long, k_long, _ = FFT_1D(sig, dx=1.0, zero_pad=512)
    assert len(s_short) == 64
    assert len(s_long) == 512
    assert len(k_long) > len(k_short)
