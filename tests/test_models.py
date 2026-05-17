"""Tests for ``drlib.models`` (dS21, Lorentzian)."""

from __future__ import annotations

import numpy as np
import pytest

from drlib.models import GAMMA, MU0, Lorentzian, dS21


# ----------------------------------------------------------------------
# dS21 — shape, dtype, antisymmetry
# ----------------------------------------------------------------------
def test_dS21_returns_real_array_of_same_shape():
    f = np.linspace(2e9, 5e9, 401)
    y = dS21(f, A=500.0, Psi=-0.97, fres=3.5e9, Df=0.1e9,
             mod=1e-3, Msat=0.2, H0=0.05)
    assert y.shape == f.shape
    assert np.isrealobj(y)


def test_dS21_has_resonance_near_fres():
    """The derivative line crosses zero close to fres."""
    f = np.linspace(2e9, 5e9, 4001)
    fres = 3.5e9
    y = dS21(f, A=500.0, Psi=0.0, fres=fres, Df=0.05e9,
             mod=1e-3, Msat=0.2, H0=0.05)
    # Pick the zero crossing closest to fres (within ±200 MHz)
    mask = (f > fres - 2e8) & (f < fres + 2e8)
    yy = y[mask]; ff = f[mask]
    sign_change = np.where(np.diff(np.sign(yy)) != 0)[0]
    assert len(sign_change) >= 1
    nearest_zero = ff[sign_change[np.argmin(np.abs(ff[sign_change] - fres))]]
    assert abs(nearest_zero - fres) < 5e7   # within 50 MHz


# ----------------------------------------------------------------------
# Lorentzian — shape, peak position, mod is ignored
# ----------------------------------------------------------------------
def test_lorentzian_returns_real_array_of_same_shape():
    f = np.linspace(2e9, 5e9, 401)
    y = Lorentzian(f, A=500.0, Psi=0.0, fres=3.5e9, Df=0.1e9,
                   mod=0.0, Msat=0.2, H0=0.05)
    assert y.shape == f.shape
    assert np.isrealobj(y)


def test_lorentzian_peaks_near_fres():
    """The Lorentzian magnitude peaks within an HWHM of fres."""
    f = np.linspace(2e9, 5e9, 4001)
    fres = 3.5e9
    Df = 0.05e9
    y = Lorentzian(f, A=500.0, Psi=0.0, fres=fres, Df=Df,
                   mod=0.0, Msat=0.2, H0=0.05)
    peak_f = f[np.argmax(np.abs(y))]
    assert abs(peak_f - fres) < Df


def test_lorentzian_mod_is_ignored():
    """`mod` is accepted only for signature parity with dS21."""
    f = np.linspace(2e9, 5e9, 401)
    kw = dict(A=1.0, Psi=0.3, fres=3.5e9, Df=0.1e9,
              Msat=0.2, H0=0.05)
    y0 = Lorentzian(f, mod=0.0, **kw)
    y1 = Lorentzian(f, mod=99.0, **kw)
    assert np.allclose(y0, y1)


# ----------------------------------------------------------------------
# Physical constants — make sure the canonical values haven't drifted
# ----------------------------------------------------------------------
def test_constants_are_canonical():
    assert MU0 == pytest.approx(4.0 * np.pi * 1e-7, rel=1e-12)
    assert GAMMA == pytest.approx(1.76085963023e11, rel=1e-12)
