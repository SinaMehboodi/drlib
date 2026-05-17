"""Tests for ``drlib.Linecut``: extraction, fitting, denoising."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_data, scan_index_at_field


# ----------------------------------------------------------------------
# get_cut: slice has the right shape and field value
# ----------------------------------------------------------------------
def test_get_cut_shape_and_h0(synthetic_arrays):
    from drlib import Linecut, Spectrum
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    sp = Spectrum.from_arrays(f, B, mlin, mlin_ref,
                              saturation=B.size - 1,
                              derivative_divide=True, warning=False)
    cut_idx = B.size // 2
    cut = Linecut(cut=cut_idx, Spectrum=sp, frequencies=(1.5e9, 5e9))

    assert cut.Freq_cut.shape == cut.S21dd_cut.shape
    assert cut.H0 == B[cut_idx]
    assert cut.Freq_cut.min() >= 1.5e9 and cut.Freq_cut.max() <= 5e9


# ----------------------------------------------------------------------
# one_peak — dS21 fit on the bundled real data
# ----------------------------------------------------------------------
@needs_data
def test_one_peak_dS21_fit_recovers_resonance(spectrum_dd):
    from drlib import Linecut

    cut_idx = scan_index_at_field(spectrum_dd, 80)   # B ≈ 80 mT
    cut = Linecut(cut=cut_idx, Spectrum=spectrum_dd, frequencies=(3.5e9, 4.7e9))
    _, fit, params, _ = cut.one_peak(resonance=[4.2e9])

    assert fit.shape == cut.Freq_cut.shape
    fres = params[2]
    Df = params[3]
    assert 4.0e9 < fres < 4.5e9, f"fres out of range: {fres:.3e}"
    assert 0 < Df < 0.5e9


# ----------------------------------------------------------------------
# one_peak — Lorentzian fit on a derivative spectrum
# ----------------------------------------------------------------------
@needs_data
def test_one_peak_Lorentzian_fit_recovers_resonance(raw_arrays):
    from drlib import Linecut, Spectrum

    freq, field, mlin, mlin_ref = raw_arrays
    sp = Spectrum.from_arrays(freq, field, mlin, mlin_ref,
                              saturation=int(field.size - 1),
                              derivative_divide=False,
                              derivative=True, warning=False)
    cut_idx = scan_index_at_field(sp, 80)
    cut = Linecut(cut=cut_idx, Spectrum=sp, frequencies=(3.5e9, 4.7e9))
    _, fit, params, _ = cut.one_peak(resonance=[4.2e9])

    assert fit.shape == cut.Freq_cut.shape
    fres = params[2]
    assert 4.0e9 < fres < 4.5e9, f"fres out of range: {fres:.3e}"


# ----------------------------------------------------------------------
# Synthetic Lorentzian: full pipeline finds the planted resonance
# ----------------------------------------------------------------------
def test_synthetic_one_peak_finds_planted_resonance(synthetic_arrays):
    from drlib import Linecut, Spectrum

    f, B, mlin, mlin_ref, f_res_vec, Df = synthetic_arrays
    sp = Spectrum.from_arrays(f, B, mlin, mlin_ref,
                              saturation=B.size - 1,
                              derivative_divide=True, warning=False)
    cut_idx = B.size // 2
    f_planted = f_res_vec[cut_idx]
    cut = Linecut(cut=cut_idx, Spectrum=sp,
                  frequencies=(max(1e9, f_planted - 1e9),
                               min(6e9, f_planted + 1e9)))
    _, _, params, _ = cut.one_peak(resonance=[float(f_planted)])
    # Recovered resonance should be within a few HWHM of the planted one.
    assert abs(params[2] - f_planted) < 5 * Df


# ----------------------------------------------------------------------
# two_peak — runs and returns 14 parameters
# ----------------------------------------------------------------------
@needs_data
def test_two_peak_returns_14_params(spectrum_dd):
    from drlib import Linecut

    cut_idx = scan_index_at_field(spectrum_dd, 80)
    cut = Linecut(cut=cut_idx, Spectrum=spectrum_dd, frequencies=(2.5e9, 4.7e9))
    _, fit, params = cut.two_peak(resonance=(3.0e9, 4.3e9))
    assert fit.shape == cut.Freq_cut.shape
    assert len(params) == 14


# ----------------------------------------------------------------------
# Denoiser returns the original cut shape
# ----------------------------------------------------------------------
def test_denoising_preserves_shape(synthetic_arrays):
    from drlib import Linecut, Spectrum
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    sp = Spectrum.from_arrays(f, B, mlin, mlin_ref,
                              saturation=B.size - 1, warning=False)
    cut = Linecut(cut=B.size // 2, Spectrum=sp, frequencies=(1.5e9, 5e9))
    _, denoised = cut.denoising()
    assert denoised.shape == cut.S21dd_cut.shape


# ----------------------------------------------------------------------
# Plot runs without crashing
# ----------------------------------------------------------------------
@needs_data
def test_linecut_plot_runs(spectrum_dd):
    from drlib import Linecut
    cut_idx = scan_index_at_field(spectrum_dd, 80)
    cut = Linecut(cut=cut_idx, Spectrum=spectrum_dd, frequencies=(3.5e9, 4.7e9))
    cut.plot(style=None, pic_size=(4, 3))
