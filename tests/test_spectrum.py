"""Tests for ``drlib.Spectrum`` and ``compare_techniques``.

Covers all three constructors (``__init__``, ``from_arrays``, ``from_mat``),
all three correction modes, the arithmetic operators, and the helper
``compare_techniques``.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_data


# ----------------------------------------------------------------------
# from_arrays — constructor + the three processing modes
# ----------------------------------------------------------------------
def test_from_arrays_derivative_divide(synthetic_arrays):
    from drlib import Spectrum
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    spec = Spectrum.from_arrays(f, B, mlin, mlin_ref,
                                saturation=B.size // 2,
                                derivative_divide=True, warning=False)
    assert spec.S21dd.shape == (B.size, f.shape[0])
    assert spec.derivative_divide is True
    assert spec.Msat == B[B.size // 2]


def test_from_arrays_derivative(synthetic_arrays):
    from drlib import Spectrum
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    spec = Spectrum.from_arrays(f, B, mlin, mlin_ref,
                                saturation=B.size - 1,
                                derivative_divide=False, derivative=True,
                                warning=False)
    assert spec.derivative_divide is False
    assert spec.derivative is True
    # Plain derivative should be roughly anti-symmetric across the resonance.
    assert np.isfinite(spec.S21dd).all()


def test_from_arrays_delta(synthetic_arrays):
    from drlib import Spectrum
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    spec = Spectrum.from_arrays(f, B, mlin, mlin_ref,
                                saturation=B.size - 1,
                                derivative_divide=False, derivative=False,
                                warning=False)
    # ΔS21 = mlin − mlin_ref; should be small (synthetic ref ≈ background)
    assert abs(spec.S21dd).max() < 1.0


def test_from_arrays_requires_ref_for_delta(synthetic_arrays):
    from drlib import Spectrum
    f, B, mlin, _, _, _ = synthetic_arrays
    with pytest.raises(ValueError):
        Spectrum.from_arrays(f, B, mlin, mlin_ref=None,
                             derivative_divide=False, derivative=False,
                             warning=False)


# ----------------------------------------------------------------------
# from_mat — equivalence with from_arrays on the bundled dataset
# ----------------------------------------------------------------------
@needs_data
def test_from_mat_matches_from_arrays(data_dir, raw_arrays):
    from drlib import Spectrum
    freq, field, mlin, mlin_ref = raw_arrays
    sat = int(field.size - 1)
    sp_arr = Spectrum.from_arrays(freq, field, mlin, mlin_ref,
                                  saturation=sat, derivative_divide=True,
                                  warning=False)
    sp_mat = Spectrum.from_mat(str(data_dir), saturation=sat,
                               derivative_divide=True, warning=False)
    assert np.allclose(sp_arr.S21dd, sp_mat.S21dd)
    assert np.array_equal(sp_arr.B, sp_mat.B)


@needs_data
def test_from_mat_shapes(spectrum_dd):
    """Shapes are consistent across N_field/N_freq combinations.

    Works on both the bundled downsampled dataset (101, 2501) and the
    full-resolution one (201, 20001) by checking internal consistency.
    """
    N_field = spectrum_dd.B.size
    N_freq = spectrum_dd.Freq.shape[0]
    assert spectrum_dd.S21dd.shape == (N_field, N_freq)
    assert spectrum_dd.Freq.shape == (N_freq, N_field)
    assert N_field >= 50 and N_freq >= 1000  # bare minimum sanity


# ----------------------------------------------------------------------
# Arithmetic operators — work for from_arrays-built spectra (no disk IO)
# ----------------------------------------------------------------------
def test_arithmetic_operators(synthetic_arrays):
    from drlib import Spectrum
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    sp = Spectrum.from_arrays(f, B, mlin, mlin_ref,
                              saturation=B.size - 1, warning=False)

    sp_2x = sp * 2
    assert np.allclose(sp_2x.S21dd, 2.0 * sp.S21dd)

    sp_half = sp / 2
    assert np.allclose(sp_half.S21dd, sp.S21dd / 2.0)

    sp_sum = sp + sp_2x
    assert np.allclose(sp_sum.S21dd, 3.0 * sp.S21dd)

    sp_diff = sp_2x - sp
    assert np.allclose(sp_diff.S21dd, sp.S21dd)


def test_arithmetic_shape_mismatch_raises(synthetic_arrays):
    from drlib import Spectrum
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    sp = Spectrum.from_arrays(f, B, mlin, mlin_ref,
                              saturation=B.size - 1, warning=False)
    other = sp / 1
    other.S21dd = other.S21dd[:-1, :]   # break the shape
    with pytest.raises(ValueError):
        _ = sp + other


# ----------------------------------------------------------------------
# Repr is informative + ASCII-safe (Windows console works)
# ----------------------------------------------------------------------
def test_repr_is_ascii(synthetic_arrays):
    from drlib import Spectrum
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    sp = Spectrum.from_arrays(f, B, mlin, mlin_ref, warning=False)
    text = repr(sp)
    text.encode("ascii")   # raises if any non-ASCII char sneaked in
    assert "Spectrum" in text and "mode=" in text


# ----------------------------------------------------------------------
# compare_techniques — both reference modes; auto vmin/vmax
# ----------------------------------------------------------------------
def test_compare_techniques_runs_with_saturation_index(synthetic_arrays):
    from drlib import compare_techniques
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    fig, axes = compare_techniques(f, B, mlin, saturation_index=B.size - 1)
    assert len(axes) == 3
    assert all(len(ax.get_images()) + len(ax.collections) > 0 for ax in axes)


def test_compare_techniques_runs_with_mlin_ref(synthetic_arrays):
    from drlib import compare_techniques
    f, B, mlin, mlin_ref, _, _ = synthetic_arrays
    fig, axes = compare_techniques(f, B, mlin, mlin_ref)
    assert len(axes) == 3


def test_compare_techniques_requires_reference(synthetic_arrays):
    from drlib import compare_techniques
    f, B, mlin, _, _, _ = synthetic_arrays
    with pytest.raises(ValueError):
        compare_techniques(f, B, mlin)


# ----------------------------------------------------------------------
# Plotting methods don't crash on the real spectrum
# ----------------------------------------------------------------------
@needs_data
def test_spectrum_plot_runs(spectrum_dd):
    """``plot`` runs end-to-end (style falls back gracefully)."""
    spectrum_dd.plot(style=None, pic_size=(4, 3))


@needs_data
def test_spectrum_zoom_plot_runs(spectrum_dd):
    """``zoom_plot`` runs with a scan range derived from the grid size."""
    n = spectrum_dd.B.size
    spectrum_dd.zoom_plot(
        freq_range=(2e9, 5e9), scan_range=(n // 4, 3 * n // 4),
        style=None, pic_size=(4, 3),
    )
