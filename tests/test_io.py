"""Tests for ``drlib.io.load_mat_dataset``."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_data


@needs_data
def test_load_mat_dataset_returns_four_arrays(raw_arrays):
    freq, field, mlin, mlin_ref = raw_arrays
    assert isinstance(freq, np.ndarray)
    assert isinstance(field, np.ndarray)
    assert isinstance(mlin, np.ndarray)
    assert isinstance(mlin_ref, np.ndarray)


@needs_data
def test_load_mat_dataset_shapes(raw_arrays):
    freq, field, mlin, mlin_ref = raw_arrays
    assert freq.shape == mlin.shape
    assert mlin_ref.shape == mlin.shape
    assert field.ndim == 1
    assert field.size == mlin.shape[1]


@needs_data
def test_load_mat_dataset_units(raw_arrays):
    """Sanity-check the physical ranges of the bundled reference dataset."""
    freq, field, mlin, mlin_ref = raw_arrays
    # frequencies are 10 MHz – 10 GHz
    assert freq.min() == pytest.approx(1e7, rel=1e-9)
    assert freq.max() == pytest.approx(1e10, rel=1e-9)
    # field is integer 0..200
    assert field.min() == 0
    assert field.max() == 200
    # MLIN is a linear magnitude bounded by 0..~1
    assert 0.0 < mlin.min() < mlin.max() < 1.5


def test_load_mat_dataset_skip_optional(tmp_path):
    """``mlin_ref_name=None`` and ``field_name=None`` skip optional files."""
    from scipy.io import savemat

    from drlib.io import load_mat_dataset

    # Minimal pair of files
    nf, nh = 64, 32
    freq = np.tile(np.linspace(1e9, 5e9, nf)[:, None], (1, nh))
    mlin = 0.5 + 0.01 * np.random.default_rng(0).standard_normal((nf, nh))
    savemat(tmp_path / "freq.mat", {"Data": freq})
    savemat(tmp_path / "MLIN.mat", {"Data": mlin})

    out = load_mat_dataset(
        str(tmp_path), mlin_ref_name=None, field_name=None,
    )
    f, fld, m, r = out
    assert f.shape == (nf, nh)
    assert m.shape == (nf, nh)
    assert fld is None
    assert r is None


def test_load_mat_dataset_alternate_keys(tmp_path):
    """The loader accepts the ``Data`` key or the explicit variable name."""
    from scipy.io import savemat

    from drlib.io import load_mat_dataset

    nf, nh = 32, 8
    freq = np.tile(np.linspace(1e9, 2e9, nf)[:, None], (1, nh))
    mlin = np.ones((nf, nh))
    mlin_ref = np.full((nf, nh), 0.5)
    field = np.arange(nh)

    savemat(tmp_path / "freq.mat", {"Data": freq})
    savemat(tmp_path / "MLIN.mat", {"Data": mlin})
    savemat(tmp_path / "MLIN_REF.mat", {"Data": mlin_ref})
    savemat(tmp_path / "sample_field.mat", {"sample_field": field})

    f, fld, m, r = load_mat_dataset(str(tmp_path))
    assert np.array_equal(fld, field)
