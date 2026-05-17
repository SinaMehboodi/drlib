"""Shared pytest fixtures + matplotlib setup for the drlib test suite.

The package's plotting code calls ``matplotlib.pyplot`` directly; we
switch to the non-interactive ``Agg`` backend before any test imports
``drlib`` so test runs never pop up GUI windows.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must run before drlib imports matplotlib internally

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pytest                    # noqa: E402


# ----------------------------------------------------------------------
# Auto-cleanup: close every figure between tests
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _close_figures_between_tests():
    yield
    plt.close("all")


# ----------------------------------------------------------------------
# Path to the bundled PRJCT/Data reference dataset
# ----------------------------------------------------------------------
def _resolve_data_dir() -> Path:
    """Locate a usable FMR dataset from inside the test environment.

    Order of preference:
      1. ``DRLIB_DATA_DIR`` env var (CI / cross-machine override)
      2. ``<repo>/examples/data`` — the downsampled copy shipped in-repo
      3. ``<repo>/../Data``       — the full-resolution copy
      4. legacy absolute path on the author's machine
    """
    env = os.environ.get("DRLIB_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p

    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "examples" / "data",   # in-repo downsampled
        here.parent.parent / "Data",         # author's full-res, sibling of DrLib
        Path(r"F:/Sina_Data_Job2026/PRJCT/Data"),
    ]
    for c in candidates:
        p = c.resolve()
        if (p / "MLIN.mat").is_file():
            return p

    return candidates[0].resolve()  # let the test fail with a clear FileNotFoundError


DATA_DIR = _resolve_data_dir()
HAS_DATA = (DATA_DIR / "MLIN.mat").is_file()
needs_data = pytest.mark.skipif(
    not HAS_DATA,
    reason=f"Reference dataset not found at {DATA_DIR}. "
           f"Set DRLIB_DATA_DIR to override.",
)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Folder holding ``freq.mat`` / ``MLIN.mat`` / ``MLIN_REF.mat`` /
    ``sample_field.mat`` for the bundled reference dataset."""
    return DATA_DIR


@pytest.fixture(scope="session")
def raw_arrays(data_dir):
    """Tuple ``(freq, field, mlin, mlin_ref)`` loaded once per test session."""
    from drlib.io import load_mat_dataset
    return load_mat_dataset(str(data_dir))


@pytest.fixture(scope="session")
def spectrum_dd(raw_arrays):
    """A derivative-divide :class:`Spectrum` built from the reference data.

    ``saturation`` is set to the last field index (highest applied B) so
    the fixture works on both the full-resolution dataset (N_field=201)
    and the downsampled one shipped in-repo (N_field=101).
    """
    from drlib import Spectrum
    freq, field, mlin, mlin_ref = raw_arrays
    return Spectrum.from_arrays(
        freq, field, mlin, mlin_ref,
        saturation=int(field.size - 1),
        derivative_divide=True, warning=False,
    )


def scan_index_at_field(spec, B_target_mT: float) -> int:
    """Return the spectrum scan index closest to a given field value (mT).

    Test helper that lets the same test exercise both the full-resolution
    and the downsampled dataset without depending on integer indices.
    """
    return int(np.argmin(np.abs(spec.B - B_target_mT)))


# ----------------------------------------------------------------------
# Synthetic spectrum (always available, no real data required)
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_arrays():
    """Tiny synthetic (freq, field, mlin, mlin_ref) dataset.

    Built so that a single Lorentzian resonance sits at ``f_res(H)``
    along a linear Kittel-like dispersion — enough to exercise every
    code path without depending on the bundled .mat files.
    """
    rng = np.random.default_rng(0)
    N_freq, N_field = 401, 51
    f = np.linspace(1e9, 6e9, N_freq)
    B = np.linspace(0.0, 100.0, N_field)              # mT
    f_res = 1.5e9 + 0.03e9 * B                        # Kittel-like in GHz/mT
    Df = 0.05e9                                       # 50 MHz HWHM

    F, BB = np.meshgrid(f, B, indexing="ij")
    Fres = 1.5e9 + 0.03e9 * BB
    chi = (Df) / ((F - Fres) ** 2 + Df ** 2)          # absorption Lorentzian
    bg = 0.7 + 0.05 * F / F.max()                     # smooth background
    mlin = bg - 0.05 * chi + 0.001 * rng.standard_normal(F.shape)
    mlin_ref = bg.copy()

    return f, B, mlin, mlin_ref, f_res, Df
