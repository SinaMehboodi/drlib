"""
drlib
=====

A Python library for ferromagnetic resonance (FMR) spectroscopy analysis,
coplanar waveguide (CPW) field modelling, and REXS-style reciprocal-space
simulation of magnetic textures.

Public API
----------
Classes
~~~~~~~
Spectrum
    Load a 2-D FMR spectrum (S-parameter vs. frequency vs. magnetic field)
    from LabView-generated ``.mat`` files **or** from pre-loaded NumPy
    arrays, apply derivative-divide / derivative / ΔS21 background
    correction, and plot / zoom / denoise it.

    Three constructors:

    - :class:`Spectrum(path=...)`           – original LabView folder layout
    - :meth:`Spectrum.from_arrays(...)`     – build from NumPy arrays
    - :meth:`Spectrum.from_mat(directory)`  – flat directory of ``.mat`` files
      (matches the bundled ``PRJCT/Data/`` reference dataset)

Linecut
    Extract a constant-field cut from a :class:`Spectrum` and fit one,
    two, or three :func:`dS21` / :func:`Lorentzian` peaks to it.

CPW
    Analytical Biot-Savart magnetic field distribution above a coplanar
    waveguide (signal-gap-ground geometry). Produces real-space field
    profiles, 2-D vector maps, and the k-space excitation spectrum.

REXS2D
    Simulator for 2-D resonant elastic x-ray scattering (REXS) intensity
    of magnetic textures (skyrmion lattice, helical state, coexistence).

Functions
~~~~~~~~~
derivative, derivative_divide
    Numerical derivative tools for 2-D datasets, used as background
    correction in broadband FMR spectroscopy.

dS21, Lorentzian
    Analytical models used to fit complex S21 spectra. Both share the
    same parameter order ``(x, A, Psi, fres, Df, mod, Msat, H0)`` so
    they can be swapped in :class:`Linecut` according to whether the
    underlying data was derivative-divide processed.

B_BS_analytic
    Closed-form Biot-Savart field of a finite-cross-section conductor.

FFT_1D
    Convenience wrapper around ``numpy.fft.fft`` with zero-padding and
    a shifted angular-wavenumber axis.

compare_techniques
    One-call comparison plot of the three FMR background-correction
    techniques (ΔS21 / derivative / derivative-divide).

load_mat_dataset
    Convenience loader for the simple four-``.mat``-file directory layout
    used by the bundled ``PRJCT/Data/`` reference dataset.

set_size, timeit, safe_style
    Helper utilities (LaTeX-ready figure sizes, timing decorator,
    matplotlib-style helper with a graceful fallback).

Quickstart
----------
Using the bundled reference dataset:

>>> from drlib import Spectrum, compare_techniques
>>> from drlib.io import load_mat_dataset
>>> freq, field, mlin, mlin_ref = load_mat_dataset(r"PRJCT/Data")
>>> compare_techniques(freq, field, mlin, mlin_ref)   # 3-panel comparison
>>> spec = Spectrum.from_arrays(freq, field, mlin, mlin_ref,
...                             saturation=80, derivative_divide=True)
>>> spec.plot()

See ``README.md`` and ``DATA_SHAPES.md`` for the details of every
required input shape.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Version / metadata
# ----------------------------------------------------------------------
__version__ = "1.1.0"
__author__ = "Sina Mehboodi"
__license__ = "MIT"

# ----------------------------------------------------------------------
# Public re-exports
# ----------------------------------------------------------------------
from .utils import set_size, timeit, read_measurement_txt, safe_style
from .math_tools import derivative, derivative_divide, FFT_1D
from .models import dS21, Lorentzian
from .field import (
    indefinite_integral_x,
    integrand,
    B_BS_analytic,
    CPW,
)
from .io import load_mat_dataset
from .spectrum import Spectrum, compare_techniques
from .linecut import Linecut
from .rexs import REXS2D

__all__ = [
    # version
    "__version__",
    # utils
    "set_size",
    "timeit",
    "safe_style",
    "read_measurement_txt",
    # math
    "derivative",
    "derivative_divide",
    "FFT_1D",
    # models
    "dS21",
    "Lorentzian",
    # field
    "indefinite_integral_x",
    "integrand",
    "B_BS_analytic",
    "CPW",
    # io
    "load_mat_dataset",
    # classes
    "Spectrum",
    "Linecut",
    "REXS2D",
    # high-level helpers
    "compare_techniques",
]
