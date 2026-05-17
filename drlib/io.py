"""
drlib.io
========

Lightweight, **folder-layout-agnostic** data loaders.

The original :class:`drlib.Spectrum` constructor expects the strict LabView
on-disk layout documented in ``DATA_SHAPES.md``.  This module is for the
opposite case: you already have the raw arrays sitting in a handful of
``.mat`` files (or even in memory as NumPy arrays) and just want to drop
them straight into a :class:`drlib.Spectrum`.

Two entry points are provided:

* :func:`load_mat_dataset` – read a flat directory of four ``.mat`` files
  (``freq.mat`` / ``MLIN.mat`` / ``MLIN_REF.mat`` / ``sample_field.mat``)
  and return ready-to-use NumPy arrays.
* :class:`drlib.Spectrum.from_mat` / :meth:`drlib.Spectrum.from_arrays`
  build a :class:`Spectrum` directly on top of those arrays (see
  ``spectrum.py``).

The bundled demo dataset ``PRJCT/Data/`` matches exactly the layout
expected by :func:`load_mat_dataset`, so a one-liner is enough to get
started::

    >>> from drlib.io import load_mat_dataset
    >>> freq, field, mlin, mlin_ref = load_mat_dataset(r"PRJCT/Data")
"""

from __future__ import annotations

from os.path import join
from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat


# ----------------------------------------------------------------------
# Expected on-disk shapes (advisory; loader does its best to coerce)
# ----------------------------------------------------------------------
_DEFAULT_FILENAMES = {
    "freq":     "freq.mat",
    "mlin":     "MLIN.mat",
    "mlin_ref": "MLIN_REF.mat",
    "field":    "sample_field.mat",
}


def _pick_data_array(mat: dict) -> np.ndarray:
    """Return the first non-underscore array stored in a ``loadmat`` dict.

    ``scipy.io.loadmat`` returns a dictionary keyed by variable names
    plus a handful of housekeeping entries (``__header__``, ``__version__``,
    ``__globals__``). The bundled demo files use either the key ``Data``
    or the explicit variable name (``sample_field``). This helper just
    grabs whichever real array is present so the caller does not have
    to care which convention the LabView VI used.
    """
    candidates = [k for k in mat.keys() if not k.startswith("__")]
    if not candidates:
        raise ValueError(
            "No data array found in .mat file. "
            "Keys present: " + ", ".join(mat.keys())
        )
    # Prefer the conventional 'Data' key when present, otherwise the first
    if "Data" in candidates:
        return np.asarray(mat["Data"])
    return np.asarray(mat[candidates[0]])


# ----------------------------------------------------------------------
# Flat-directory loader
# ----------------------------------------------------------------------
def load_mat_dataset(
    directory: str,
    freq_name: str = _DEFAULT_FILENAMES["freq"],
    mlin_name: str = _DEFAULT_FILENAMES["mlin"],
    mlin_ref_name: Optional[str] = _DEFAULT_FILENAMES["mlin_ref"],
    field_name: Optional[str] = _DEFAULT_FILENAMES["field"],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Load an FMR dataset from a directory of four ``.mat`` files.

    The expected layout — and the exact layout of the bundled
    ``PRJCT/Data/`` reference dataset — is::

        <directory>/
        ├── freq.mat          # (N_freq, N_field) frequency grid (Hz)
        ├── MLIN.mat          # (N_freq, N_field) |S21| of the sample
        ├── MLIN_REF.mat      # (N_freq, N_field) reference baseline
        └── sample_field.mat  # (N_field,) or (N_field, N_freq) field grid

    All file names can be overridden via the keyword arguments. Both
    ``MLIN_REF.mat`` and ``sample_field.mat`` are optional — pass
    ``None`` for the corresponding ``*_name`` argument to skip them.

    Parameters
    ----------
    directory : str
        Path to the folder containing the four ``.mat`` files.
    freq_name : str, optional
        File name of the frequency grid. Default ``"freq.mat"``.
    mlin_name : str, optional
        File name of the measurement magnitude. Default ``"MLIN.mat"``.
    mlin_ref_name : str or None, optional
        File name of the reference (saturation) magnitude.
        Default ``"MLIN_REF.mat"``. Set to ``None`` to skip.
    field_name : str or None, optional
        File name of the magnetic-field grid.
        Default ``"sample_field.mat"``. Set to ``None`` to skip; in that
        case the caller must supply ``field`` separately when building
        a :class:`drlib.Spectrum`.

    Returns
    -------
    freq : ndarray, shape (N_freq, N_field)
        Frequency grid (Hz).
    field : ndarray, shape (N_field,) or None
        1-D magnetic-field grid (mT). ``None`` if ``field_name=None``.
        If the file stores a 2-D array, the first row/column (whichever
        spans the ``N_field`` axis) is used.
    mlin : ndarray, shape (N_freq, N_field)
        |S-parameter| of the measurement (linear magnitude).
    mlin_ref : ndarray, shape (N_freq, N_field) or None
        Reference baseline. ``None`` if ``mlin_ref_name=None``.

    Examples
    --------
    >>> from drlib.io import load_mat_dataset
    >>> freq, field, mlin, mlin_ref = load_mat_dataset(r"PRJCT/Data")
    >>> mlin.shape, freq.shape, field.shape
    ((20001, 201), (20001, 201), (201,))
    """
    freq = _pick_data_array(loadmat(join(directory, freq_name)))
    mlin = _pick_data_array(loadmat(join(directory, mlin_name)))

    mlin_ref: Optional[np.ndarray] = None
    if mlin_ref_name is not None:
        mlin_ref = _pick_data_array(loadmat(join(directory, mlin_ref_name)))

    field: Optional[np.ndarray] = None
    if field_name is not None:
        raw = _pick_data_array(loadmat(join(directory, field_name)))
        field = _coerce_field_to_1d(raw, expected_len=mlin.shape[1])

    _validate_shapes(freq, mlin, mlin_ref, field)
    return freq, field, mlin, mlin_ref


def _coerce_field_to_1d(raw: np.ndarray, expected_len: int) -> np.ndarray:
    """Reduce a possibly-tiled 2-D field array to a 1-D vector of length ``expected_len``.

    The reference dataset stores ``sample_field`` as ``(N_field, N_freq)``
    with each row equal to the same scalar field value (tiled). This
    helper picks whichever axis matches ``expected_len`` and returns the
    corresponding 1-D slice.
    """
    raw = np.asarray(raw)
    if raw.ndim == 1:
        return raw
    if raw.ndim == 2:
        if raw.shape[0] == expected_len:
            return raw[:, 0]
        if raw.shape[1] == expected_len:
            return raw[0, :]
        # As a fallback, flatten unique values
        flat = np.unique(raw)
        if flat.size == expected_len:
            return np.sort(flat)
        raise ValueError(
            f"Cannot reduce field array of shape {raw.shape} to length "
            f"{expected_len}. Pass `field` explicitly to Spectrum.from_arrays."
        )
    raise ValueError(f"Field array must be 1-D or 2-D, got ndim={raw.ndim}.")


def _validate_shapes(
    freq: np.ndarray,
    mlin: np.ndarray,
    mlin_ref: Optional[np.ndarray],
    field: Optional[np.ndarray],
) -> None:
    """Sanity-check loaded arrays. Raises ``ValueError`` on a mismatch."""
    if freq.shape != mlin.shape:
        raise ValueError(
            f"freq and MLIN must have the same shape; got "
            f"{freq.shape} and {mlin.shape}."
        )
    if mlin_ref is not None and mlin_ref.shape != mlin.shape:
        # tiled-1D references are also acceptable as long as the axis matches
        if not (mlin_ref.ndim == 1 and mlin_ref.size == mlin.shape[0]):
            raise ValueError(
                f"MLIN_REF must match MLIN's shape {mlin.shape} (or be 1-D "
                f"of length {mlin.shape[0]}); got {mlin_ref.shape}."
            )
    if field is not None and field.size != mlin.shape[1]:
        raise ValueError(
            f"Field grid must have length N_field={mlin.shape[1]}; "
            f"got length {field.size}."
        )
