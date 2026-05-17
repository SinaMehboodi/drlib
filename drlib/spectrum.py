"""
drlib.spectrum
==============

The :class:`Spectrum` class loads a 2-D ferromagnetic-resonance (FMR) data
set saved by Aisha's LabView VI and exposes it as a clean Python object,
optionally with derivative-divide background correction applied.

Expected on-disk layout (see ``DATA_SHAPES.md`` for the full spec)::

    <path>/
    ├── MagneticField.csv               (auto-generated on first load)
    └── measurement/
        ├── MEAS/  (only used to scrape field values on first load)
        └── store/
            ├── MEAS/<S_param>/
            │   ├── freq.mat            (shape: N_freq, N_field)
            │   └── MLIN.mat            (shape: N_freq, N_field)
            └── REF/<S_param>/MLIN.mat  (reference baseline)

Use :meth:`Spectrum.plot` to view the spectrum, :meth:`Spectrum.zoom_plot`
to look at a sub-region, and :meth:`Spectrum.scatter_plot` to overlay
peak-fit results on top of the spectrum.
"""

from __future__ import annotations

from os import listdir
from os.path import exists
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from pandas import DataFrame, to_numeric, read_csv
from scipy.fft import fft2, fftfreq, ifft2
from scipy.io import loadmat

from .math_tools import derivative, derivative_divide
from .utils import safe_style as _safe_style, set_size, timeit


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def compare_techniques(
    freq: np.ndarray,
    field: np.ndarray,
    mlin: np.ndarray,
    mlin_ref: Optional[np.ndarray] = None,
    *,
    saturation_index: Optional[int] = None,
    modulation_amp: int = 1,
    v_min_delta: Optional[Tuple[float, float]] = None,
    v_min_der: Tuple[float, float] = (-1e-3, 1e-3),
    v_min_dd: Tuple[float, float] = (-1e-3, 1e-3),
    c_map: str = "PuOr",
    pic_size: Tuple[float, float] = (12, 4),
    titles: Tuple[str, str, str] = (
        r"$\Delta S_{21} = S_{21}(H) - S_{21}(H_\mathrm{ref})$",
        r"$\partial S_{21} / \partial H$",
        r"$(\partial S_{21}/\partial H)\,/\,S_{21}$",
    ),
):
    """Plot the three FMR background-correction techniques side by side.

    Reproduces the comparison panel the user is used to making by hand
    (see ``CSOBS10QL.ipynb``): one figure with three ``pcolormesh`` panels
    showing the same dataset processed with

    1. **ΔS21**              – plain reference subtraction
    2. **Derivative**        – :func:`drlib.derivative` along the field axis
    3. **Derivative-divide** – :func:`drlib.derivative_divide` along the field axis

    The reference used for the ΔS21 panel is one of:

    * the slice ``mlin[:, saturation_index]`` if ``saturation_index`` is
      given (matches the notebook practice of taking the saturation scan
      as the baseline);
    * otherwise ``mlin_ref`` as supplied (a 1-D or 2-D array).

    Parameters
    ----------
    freq : ndarray, shape (N_freq, N_field)
        Frequency grid (Hz).
    field : ndarray, shape (N_field,)
        Magnetic-field grid (mT).
    mlin : ndarray, shape (N_freq, N_field)
        |S-parameter| of the measurement (linear magnitude).
    mlin_ref : ndarray, optional
        External reference baseline, same shape as ``mlin`` (or 1-D of
        length ``N_freq``, which will be tiled along the field axis).
        Ignored when ``saturation_index`` is given.
    saturation_index : int, optional
        Field-axis index whose ``mlin`` column is used as the ΔS21
        reference. This is the canonical "pick a high-field scan and
        subtract it" workflow.
    modulation_amp : int, optional
        Stride passed to :func:`derivative` and :func:`derivative_divide`.
        Default ``1``.
    v_min_delta : (float, float), optional
        ``(vmin, vmax)`` for the ΔS21 panel. If ``None`` (default), the
        scale is chosen automatically as ``±3·std(ΔS21)``.
    v_min_der, v_min_dd : (float, float)
        Colour-scale limits for the other two panels.
    c_map : str
        Matplotlib colormap. Default ``"PuOr"``.
    pic_size : (float, float)
        ``figsize`` in inches.
    titles : (str, str, str)
        Per-panel titles (LaTeX).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : (Axes, Axes, Axes)
        The three subplot axes, in display order.

    Examples
    --------
    Using an external reference baseline:

    >>> from drlib import compare_techniques
    >>> from drlib.io import load_mat_dataset
    >>> freq, field, mlin, mlin_ref = load_mat_dataset(r"PRJCT/Data")
    >>> fig, axes = compare_techniques(freq, field, mlin, mlin_ref)

    Using a chosen saturation scan as the ΔS21 reference (preferred
    when ``mlin_ref`` is not a true saturation measurement):

    >>> fig, axes = compare_techniques(freq, field, mlin,
    ...                                saturation_index=200)
    """
    mlin = np.asarray(mlin)
    if mlin.ndim != 2:
        raise ValueError(f"mlin must be 2-D; got shape {mlin.shape}.")

    freq = np.asarray(freq)
    if freq.ndim == 1:
        freq = np.tile(freq[:, None], (1, mlin.shape[1]))
    elif freq.shape != mlin.shape:
        raise ValueError(
            f"freq shape {freq.shape} must equal mlin shape {mlin.shape} "
            f"or be 1-D of length N_freq."
        )

    if saturation_index is not None:
        ref_col = mlin[:, saturation_index]
        ref2d = np.tile(ref_col[:, None], (1, mlin.shape[1]))
    elif mlin_ref is not None:
        ref2d = np.asarray(mlin_ref)
        if ref2d.ndim == 1:
            ref2d = np.tile(ref2d[:, None], (1, mlin.shape[1]))
    else:
        raise ValueError("Pass either `mlin_ref` or `saturation_index`.")

    # Build a 2-D scan-index array used by the math helpers.
    n_field = mlin.shape[1]
    n_freq = mlin.shape[0]
    Field2d = np.transpose([np.arange(n_field)] * n_freq)  # (N_field, N_freq)

    delta = (mlin - ref2d).T  # (N_field, N_freq)
    _, _, deriv, _ = derivative(
        X=Field2d, Y=freq, Z=mlin.T,
        modulation_amp=modulation_amp, axis=0,
    )
    _, _, divd, _ = derivative_divide(
        X=Field2d, Y=freq, Z=mlin.T,
        modulation_amp=modulation_amp, axis=0,
    )

    if v_min_delta is None:
        s = 3.0 * float(np.std(delta))
        v_min_delta = (-s, +s)

    fig, axes = plt.subplots(1, 3, figsize=pic_size, sharey=True)
    panels = [
        (axes[0], delta,  v_min_delta, titles[0]),
        (axes[1], deriv,  v_min_der,   titles[1]),
        (axes[2], divd,   v_min_dd,    titles[2]),
    ]
    for ax, data, (vmin, vmax), title in panels:
        im = ax.pcolormesh(
            Field2d, freq.T * 1e-9, data,
            vmin=vmin, vmax=vmax, cmap=c_map, shading="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("scan index")
        fig.colorbar(im, ax=ax, pad=0.02)
    axes[0].set_ylabel("Frequency (GHz)")
    fig.tight_layout()
    return fig, axes


class Spectrum:
    """A 2-D FMR spectrum (frequency × magnetic field) backed by LabView files.

    On construction the class loads the magnitude (``MLIN.mat``) and
    frequency axis (``freq.mat``) for the chosen S-parameter, applies an
    optional derivative-divide (or plain derivative) background correction,
    and stores everything as ready-to-plot ndarrays.

    Parameters
    ----------
    path : str
        Absolute path to the **measurement root folder** (the folder that
        contains the ``measurement/`` subdirectory). See module docstring
        for the expected layout.
    saturation : int, optional
        Index along the field axis at which the Kittel mode saturates
        (``Hc2``). When provided, :attr:`Msat` is fixed to that field
        value, improving fit stability. If ``None`` (default), ``Msat``
        is left as a free fit parameter and a warning is printed (unless
        ``warning=False``).
    skip_field : int, optional
        Down-sampling stride along the field axis. Default ``1``.
    skip_freq : int, optional
        Down-sampling stride along the frequency axis. Default ``1``.
    derivative_divide : bool, optional
        If ``True`` (default) the loaded magnitude is processed by
        :func:`drlib.derivative_divide`. Overrides ``derivative``.
    derivative : bool, optional
        If ``True`` (and ``derivative_divide=False``) the plain
        :func:`drlib.derivative` is applied instead.
    warning : bool, optional
        Print the "no saturation set" warning when ``saturation is None``.
        Default ``True``.
    modulation_amp : int, optional
        Modulation amplitude (in pixels) passed through to the derivative
        operations. Default ``1``.
    S_param : {'S21', 'S11', 'S12', 'S22'}, optional
        Which S-parameter to load. Default ``'S21'``.

    Attributes
    ----------
    B : ndarray, shape (N_field,)
        Calibrated magnetic-field values (mT).
    Field : ndarray, shape (N_freq, N_field)
        Column indices broadcast to 2-D ("No. of Scan").
    Freq : ndarray, shape (N_freq, N_field)
        Frequency values broadcast to 2-D (Hz).
    S21dd : ndarray, shape (N_field, N_freq)
        Processed S-parameter data (after derivative-divide / derivative /
        reference subtraction).
    Msat : float or None
        Saturation field (T) inferred from ``saturation`` index, or
        ``None`` if not set.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        path: str,
        saturation: Optional[int] = None,
        skip_field: int = 1,
        skip_freq: int = 1,
        derivative_divide: bool = True,
        derivative: bool = False,
        warning: bool = True,
        modulation_amp: int = 1,
        S_param: str = "S21",
    ) -> None:
        self.modulation_amp = modulation_amp
        self.path = path
        self.skip_field = skip_field
        self.skip_freq = skip_freq
        self.derivative_divide = derivative_divide
        self.derivative = derivative
        self.S_param = S_param

        # Load and process
        self.B, self.Field, self.Freq, self.S21dd = self.get()
        self._finalize_saturation(saturation, warning=warning)

    # ------------------------------------------------------------------
    # Alternative constructors (folder-layout-agnostic)
    # ------------------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        freq: np.ndarray,
        field: np.ndarray,
        mlin: np.ndarray,
        mlin_ref: Optional[np.ndarray] = None,
        *,
        saturation: Optional[int] = None,
        derivative_divide: bool = True,
        derivative: bool = False,
        modulation_amp: int = 1,
        skip_field: int = 1,
        skip_freq: int = 1,
        S_param: str = "S21",
        warning: bool = True,
    ) -> "Spectrum":
        """Build a :class:`Spectrum` from in-memory NumPy arrays.

        Use this constructor when you already have your data loaded —
        for example, from the bundled demo dataset in ``PRJCT/Data/``
        via :func:`drlib.io.load_mat_dataset`, or from a custom on-disk
        format the LabView-folder loader does not understand.

        Parameters
        ----------
        freq : ndarray, shape (N_freq, N_field)
            Frequency grid in **Hz**.
        field : ndarray, shape (N_field,)
            Magnetic-field grid in **mT** (a 1-D array — one value per
            scan column).
        mlin : ndarray, shape (N_freq, N_field)
            Linear magnitude of the measured S-parameter.
        mlin_ref : ndarray, optional
            Reference baseline, same shape as ``mlin`` (or 1-D of length
            ``N_freq``, in which case it is tiled along the field axis).
            Only used when both ``derivative_divide`` and ``derivative``
            are ``False`` (plain ΔS21 path).
        saturation : int, optional
            Field-axis index at which the Kittel mode saturates.
        derivative_divide : bool, optional
            Apply :func:`drlib.derivative_divide`. Default ``True``.
        derivative : bool, optional
            Apply :func:`drlib.derivative` (used only if
            ``derivative_divide=False``).
        modulation_amp, skip_field, skip_freq, S_param, warning
            See the main :class:`Spectrum` docstring.

        Returns
        -------
        Spectrum
            A fully initialised :class:`Spectrum` with attributes
            ``B``, ``Field``, ``Freq``, ``S21dd`` already populated.

        Examples
        --------
        >>> from drlib import Spectrum
        >>> from drlib.io import load_mat_dataset
        >>> freq, field, mlin, mlin_ref = load_mat_dataset(r"PRJCT/Data")
        >>> spec = Spectrum.from_arrays(freq, field, mlin, mlin_ref,
        ...                             saturation=80, derivative_divide=True)
        >>> spec.plot(v_min=-1e-3, v_max=+1e-3)
        """
        obj = cls.__new__(cls)
        obj.path = None
        obj.modulation_amp = modulation_amp
        obj.skip_field = skip_field
        obj.skip_freq = skip_freq
        obj.derivative_divide = derivative_divide
        obj.derivative = derivative
        obj.S_param = S_param

        obj.B, obj.Field, obj.Freq, obj.S21dd = obj._process_signal(
            freq=freq, field=field, mlin=mlin, mlin_ref=mlin_ref,
        )
        obj._finalize_saturation(saturation, warning=warning)
        return obj

    @classmethod
    def from_mat(
        cls,
        directory: str,
        *,
        freq_name: str = "freq.mat",
        mlin_name: str = "MLIN.mat",
        mlin_ref_name: Optional[str] = "MLIN_REF.mat",
        field_name: Optional[str] = "sample_field.mat",
        saturation: Optional[int] = None,
        derivative_divide: bool = True,
        derivative: bool = False,
        modulation_amp: int = 1,
        skip_field: int = 1,
        skip_freq: int = 1,
        S_param: str = "S21",
        warning: bool = True,
    ) -> "Spectrum":
        """Build a :class:`Spectrum` from a flat directory of four ``.mat`` files.

        Thin wrapper around :func:`drlib.io.load_mat_dataset` +
        :meth:`Spectrum.from_arrays`. The on-disk layout it expects is the
        same as the bundled ``PRJCT/Data/`` reference dataset::

            <directory>/
            ├── freq.mat          (N_freq, N_field)  – frequency (Hz)
            ├── MLIN.mat          (N_freq, N_field)  – sample |S21|
            ├── MLIN_REF.mat      (N_freq, N_field)  – reference baseline
            └── sample_field.mat  (N_field, …)       – field grid (mT)

        Parameters
        ----------
        directory : str
            Path to the folder holding the four ``.mat`` files.
        freq_name, mlin_name, mlin_ref_name, field_name : str or None
            Override the default file names (or pass ``None`` to skip).
        saturation, derivative_divide, derivative, modulation_amp,
        skip_field, skip_freq, S_param, warning
            See :meth:`Spectrum.from_arrays`.

        Returns
        -------
        Spectrum

        Examples
        --------
        >>> from drlib import Spectrum
        >>> spec = Spectrum.from_mat(r"PRJCT/Data", saturation=80,
        ...                          derivative_divide=True)
        >>> spec.plot()
        """
        from .io import load_mat_dataset
        freq, field, mlin, mlin_ref = load_mat_dataset(
            directory,
            freq_name=freq_name, mlin_name=mlin_name,
            mlin_ref_name=mlin_ref_name, field_name=field_name,
        )
        return cls.from_arrays(
            freq=freq, field=field, mlin=mlin, mlin_ref=mlin_ref,
            saturation=saturation,
            derivative_divide=derivative_divide, derivative=derivative,
            modulation_amp=modulation_amp,
            skip_field=skip_field, skip_freq=skip_freq,
            S_param=S_param, warning=warning,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _finalize_saturation(self, saturation: Optional[int], *, warning: bool) -> None:
        """Resolve ``self.Msat`` from a saturation scan index."""
        self.saturation = saturation
        if saturation is None:
            if warning:
                print(
                    "The scan no. where the Kittel mode ends was not chosen. "
                    "All fits called from this object will treat Msat as a free "
                    "parameter. This might decrease fitting accuracy. It is "
                    "strongly suggested to call the plot method of this object, "
                    "use it to establish the saturation no. of scan, and then "
                    "reinitialize the object with the corresponding saturation value."
                )
            self.Msat = None
        else:
            # Msat is the field at the saturation index (kept in the units
            # of the supplied field grid — typically mT).
            self.Msat = self.B[np.where(self.B == self.B[saturation])[0][0]]

    def _process_signal(
        self,
        freq: np.ndarray,
        field: np.ndarray,
        mlin: np.ndarray,
        mlin_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply down-sampling + the chosen background correction.

        Centralised processing pipeline used by the folder-based
        :meth:`get` and by the array-based classmethods. Keeps the
        actual maths in **one** place so all three input paths produce
        identical output shapes and conventions.

        Parameters
        ----------
        freq : ndarray, shape (N_freq, N_field)
        field : ndarray, shape (N_field,)
        mlin : ndarray, shape (N_freq, N_field)
        mlin_ref : ndarray, optional
            Same shape as ``mlin`` (or 1-D of length ``N_freq``).

        Returns
        -------
        B : ndarray, shape (N_field,)
            Magnetic-field grid (down-sampled).
        Field : ndarray, shape (N_field, N_freq)
            2-D scan-index array used by the plot machinery.
        Freq : ndarray, shape (N_freq, N_field)
            2-D frequency array (down-sampled).
        S21 : ndarray, shape (N_field, N_freq)
            Processed signal.
        """
        sf = self.skip_freq
        sx = self.skip_field
        mlin_arr = np.asarray(mlin)
        if mlin_arr.ndim != 2:
            raise ValueError(
                f"mlin must be 2-D (N_freq, N_field); got shape {mlin_arr.shape}."
            )
        # Allow a 1-D freq vector of length N_freq for convenience: tile it.
        freq_arr = np.asarray(freq)
        if freq_arr.ndim == 1:
            freq_arr = np.tile(freq_arr[:, None], (1, mlin_arr.shape[1]))
        elif freq_arr.shape != mlin_arr.shape:
            raise ValueError(
                f"freq shape {freq_arr.shape} must equal mlin shape "
                f"{mlin_arr.shape} or be 1-D of length N_freq."
            )

        freq_ds = freq_arr[::sf, ::sx]
        mlin_ds = mlin_arr[::sf, ::sx]
        B = np.asarray(field)[::sx]

        # 2-D scan-index array, broadcast to (N_field, N_freq)
        n_field = mlin_ds.shape[1]
        n_freq = mlin_ds.shape[0]
        Field2d = np.transpose([np.arange(n_field)] * n_freq)  # (N_field, N_freq)

        if self.derivative_divide:
            _, _, S21dd, _ = derivative_divide(
                X=Field2d, Y=freq_ds, Z=mlin_ds.T,
                modulation_amp=self.modulation_amp, axis=0,
            )
            return B, Field2d, freq_ds, S21dd

        if self.derivative:
            _, _, S21d, _ = derivative(
                X=Field2d, Y=freq_ds, Z=mlin_ds.T,
                modulation_amp=self.modulation_amp, axis=0,
            )
            return B, Field2d, freq_ds, S21d

        # Plain reference-subtraction (ΔS21).  Broadcast a 1-D reference.
        if mlin_ref is None:
            raise ValueError(
                "`mlin_ref` must be supplied when both `derivative_divide` "
                "and `derivative` are False (plain ΔS21 path)."
            )
        mlin_ref_arr = np.asarray(mlin_ref)
        if mlin_ref_arr.ndim == 1:
            mlin_ref_arr = np.tile(mlin_ref_arr[:, None], (1, mlin.shape[1]))
        mlin_ref_ds = mlin_ref_arr[::sf, ::sx]
        S21d = mlin_ds.T - mlin_ref_ds.T
        return B, Field2d, freq_ds, S21d

    # ------------------------------------------------------------------
    # Data loading (folder-based)
    # ------------------------------------------------------------------
    def get(
        self,
        skip_field: int = 1,
        skip_freq: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and process the raw LabView data into ready-to-plot arrays.

        Reads the on-disk files described in :meth:`data_loader` and runs
        :meth:`_process_signal` on top of them. Together with the
        :meth:`from_arrays` / :meth:`from_mat` classmethods, this keeps the
        actual maths in **one** place so every entry point produces an
        identical ``(B, Field, Freq, S21dd)`` quadruple.

        Parameters
        ----------
        skip_field : int, optional
            Additional down-sampling along the field axis (multiplies any
            ``skip_field`` already given at construction time).
        skip_freq : int, optional
            Additional down-sampling along the frequency axis.

        Returns
        -------
        B : ndarray, shape (N_field,)
            Magnetic-field values (mT).
        Field : ndarray, shape (N_field, N_freq)
            2-D scan-index array.
        Freq : ndarray, shape (N_freq, N_field)
            2-D frequency array (Hz).
        S21 : ndarray, shape (N_field, N_freq)
            Processed S-parameter array.
        """
        f1, Freq, S21Refmag, S21Mag = self.data_loader()

        field0 = f1["field"][1::1]
        field1 = to_numeric(field0, errors="coerce").to_numpy()

        # Compose constructor + per-call strides without mutating self.skip_*,
        # so calling get() twice does NOT compound the down-sampling.
        eff_sf = self.skip_freq * skip_freq
        eff_sx = self.skip_field * skip_field

        old_sf, old_sx = self.skip_freq, self.skip_field
        try:
            self.skip_freq, self.skip_field = eff_sf, eff_sx
            return self._process_signal(
                freq=Freq, field=field1, mlin=S21Mag, mlin_ref=S21Refmag,
            )
        finally:
            self.skip_freq, self.skip_field = old_sf, old_sx

    def data_loader(self) -> Tuple:
        """Read the LabView .mat files and the magnetic-field CSV.

        On the first call the field values are scraped from filenames in
        ``measurement/MEAS/`` and cached to ``MagneticField.csv`` next to
        the measurement folder.

        Returns
        -------
        f1 : pandas.DataFrame
            DataFrame with columns ``num`` and ``field`` (mT).
        Freq : ndarray
            Frequency axis (Hz), shape ``(N_freq, N_field)``.
        S21Refmag : ndarray
            Reference S-parameter magnitude (linear), same shape as ``Freq``.
        S21Mag : ndarray
            Sample S-parameter magnitude (linear), same shape as ``Freq``.
        """
        MeasPath = self.path

        if not exists(MeasPath + "\\" + "MagneticField.csv"):
            listSubDir = listdir(MeasPath + r"\measurement\MEAS")
            main_Directory: List[str] = []
            for i in np.arange(0, len(listSubDir)):
                main_Directory.append(MeasPath + r"\measurement\MEAS" + "\\" + listSubDir[i])

            ListFiles = listdir(main_Directory[0])
            ListFilesfreq = list(filter(lambda x: "__freq.txt" in x, ListFiles))

            MagneticField: List[float] = []
            FieldName: List[str] = []
            for i in np.arange(0, len(ListFilesfreq)):
                FileName = ListFilesfreq[i].split("_")
                FieldName.append(FileName[2])
                FieldN = float(FileName[2].replace("mT", ""))
                MagneticField.append(FieldN)

            MagneticField = DataFrame(MagneticField, columns=["Magnetic Field Measure"])
            print("Magnetic field list is saved in this address:\n", MeasPath)
            MagneticField.to_csv(MeasPath + "\\" + "MagneticField.csv")

        Freq1 = loadmat(MeasPath + "\\" + fr"measurement\store\MEAS\{self.S_param}\freq.mat")
        S21Mag1 = loadmat(MeasPath + "\\" + fr"measurement\store\MEAS\{self.S_param}\MLIN.mat")
        Refmag = loadmat(MeasPath + "\\" + fr"measurement\store\REF\{self.S_param}\MLIN.mat")

        f1 = read_csv(self.path + "\\MagneticField.csv", names=["num", "field"])

        S21Mag = S21Mag1["Data"]
        Freq = Freq1["Data"]
        S21Refmag = Refmag["Data"]

        return f1, Freq, S21Refmag, S21Mag

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    @timeit
    def plot(
        self,
        save_name: Optional[str] = None,
        denoise: bool = False,
        Fnoise: float = 10,
        v_min: float = -0.001,
        v_max: float = +0.001,
        c_map: str = "bone",
        pic_size=480,
        style: str = "Beam",
        nom_locator: int = 20,
    ) -> None:
        """Plot the full FMR spectrum as a 2-D colour map.

        Parameters
        ----------
        save_name : str, optional
            Path to save the figure (must include the extension, e.g.
            ``"figure.png"``). If ``None`` (default) the figure is shown
            but not saved.
        denoise : bool, optional
            If ``True``, run :meth:`denoise_spectrum` and plot the cleaned
            data. Default ``False``.
        Fnoise : float, optional
            Cut-off for the low-pass filter used when ``denoise=True``.
        v_min, v_max : float, optional
            Colour-scale limits passed to ``pcolor``.
        c_map : str, optional
            Matplotlib colormap. Default ``"bone"``.
        pic_size : int or (float, float), optional
            If an int, treated as a LaTeX text-width in points and
            converted with :func:`drlib.set_size`. If a tuple, used as
            ``figsize=(w, h)`` in inches.
        style : str, optional
            Matplotlib style sheet. Default ``"Beam"``.
        nom_locator : int, optional
            Major-tick spacing on the field axis.
        """
        _safe_style(style)
        pic_s = pic_size if isinstance(pic_size, tuple) else set_size(pic_size)

        fig, (ax1, ax) = plt.subplots(
            nrows=2, sharex=True, figsize=pic_s,
            gridspec_kw={"height_ratios": [2, 6]},
        )

        ax1.plot(self.Field[:, 0], self.B, c="blue")

        if denoise:
            S21dd = np.real(self.denoise_spectrum(fnoise=Fnoise))
            Freq = self.Freq[:-1, :]
            Field = self.Field[:, :-1]
            im = ax.pcolor(
                Field, Freq.T, S21dd, vmin=-0.001, vmax=+0.001,
                cmap=c_map, shading="auto",
            )
        else:
            im = ax.pcolor(
                self.Field, self.Freq.T, self.S21dd,
                vmin=v_min, vmax=v_max, cmap=c_map, shading="auto",
            )

        self._format_colorbar(fig, im, v_min, v_max)
        self._format_axes(ax, ax1, nom_locator)

        ax.set_xlim(0, len(self.Field[:, 0]))
        ax.set_ylim(min(self.Freq[:, 0]), max(self.Freq[:, 0]))
        ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1e9))
        ax.yaxis.set_major_formatter(ticks_y)
        plt.subplots_adjust(wspace=0, hspace=0)

        if save_name is not None:
            plt.savefig(save_name, bbox_inches="tight", dpi=900)

    @timeit
    def zoom_plot(
        self,
        freq_range: Tuple[float, float],
        scan_range: Tuple[int, int],
        clip_range: Tuple[float, float] = (-10, 10),
        clipping: bool = False,
        save_name: Optional[str] = None,
        denoise: bool = False,
        Fnoise: float = 10,
        v_min: float = -0.001,
        v_max: float = +0.001,
        c_map: str = "PuOr",
        pic_size=480,
        style: str = "Beam",
        unit: float = 100,
        nom_locator: int = 5,
    ) -> None:
        """Plot a rectangular sub-region of the spectrum.

        Parameters
        ----------
        freq_range : (float, float)
            ``(f_min, f_max)`` in Hz.
        scan_range : (int, int)
            ``(scan_min, scan_max)`` — column-index bounds on the field axis.
        clip_range : (float, float), optional
            Used only when ``clipping=True``.
        clipping : bool, optional
            If ``True``, clip ``S21dd`` values to ``clip_range`` before
            plotting (useful when isolated outliers are saturating the
            colour scale).
        save_name : str, optional
            Output file path. ``None`` skips saving.
        denoise : bool, optional
            Apply :meth:`denoise_spectrum` before plotting.
        Fnoise, v_min, v_max, c_map, pic_size, style, nom_locator
            See :meth:`plot`.
        unit : float, optional
            Scale factor applied to the field-axis tick labels (default
            ``100``). Useful when the underlying field is stored as mT but
            should be displayed as oersted or A/m.
        """
        freq_min, freq_max = freq_range
        scan_min, scan_max = scan_range

        f_min = np.where(self.Freq > freq_min)[0][0]
        f_max = np.where(self.Freq < freq_max)[0][-1]

        _safe_style(style)
        pic_s = pic_size if isinstance(pic_size, tuple) else set_size(pic_size)
        fig, (ax1, ax) = plt.subplots(
            nrows=2, sharex=True, figsize=pic_s,
            gridspec_kw={"height_ratios": [2, 6]},
        )

        ax1.plot(self.Field[scan_min:scan_max, 0], self.B[scan_min:scan_max], c="blue")

        if denoise:
            S21dd = np.real(self.denoise_spectrum(fnoise=Fnoise))
            Freq = self.Freq[:-1, :]
            Field = self.Field[:, :-1]
            im = ax.pcolor(Field, Freq.T, S21dd, vmin=-0.001, vmax=+0.001,
                           cmap=c_map, shading="auto")
        else:
            if clipping:
                S21dd = np.clip(self.S21dd, clip_range[0], clip_range[1])
            else:
                S21dd = self.S21dd
            im = ax.pcolor(
                self.Field[scan_min:scan_max, f_min:f_max],
                self.Freq[f_min:f_max, scan_min:scan_max].T,
                S21dd[scan_min:scan_max, f_min:f_max],
                vmin=v_min, vmax=v_max, cmap=c_map, shading="auto",
            )

        self._format_colorbar(fig, im, v_min, v_max)

        ax1.set_ylabel("$B$ (mT)")
        ax.set_xlabel("No. of Scan")
        ax.set_ylabel("$f$ (GHz)")

        second = ax1.secondary_xaxis("top")
        second.xaxis.set_major_locator(ticker.MaxNLocator((scan_max - scan_min) / nom_locator))
        second.set_xticks([])
        ax1.yaxis.set_major_locator(
            ticker.FixedLocator(
                [
                    min(self.B[scan_min:scan_max]),
                    (min(self.B[scan_min:scan_max]) + max(self.B[scan_min:scan_max])) / 2,
                    max(self.B[scan_min:scan_max]),
                ]
            )
        )
        ax1.set_yticks(
            [
                min(self.B[scan_min:scan_max]),
                (min(self.B[scan_min:scan_max]) + max(self.B[scan_min:scan_max])) / 2,
                max(self.B[scan_min:scan_max]),
            ],
            [
                min(self.B[scan_min:scan_max]) * unit,
                (min(self.B[scan_min:scan_max]) + max(self.B[scan_min:scan_max])) / 2 * unit,
                max(self.B[scan_min:scan_max]) * unit,
            ],
        )
        ax1.grid()
        ax.xaxis.set_major_locator(ticker.MaxNLocator((scan_max - scan_min) / nom_locator))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_ylim(freq_min, freq_max)
        ax.set_xlim(scan_min, scan_max)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1e9))
        ax.yaxis.set_major_formatter(ticks_y)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        if save_name is not None:
            plt.savefig(save_name, bbox_inches="tight", dpi=900)

    @timeit
    def scatter_plot(
        self,
        field_cuts_list: List[np.ndarray],
        frequencies_list: List[Tuple[float, float]],
        frequencies_end_list: Optional[List[Optional[Tuple[float, float]]]] = None,
        colors: Optional[List[str]] = None,
        denoise: bool = False,
        save_name: Optional[str] = None,
        v_min: float = -0.005,
        v_max: float = +0.005,
        c_map: str = "PuOr",
        style: str = "Beam",
        pic_size=240,
        f_lim: Optional[Tuple[float, float]] = None,
        s_lim: Optional[Tuple[int, int]] = None,
        nom_locator: int = 50,
    ) -> None:
        """Overlay peak-fit resonance points on top of the full spectrum.

        For each entry in ``field_cuts_list`` / ``frequencies_list``, a series
        of :class:`drlib.Linecut` one-peak fits is run between the given
        scan-index limits and frequency limits, and the resulting resonance
        frequencies are scattered on top of the colour map.

        Parameters
        ----------
        field_cuts_list : list of ndarray
            One entry per fit series. Each entry is an array of scan-index
            cuts (e.g. ``np.arange(33, 36)``).
        frequencies_list : list of (float, float)
            ``(freq_min, freq_max)`` per fit series — defines the frequency
            window at the *start* of each series.
        frequencies_end_list : list of (float, float) or None, optional
            If given, the frequency window is linearly interpolated between
            ``frequencies_list[i]`` and ``frequencies_end_list[i]`` across
            the scans in ``field_cuts_list[i]``. ``None`` for an entry
            keeps the window constant. Default ``None``.
        colors : list of str, optional
            Matplotlib short specs (e.g. ``["ro", "gs"]``). Default
            ``["ro"] * len(field_cuts_list)``.
        denoise : bool, optional
            Plot the denoised spectrum in the background.
        save_name : str, optional
            Output file path.
        v_min, v_max, c_map, style, pic_size, f_lim, s_lim, nom_locator
            See :meth:`plot` and :meth:`zoom_plot`.
        """
        assert len(frequencies_list) == len(field_cuts_list), (
            "frequencies_list and field_cuts_list must be the same length."
        )
        for i in range(len(frequencies_list)):
            assert len(frequencies_list[i]) == 2, (
                "Each frequencies_list entry must be (freq_min, freq_max)."
            )

        if colors is None:
            colors = ["ro"] * len(field_cuts_list)
        else:
            assert len(colors) == len(frequencies_list), (
                "colors must match the length of frequencies_list."
            )

        pic_s = pic_size if isinstance(pic_size, tuple) else set_size(pic_size)
        fig, (ax1, ax) = plt.subplots(
            nrows=2, sharex=True, figsize=pic_s,
            gridspec_kw={"height_ratios": [2, 6]},
        )
        _safe_style(style)

        for i in range(len(field_cuts_list)):
            frequencies = (frequencies_list[i][0], frequencies_list[i][1])
            if frequencies_end_list is None or frequencies_end_list[i] is None:
                frequencies_end = None
            else:
                frequencies_end = (frequencies_end_list[i][0], frequencies_end_list[i][1])
            field_cuts = field_cuts_list[i]
            fd_reso, fr_reso, _, _ = self.scatter_fit(field_cuts, frequencies, frequencies_end)
            ax.plot(fd_reso, fr_reso, colors[i])

        ax1.plot(self.Field[:, 0], self.B, c="blue")

        if denoise:
            S21dd = np.real(self.denoise_spectrum())
            Freq = self.Freq[:-1, :]
            Field = self.Field[:, :-1]
            im = ax.pcolor(Field, Freq.T, S21dd, vmin=v_min, vmax=v_max,
                           cmap=c_map, shading="auto")
        else:
            im = ax.pcolor(self.Field, self.Freq.T, self.S21dd,
                           vmin=v_min, vmax=v_max, cmap=c_map, shading="auto")

        self._format_colorbar(fig, im, v_min, v_max)
        self._format_axes(ax, ax1, nom_locator)

        ax.set_ylim(min(self.Freq[:, 0]), max(self.Freq[:, 0]))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1e9))
        ax.yaxis.set_major_formatter(ticks_y)
        if s_lim is not None:
            ax.set_xlim(s_lim[0], s_lim[1])
        if f_lim is not None:
            ax.set_ylim(f_lim[0], f_lim[1])
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_name is not None:
            plt.savefig(save_name, bbox_inches="tight")

    @timeit
    def zoom_cycle_plot(
        self,
        cycle_range: Tuple[int, int],
        freq_range: Tuple[float, float],
        scan_range: Tuple[int, int],
        save_name: Optional[str] = None,
        cycling: bool = False,
        denoise: bool = False,
        v_min: float = -0.001,
        v_max: float = +0.001,
        c_map: str = "PuOr",
        pic_size=480,
        style: str = "Beam",
        unit: float = 100,
        nom_locator: int = 5,
        colorbar_cord: Tuple[float, float, float, float] = (1.01, 0.135, 0.02, 0.6),
    ) -> None:
        """Like :meth:`zoom_plot` but removes a "cycle" range in the middle.

        Useful when the field sweep has a repeated cycle (e.g. zero-field
        relaxation traces) that should be hidden from the plot.

        Parameters
        ----------
        cycle_range : (int, int)
            Scan-index range to remove from the displayed map.
        freq_range, scan_range, save_name, denoise, v_min, v_max, c_map,
        pic_size, style, unit, nom_locator
            See :meth:`zoom_plot`.
        cycling : bool, optional
            If ``True``, perform the actual removal. If ``False``, the
            method behaves like :meth:`zoom_plot`.
        colorbar_cord : (float, float, float, float), optional
            ``(left, bottom, width, height)`` axes coordinates for the
            colour bar — adjust if your figure size changes.
        """
        cycle_range_min, cycle_range_max = cycle_range
        index_off = cycle_range_max - cycle_range_min
        freq_min, freq_max = freq_range
        scan_min, scan_max = scan_range

        f_min = np.where(self.Freq > freq_min)[0][0]
        f_max = np.where(self.Freq < freq_max)[0][-1]

        _safe_style(style)
        pic_s = pic_size if isinstance(pic_size, tuple) else set_size(pic_size)
        fig, (ax1, ax) = plt.subplots(
            nrows=2, sharex=True, figsize=pic_s,
            gridspec_kw={"height_ratios": [2, 6]},
        )

        if cycling:
            new_s21dd = np.concatenate(
                (self.S21dd[scan_min:cycle_range_min, f_min:f_max],
                 self.S21dd[cycle_range_max:scan_max, f_min:f_max])
            )
            new_array = np.concatenate(
                (self.Field[scan_min:cycle_range_min, f_min:f_max],
                 self.Field[cycle_range_max:scan_max, f_min:f_max])
            )
            new_Freq = np.concatenate(
                (self.Freq[f_min:f_max, scan_min:cycle_range_min],
                 self.Freq[f_min:f_max, cycle_range_max:scan_max]),
                axis=1,
            )
            n_rows = new_array.shape[0]
            n_col = new_array.shape[1]
            new_Field = np.tile(np.arange(n_rows).reshape(n_rows, 1), (1, n_col))
            new_B_1D = np.concatenate(
                (self.B[scan_min:cycle_range_min], self.B[cycle_range_max:scan_max])
            )
            ax1.plot(new_Field[:, 0], new_B_1D, c="blue")
        else:
            ax1.plot(self.Field[scan_min:scan_max, 0], self.B[scan_min:scan_max], c="blue")

        if denoise:
            S21dd = np.real(self.denoise_spectrum())
            Freq = self.Freq[:-1, :]
            Field = self.Field[:, :-1]
            im = ax.pcolor(Field, Freq.T, S21dd, vmin=-0.001, vmax=+0.001,
                           cmap=c_map, shading="auto")
        elif cycling:
            im = ax.pcolor(new_Field, new_Freq.T, new_s21dd,
                           vmin=v_min, vmax=v_max, cmap=c_map, shading="auto")
        else:
            im = ax.pcolor(
                self.Field[scan_min:scan_max, f_min:f_max],
                self.Freq[f_min:f_max, scan_min:scan_max].T,
                self.S21dd[scan_min:scan_max, f_min:f_max],
                vmin=v_min, vmax=v_max, cmap=c_map, shading="auto",
            )

        cb_ax = fig.add_axes(list(colorbar_cord))
        cbar = fig.colorbar(im, cax=cb_ax, ticks=[v_min, 0, v_max])
        cbar.set_label(r"$Re(\partial_DS_{21}/\partial H)$")
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        cbar.minorticks_on()
        if not self.derivative_divide:
            cbar.set_label(r"$\Delta S_{21}$")

        second = ax1.secondary_xaxis("top")
        ax1.set_ylabel("$B$ (mT)")
        ax.set_xlabel("No. of Scan")
        ax.set_ylabel("$f$ (GHz)")
        second.xaxis.set_major_locator(
            ticker.MaxNLocator((scan_max - index_off - scan_min) / nom_locator)
        )
        second.set_xticks([])
        ax1.yaxis.set_major_locator(
            ticker.FixedLocator(
                [
                    min(self.B[scan_min:scan_max - index_off]),
                    (min(self.B[scan_min:scan_max - index_off]) +
                     max(self.B[scan_min:scan_max - index_off])) / 2,
                    max(self.B[scan_min:scan_max - index_off]),
                ]
            )
        )
        ax1.set_yticks(
            [min(self.B[scan_min:scan_max]),
             (min(self.B[scan_min:scan_max]) + max(self.B[scan_min:scan_max])) / 2,
             max(self.B[scan_min:scan_max])],
            [min(self.B[scan_min:scan_max]) * unit,
             (min(self.B[scan_min:scan_max]) + max(self.B[scan_min:scan_max])) / 2 * unit,
             max(self.B[scan_min:scan_max]) * unit],
        )
        ax1.grid()
        ax.xaxis.set_major_locator(
            ticker.MaxNLocator((scan_max - index_off - scan_min) / nom_locator)
        )
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_ylim(freq_min, freq_max)
        ax.set_xlim(scan_min, scan_max - index_off)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1e9))
        ax.yaxis.set_major_formatter(ticks_y)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_name is not None:
            plt.savefig(save_name, bbox_inches="tight", dpi=900)

    # ------------------------------------------------------------------
    # Fitting helpers
    # ------------------------------------------------------------------
    def scatter_fit(
        self,
        field_cuts: np.ndarray,
        frequencies: Tuple[float, float],
        frequencies_end: Optional[Tuple[float, float]] = None,
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """Run a series of one-peak Linecut fits and collect the resonances.

        Parameters
        ----------
        field_cuts : ndarray
            Scan-indices at which to fit (e.g. ``np.arange(33, 36)``).
        frequencies : (float, float)
            Lower and upper frequency limits for the fits at the **first**
            ``field_cuts`` entry.
        frequencies_end : (float, float), optional
            Frequency limits at the **last** ``field_cuts`` entry. If
            given, the limits are linearly interpolated across the cuts.
            Default ``None`` (constant frequency window).

        Returns
        -------
        fd_reso : list of int
            The ``field_cuts`` indices (unchanged), one per fit.
        fr_reso : list of float
            Fitted resonance frequency at each cut.
        Df_reso : list of float
            Fitted line width at each cut.
        Df_stdr : list of float
            Standard error on the line width (from lmfit).
        """
        # Local import to break cyclic dependency: linecut imports nothing
        # from spectrum but spectrum uses Linecut here.
        from .linecut import Linecut

        fr_reso: List[float] = []
        fd_reso: List[int] = []
        Df_reso: List[float] = []
        Df_stdr: List[float] = []

        freq_min, freq_max = frequencies
        resonance = (freq_min + freq_max) / 2

        if frequencies_end is None:
            ftop = np.linspace(frequencies[1], frequencies[1], len(field_cuts))
            fbot = np.linspace(frequencies[0], frequencies[0], len(field_cuts))
        else:
            ftop = np.linspace(frequencies[1], frequencies_end[1], len(field_cuts))
            fbot = np.linspace(frequencies[0], frequencies_end[0], len(field_cuts))

        fit_params = None
        for fit_num, cut in enumerate(field_cuts):
            fd_reso.append(cut)
            offset = 0 if field_cuts[0] < field_cuts[1] else 1
            idx = cut - np.min(field_cuts) - offset
            linecut = Linecut(cut, self, (fbot[idx], ftop[idx]))

            if fit_num == 0:
                _, _, fit_params, report = linecut.one_peak([resonance])
            else:
                _, _, fit_params, report = linecut.one_peak([resonance], fit_params)

            fr_reso.append(fit_params[2])
            Df_reso.append(fit_params[3])
            Df_stdr.append(report.params["m1_Df"].stderr)

        return fd_reso, fr_reso, Df_reso, Df_stdr

    # ------------------------------------------------------------------
    # Denoise / arithmetic
    # ------------------------------------------------------------------
    @timeit
    def denoise_spectrum(self, fnoise: float = 10) -> np.ndarray:
        """Denoise the spectrum with a 2-D FFT low-pass filter.

        Parameters
        ----------
        fnoise : float, optional
            Cut-off frequency (cycles per sample). Larger values keep more
            high-frequency content. Default ``10``.

        Returns
        -------
        ndarray
            Denoised ``S21dd``, shape ``(N_field, N_freq - 1)``.
        """
        S21dd_denoised = np.zeros(np.shape(self.S21dd))
        Zfft = fft2(self.S21dd)
        high_freq_fft = Zfft.copy()
        for i in range(len(S21dd_denoised[:, 0])):
            sample_freq = fftfreq(self.S21dd[:, i].size, d=0.001)
            high_freq_fft[np.where(fnoise < np.abs(sample_freq)), i] = 0
        S21dd_denoised = ifft2(high_freq_fft)
        return S21dd_denoised[:, :-1]

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------
    def _shallow_copy_with(self, new_S21dd: np.ndarray) -> "Spectrum":
        """Return a Spectrum that shares ``B``/``Field``/``Freq`` with ``self``
        but carries a different ``S21dd`` array. Used by the arithmetic
        operators so they work for spectra built via :meth:`from_arrays`
        too (without re-reading anything from disk)."""
        clone = Spectrum.__new__(Spectrum)
        for attr in ("path", "modulation_amp", "skip_field", "skip_freq",
                     "derivative_divide", "derivative", "S_param",
                     "B", "Field", "Freq", "Msat", "saturation"):
            setattr(clone, attr, getattr(self, attr))
        clone.S21dd = new_S21dd
        return clone

    def __add__(self, other: "Spectrum") -> "Spectrum":
        """Return a Spectrum whose ``S21dd`` is ``self.S21dd + other.S21dd``."""
        if np.shape(self.S21dd) != np.shape(other.S21dd):
            raise ValueError(
                f"S21dd shape mismatch: {self.S21dd.shape} vs {other.S21dd.shape}."
            )
        return self._shallow_copy_with(self.S21dd + other.S21dd)

    def __sub__(self, other: "Spectrum") -> "Spectrum":
        """Return a Spectrum whose ``S21dd`` is ``self.S21dd - other.S21dd``."""
        if np.shape(self.S21dd) != np.shape(other.S21dd):
            raise ValueError(
                f"S21dd shape mismatch: {self.S21dd.shape} vs {other.S21dd.shape}."
            )
        return self._shallow_copy_with(self.S21dd - other.S21dd)

    def __mul__(self, num: float) -> "Spectrum":
        """Return a Spectrum whose ``S21dd`` is scaled by ``num``."""
        return self._shallow_copy_with(self.S21dd * num)

    def __truediv__(self, num: float) -> "Spectrum":
        """Return a Spectrum whose ``S21dd`` is divided by ``num``."""
        return self._shallow_copy_with(self.S21dd / num)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        mode = (
            "derivative_divide" if self.derivative_divide
            else ("derivative" if self.derivative else "ΔS21")
        )
        nb = len(self.B)
        nf = self.Freq.shape[0]
        f_lo = float(np.min(self.Freq))
        f_hi = float(np.max(self.Freq))
        b_lo = float(np.min(self.B))
        b_hi = float(np.max(self.B))
        return (
            f"<drlib.Spectrum {self.S_param} N_field={nb} N_freq={nf}, "
            f"B=[{b_lo:g},{b_hi:g}], f=[{f_lo:g},{f_hi:g}] Hz, "
            f"mode='{mode}', saturation={self.saturation}>"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _format_colorbar(self, fig, im, v_min: float, v_max: float) -> None:
        """Attach a standardised colorbar to ``fig`` for image ``im``."""
        cb_ax = fig.add_axes([1.01, 0.135, 0.02, 0.6])
        cbar = fig.colorbar(im, cax=cb_ax, ticks=[v_min, 0, v_max])
        cbar.set_label(r"$Re(\partial_DS_{21}/\partial H)$")
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        cbar.minorticks_on()
        if not self.derivative_divide:
            cbar.set_label(r"$\Delta S_{21}$")

    def _format_axes(self, ax, ax1, nom_locator: int) -> None:
        """Apply the common axis/tick layout used by full-spectrum plots."""
        second = ax1.secondary_xaxis("top")
        second.set_xticks(self.Field[:, 0], self.B, c="blue")
        second.xaxis.set_major_locator(ticker.MultipleLocator(nom_locator))
        ax1.yaxis.set_major_locator(ticker.FixedLocator([min(self.B), max(self.B)]))
        ax1.set_ylabel("$B$ (mT)")
        ax.set_xlabel("No. of Scan")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(nom_locator))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_ylabel("$f$ (GHz)")
