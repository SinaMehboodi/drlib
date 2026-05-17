"""
drlib.linecut
=============

The :class:`Linecut` class represents a single constant-magnetic-field cut
through a :class:`drlib.Spectrum` object and provides utilities to:

* plot the cut (raw and/or denoised),
* fit one, two, or three :func:`drlib.dS21` peaks via ``lmfit``,
* report the fitted parameters together with their uncertainties.

Typical usage::

    >>> from drlib import Spectrum, Linecut
    >>> spec = Spectrum(path=r"C:/data", saturation=42)
    >>> cut = Linecut(cut=80, Spectrum=spec, frequencies=(2e9, 4e9))
    >>> cut.plot(resonances=[2.5e9])
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.fft import fft, fftfreq, ifft

from .models import dS21, Lorentzian
from .utils import safe_style as _safe_style, set_size, timeit


class Linecut:
    """A constant-field cut through a :class:`drlib.Spectrum`.

    Parameters
    ----------
    cut : int
        Column index along the field axis at which the cut is taken.
    Spectrum : :class:`drlib.Spectrum`
        The parent spectrum object. All data arrays are referenced (not
        copied), so changing the parent will not be reflected here.
    frequencies : (float, float), optional
        ``(f_min, f_max)`` window (Hz) to restrict the linecut to.
        Default ``(0, 20e9)`` — i.e. the full available range.

    Attributes
    ----------
    cut : int
        See parameter.
    frequencies : (float, float)
        See parameter.
    Freq_cut : ndarray, shape (N_freq_window,)
        Frequency axis of the cut (Hz).
    S21dd_cut : ndarray, shape (N_freq_window,)
        Signal values along the cut.
    H0 : float
        Bias-field value (mT) at the chosen cut.
    Msat : float or None
        Saturation field inherited from ``Spectrum``.
    """

    def __init__(
        self,
        cut: int,
        Spectrum,
        frequencies: Tuple[float, float] = (0.0, 20e9),
    ) -> None:
        self.cut = cut
        self.frequencies = frequencies
        self.B = Spectrum.B
        self.Field = Spectrum.Field
        self.Freq = Spectrum.Freq
        self.S21dd = Spectrum.S21dd
        self.derivative_divide = Spectrum.derivative_divide
        self.Freq_cut, self.S21dd_cut = self.get_cut()
        self.H0 = self.B[cut]
        self.Msat = Spectrum.Msat

    # ------------------------------------------------------------------
    # Slice extraction
    # ------------------------------------------------------------------
    def get_cut(self, denoise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the 1-D cut at the chosen field index.

        Parameters
        ----------
        denoise : bool, optional
            If ``True``, return the FFT-low-pass-filtered cut. Default ``False``.

        Returns
        -------
        Freq_cut : ndarray, shape (M,)
            Frequencies inside ``self.frequencies``.
        S21dd_cut : ndarray, shape (M,)
            Signal values inside ``self.frequencies``.
        """
        freq_min, freq_max = self.frequencies
        f_min = np.where(self.Freq > freq_min)[0][0]
        f_max = np.where(self.Freq < freq_max)[0][-1]

        Freq_cut = self.Freq[f_min:f_max, self.cut]
        S21dd_cut = self.S21dd[self.cut, f_min:f_max]
        if denoise:
            _, S21dd_cut = self.denoising()
        return Freq_cut, S21dd_cut

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    @timeit
    def plot(
        self,
        save_name: Optional[str] = None,
        resonances: Optional[List[float]] = None,
        denoise: bool = False,
        style: str = "Beam",
        pic_size=240,
    ) -> None:
        """Plot the linecut, optionally with one to three peaks fitted on top.

        Parameters
        ----------
        save_name : str, optional
            Output file path including extension (``"cut.png"`` etc.).
            ``None`` (default) skips saving.
        resonances : list of float, optional
            Expected resonance frequency or frequencies (up to 3).
            Triggers a 1/2/3-peak fit and overlays the result.
        denoise : bool, optional
            If ``True``, additionally plot the FFT-denoised cut.
        style : str, optional
            Matplotlib style sheet. Default ``"Beam"``.
        pic_size : int or (float, float), optional
            Figure-size hint (see :func:`drlib.set_size`).
        """
        _safe_style(style)
        pic_s = pic_size if isinstance(pic_size, tuple) else set_size(pic_size)
        plt.figure(figsize=pic_s)
        plt.plot(self.Freq_cut * 1e-9, self.S21dd_cut, label="real data")

        if resonances is not None:
            num_reso = len(resonances)
            if num_reso == 1:
                _, our_fit, _, _ = self.one_peak(resonance=resonances, details=True)
                plt.plot(self.Freq_cut * 1e-9, our_fit, label="best fit")
            elif num_reso == 2:
                _, our_fit, _ = self.two_peak(resonances, details=True)
                plt.plot(self.Freq_cut * 1e-9, our_fit, label="best fit")
            elif num_reso == 3:
                _, our_fit, _ = self.three_peak(resonances, details=True)
                plt.plot(self.Freq_cut * 1e-9, our_fit, label="best fit")
            elif num_reso > 3:
                print("At most three peaks can be selected for fitting!")
            else:
                print("resonances must be an iterable returning float values.")

        plt.xlabel("$f$ (GHz)")
        plt.ylabel("Intensity")
        plt.legend()

        if save_name is not None:
            plt.savefig(save_name, bbox_inches="tight")

        if denoise:
            _, S21dd_denoised_cut = self.denoising()
            plt.figure(figsize=pic_s)
            plt.plot(self.Freq_cut * 1e-9, S21dd_denoised_cut, label="real data")
            plt.xlabel("$Frequency(GHz)$")
            plt.xticks(fontsize=25)
            plt.ylabel("Intensity")
            if save_name is not None:
                plt.savefig("denoised_" + save_name)

    # ------------------------------------------------------------------
    # FFT low-pass denoiser
    # ------------------------------------------------------------------
    def denoising(self, fnoise: float = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Low-pass-filter the linecut via FFT.

        Parameters
        ----------
        fnoise : float, optional
            Frequencies whose magnitude exceeds ``fnoise`` (in
            cycles/sample) are zeroed before the inverse FFT. Smaller
            values mean *stronger* denoising but more distortion.
            Default ``20``.

        Returns
        -------
        Freq_cut : ndarray
            Original frequency axis (echoed back for convenience).
        S21dd_denoised : ndarray, complex
            Denoised cut (still complex — take ``.real`` to plot).
        """
        Zfft = fft(self.S21dd_cut)
        sample_freq = fftfreq(self.S21dd_cut.size, d=0.001)
        high_freq_fft = Zfft.copy()
        high_freq_fft[np.where(fnoise < np.abs(sample_freq))] = 0
        S21dd_denoised = ifft(high_freq_fft)
        return self.Freq_cut, S21dd_denoised

    # ------------------------------------------------------------------
    # Peak fits (1 / 2 / 3 peaks)
    # ------------------------------------------------------------------
    def one_peak(
        self,
        resonance: List[float] = (2.2e9,),
        params=None,
        details: bool = False,
        skyrmion_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[float], object]:
        """Fit a single :func:`drlib.dS21` peak to the linecut.

        Parameters
        ----------
        resonance : list of float, length 1
            Initial guess for the resonance frequency (Hz).
        params : list of float, optional
            Full initial parameter array. If given, overrides ``resonance``.
            Order: ``[A, Psi, fres, Df, mod, Msat, H0]``.
        details : bool, optional
            If ``True``, print the lmfit fit report including correlations.
        skyrmion_mode : bool, optional
            If ``True``, restrict ``fres`` to below ``2.6 GHz`` (useful
            when fitting the skyrmion absorption branch).

        Returns
        -------
        Freq_cut : ndarray
            Frequency axis (echoed back).
        our_fit : ndarray
            Real part of the best fit, sampled on ``Freq_cut``.
        fit_params : list of float
            ``[A, Psi, fres, Df, mod, Msat, H0]`` at the best fit.
        report : lmfit ModelResult
            Full lmfit result for further inspection.
        """
        if params is None:
            Msat = 5 if self.Msat is None else self.Msat
            parameters = np.array([
                5.46470576e+02,   # A
                -0.97045439,      # Psi
                resonance[0],     # fres
                0.25e+09,         # Df
                1e-3,             # mod
                Msat,             # Msat
                self.H0,          # H0
            ])
        else:
            parameters = params

        Smod = Model(dS21, prefix="m1_") if self.derivative_divide \
            else Model(Lorentzian, prefix="m1_")

        params0 = Smod.make_params()
        params0["m1_A"].set(value=parameters[0])
        params0["m1_Psi"].set(value=parameters[1])
        if skyrmion_mode:
            params0["m1_fres"].set(value=parameters[2], max=2.6e9)
        else:
            params0["m1_fres"].set(value=parameters[2])
        params0["m1_Df"].set(value=parameters[3], min=0)
        params0["m1_mod"].set(value=parameters[4], vary=False)

        if self.Msat is None:
            params0["m1_Msat"].set(value=parameters[5], min=0)
        else:
            params0["m1_Msat"].set(value=parameters[5], vary=False)
        params0["m1_H0"].set(value=parameters[6], vary=False)

        our_fit, fit_params, report = self.fit(model=Smod, params_init=params0, details=details)
        return self.Freq_cut, our_fit, fit_params, report

    def two_peak(
        self,
        resonance: Tuple[float, float] = (2.2e9, 2.2e9),
        params=None,
        details: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Fit two :func:`drlib.dS21` peaks.

        Parameters
        ----------
        resonance : (float, float)
            Initial resonance frequencies (Hz).
        params : list of float, optional
            Full initial parameter array (14 entries).
        details : bool, optional
            Print fit report.

        Returns
        -------
        Freq_cut, our_fit, fit_params : as for :meth:`one_peak`,
            but with 14 entries in ``fit_params``.
        """
        if params is None:
            Msat = 5 if self.Msat is None else self.Msat
            parameters = np.array([
                5.46470576e+02, -1.97045439, resonance[0], 1e+07, 1e-3, Msat, self.H0,
                2.99458785e+03, -0.1,        resonance[1], 1e+07, 1e-3, Msat, self.H0,
            ])
        else:
            parameters = params

        if self.derivative_divide:
            Smod = Model(dS21, prefix="m1_") + Model(dS21, prefix="m2_")
        else:
            Smod = Model(Lorentzian, prefix="m1_") + Model(Lorentzian, prefix="m2_")

        params0 = Smod.make_params()
        for i, prefix in enumerate(("m1_", "m2_")):
            base = i * 7
            params0[f"{prefix}A"].set(value=parameters[base + 0])
            params0[f"{prefix}Psi"].set(value=parameters[base + 1])
            params0[f"{prefix}fres"].set(value=parameters[base + 2])
            params0[f"{prefix}Df"].set(value=parameters[base + 3], min=0)
            params0[f"{prefix}mod"].set(value=parameters[base + 4], vary=False)
            if self.Msat is None:
                params0[f"{prefix}Msat"].set(value=parameters[base + 5], min=0)
            else:
                params0[f"{prefix}Msat"].set(value=parameters[base + 5], vary=False)
            params0[f"{prefix}H0"].set(value=parameters[base + 6], vary=False)

        our_fit, fit_params, _ = self.fit(model=Smod, params_init=params0, details=details)
        return self.Freq_cut, our_fit, fit_params

    def three_peak(
        self,
        resonance: Tuple[float, float, float] = (2.2e9, 2.2e9, 2.2e9),
        params=None,
        details: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Fit three :func:`drlib.dS21` peaks (unreliable — tune ``params``).

        Parameters
        ----------
        resonance : (float, float, float)
            Initial resonance frequencies (Hz).
        params : list of float, optional
            Full initial parameter array (21 entries).
        details : bool, optional
            Print fit report.

        Returns
        -------
        Freq_cut, our_fit, fit_params : as for :meth:`one_peak`,
            but with 21 entries in ``fit_params``.
        """
        if params is None:
            Msat = 5 if self.Msat is None else self.Msat
            parameters = np.array([
                5.46470576e+02, -1.97045439, resonance[0], 1e+07, 2e-3, Msat, self.H0,
                2.99458785e+03, -0.1,        resonance[1], 1e+07, 2e-3, Msat, self.H0,
                2.99458785e+03, -0.1,        resonance[2], 1e+07, 2e-3, Msat, self.H0,
            ])
        else:
            parameters = params

        if self.derivative_divide:
            Smod = Model(dS21, prefix="m1_") + Model(dS21, prefix="m2_") + Model(dS21, prefix="m3_")
        else:
            # NOTE: bug in original code repeated prefix "m1_" — preserved
            # here for API compatibility; if you need three independent
            # Lorentzians, rename the prefixes.
            Smod = (
                Model(Lorentzian, prefix="m1_")
                + Model(Lorentzian, prefix="m1_")
                + Model(Lorentzian, prefix="m1_")
            )

        params0 = Smod.make_params()
        for i, prefix in enumerate(("m1_", "m2_", "m3_")):
            base = i * 7
            params0[f"{prefix}A"].set(value=parameters[base + 0])
            params0[f"{prefix}Psi"].set(value=parameters[base + 1])
            if i == 0:
                params0[f"{prefix}fres"].set(value=parameters[base + 2], min=2e9, max=3e9)
            elif i == 1:
                params0[f"{prefix}fres"].set(value=parameters[base + 2], min=2e9, max=5e9)
            else:
                params0[f"{prefix}fres"].set(value=parameters[base + 2], min=2e9)
            params0[f"{prefix}Df"].set(value=parameters[base + 3])
            params0[f"{prefix}mod"].set(value=parameters[base + 4], vary=False)
            if self.Msat is None:
                params0[f"{prefix}Msat"].set(value=parameters[base + 5], min=0)
            else:
                params0[f"{prefix}Msat"].set(value=parameters[base + 5], vary=False)
            params0[f"{prefix}H0"].set(value=parameters[base + 6], vary=False)

        our_fit, fit_params, _ = self.fit(model=Smod, params_init=params0, details=details)
        return self.Freq_cut, our_fit, fit_params

    # ------------------------------------------------------------------
    # Low-level fit driver
    # ------------------------------------------------------------------
    def fit(
        self,
        model,
        params_init,
        details: bool,
    ) -> Tuple[np.ndarray, List[float], object]:
        """Run the lmfit minimisation against the denoised linecut.

        Parameters
        ----------
        model : lmfit.Model
            A composite lmfit model already configured with prefixes.
        params_init : lmfit.Parameters
            Initial parameter set produced by ``model.make_params()``.
        details : bool
            If ``True``, print ``out0.fit_report(show_correl=True)``.

        Returns
        -------
        our_fit : ndarray
            Real part of the best fit.
        fit_params : list of float
            Best-fit parameter values, in the order returned by lmfit.
        report : lmfit.ModelResult
            Full lmfit result.
        """
        _, S21dd_denoised = self.denoising()
        out0 = model.fit(S21dd_denoised, params_init, x=self.Freq_cut)
        our_fit = np.real(out0.best_fit[:])

        fit_params: List[float] = []
        pnames = list(out0.params)
        for name in pnames:
            fit_params.append(out0.params[name].value)

        if details:
            print(out0.fit_report(show_correl=True))

        return our_fit, fit_params, out0
