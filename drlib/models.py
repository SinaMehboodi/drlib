"""
drlib.models
============

Analytical line-shape models used by :class:`drlib.Linecut` for fitting.

* :func:`dS21`        – Complex S21 derivative-divide line shape (eq. (5) of
  Maeda *et al.*, AIP Adv. 8, 075302 (2018), modulation-broadened).
* :func:`Lorentzian`  – Bare complex Lorentzian line shape (no modulation),
  with the same calling signature as :func:`dS21` so the two are
  interchangeable inside :class:`drlib.Linecut`.

Both models share the **same parameter order** ::

    (x, A, Psi, fres, Df, mod, Msat, H0)

so a single set of initial parameters can be reused when switching between
the derivative-divide processed data (``dS21``) and the raw / delta-S21
processed data (``Lorentzian``).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------
MU0 = 4.0 * np.pi * 1e-7              # vacuum permeability [T·m/A]
GAMMA = 1.76085963023e11              # gyromagnetic ratio [rad/(s·T)]


# ----------------------------------------------------------------------
# Derivative-divide S21 line shape
# ----------------------------------------------------------------------
def dS21(
    x: np.ndarray,
    A: float,
    Psi: float,
    fres: float,
    Df: float,
    mod: float,
    Msat: float,
    H0: float,
) -> np.ndarray:
    r"""Derivative-divide S21 line shape, evaluated at frequencies ``x``.

    Implements eq. (5) of *Maeda et al., AIP Adv. 8, 075302 (2018)*,
    https://aip.scitation.org/doi/pdf/10.1063/1.5045135.

    The two susceptibility contributions
    :math:`\chi^+(\omega + \omega_\mathrm{mod})` and
    :math:`\chi^-(\omega - \omega_\mathrm{mod})` are combined to form the
    modulation-broadened signal

    .. math::

        \mathrm{model}(\omega) \;=\; \mathrm{Re}\Bigl\{\,
            -\mathrm{i}\,\omega\, A\, e^{\mathrm{i}\Psi}\,
            \frac{\chi^+ - \chi^-}{2\,\omega_\mathrm{mod}}
        \,\Bigr\}^{*}.

    Parameters
    ----------
    x : ndarray
        Frequency axis in **Hz** (despite the legacy doc-string above; the
        function uses ``omega = 2π·x``).
    A : float
        Overall amplitude (arbitrary units).
    Psi : float
        Phase of the complex amplitude (radians).
    fres : float
        Resonance frequency in **Hz**.
    Df : float
        Half-width-at-half-maximum (HWHM) in frequency units (Hz).
    mod : float
        Field-modulation amplitude in **Tesla**.
    Msat : float
        Saturation magnetisation (T).
    H0 : float
        Bias magnetic field at the line cut (T).

    Returns
    -------
    ndarray
        Real part of the model evaluated at ``x``, same shape as ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> from drlib.models import dS21
    >>> f = np.linspace(2e9, 4e9, 401)
    >>> y = dS21(f, A=500.0, Psi=-0.97, fres=3.0e9, Df=0.25e9,
    ...          mod=1e-3, Msat=0.5, H0=0.05)
    """
    omega = 2 * np.pi * x
    omegaRes = 2 * np.pi * fres
    Domega = 2 * np.pi * Df

    modOmega = mod * GAMMA * MU0  # dω/dH ≈ γ µ₀

    Chiplus = (
        GAMMA * MU0 * Msat * (GAMMA * MU0 * H0 - 1j * Domega)
        / (omegaRes ** 2 - (omega + modOmega) ** 2 - 1j * (omega + modOmega) * Domega)
    )
    Chiminus = (
        GAMMA * MU0 * Msat * (GAMMA * MU0 * H0 - 1j * Domega)
        / (omegaRes ** 2 - (omega - modOmega) ** 2 - 1j * (omega - modOmega) * Domega)
    )

    model = np.conjugate(
        -1j * omega * A * np.exp(1j * Psi) * (Chiplus - Chiminus) / (2 * modOmega)
    )

    return np.real(model)


# ----------------------------------------------------------------------
# Bare complex Lorentzian line shape
# ----------------------------------------------------------------------
def Lorentzian(
    x: np.ndarray,
    A: float,
    Psi: float,
    fres: float,
    Df: float,
    mod: float,           # kept for API parity with dS21 (ignored)
    Msat: float,
    H0: float,
) -> np.ndarray:
    r"""Bare complex Lorentzian S21 line shape (no field modulation).

    Used in :class:`drlib.Linecut` when ``Spectrum.derivative_divide=False``
    (i.e. plain :math:`\Delta S_{21}` or :func:`drlib.derivative` data),
    where the field-modulation broadening of :func:`dS21` is absent.

    The shape is the complex susceptibility

    .. math::

        \chi(\omega) \;=\;
        \frac{\gamma\mu_0\,M_s\,(\gamma\mu_0 H_0 - \mathrm{i}\,\Delta\omega)}
             {\omega_\mathrm{res}^2 - \omega^2 - \mathrm{i}\,\omega\,\Delta\omega},

    multiplied by an arbitrary complex amplitude :math:`A\,e^{\mathrm{i}\Psi}`
    and returned as its real part. The ``mod`` argument is accepted (and
    ignored) so the parameter order matches :func:`dS21` exactly — making
    the two functions drop-in replacements inside :class:`drlib.Linecut`.

    Parameters
    ----------
    x : ndarray
        Frequency axis in **Hz**.
    A : float
        Overall amplitude (arbitrary units).
    Psi : float
        Phase of the complex amplitude (radians).
    fres : float
        Resonance frequency (Hz).
    Df : float
        HWHM line width (Hz).
    mod : float
        Ignored — present so the signature matches :func:`dS21`.
    Msat : float
        Saturation magnetisation (T).
    H0 : float
        Bias magnetic field at the cut (T).

    Returns
    -------
    ndarray
        Real part of the model evaluated at ``x``, same shape as ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> from drlib.models import Lorentzian
    >>> f = np.linspace(2e9, 4e9, 401)
    >>> y = Lorentzian(f, A=500.0, Psi=0.0, fres=3.0e9, Df=0.25e9,
    ...                mod=0.0, Msat=0.5, H0=0.05)
    """
    del mod  # explicitly unused — kept for signature parity with dS21
    omega = 2 * np.pi * x
    omegaRes = 2 * np.pi * fres
    Domega = 2 * np.pi * Df

    chi = (
        GAMMA * MU0 * Msat * (GAMMA * MU0 * H0 - 1j * Domega)
        / (omegaRes ** 2 - omega ** 2 - 1j * omega * Domega)
    )

    model = A * np.exp(1j * Psi) * chi
    return np.real(model)
