"""
drlib.field
===========

Analytical magnetic-field tools for coplanar-waveguide (CPW) geometries.

Low-level (Biot-Savart):
    * :func:`indefinite_integral_x` – indefinite x-integral for a rectangular
      conductor cross-section.
    * :func:`integrand`             – definite x-integrand between two limits.
    * :func:`B_BS_analytic`         – analytical Biot-Savart field of a
      rectangular-cross-section conductor carrying current ``I``.

High-level:
    * :class:`CPW` – wraps a 3-conductor signal/ground/signal-line geometry
      and produces real-space field profiles, 2D vector maps, and k-space
      excitation spectra.

All expressions are derived from WolframAlpha indefinite integrals; the
factor of two in front of the integrand is already folded in.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from .math_tools import FFT_1D
from .utils import safe_style as _safe_style


# ----------------------------------------------------------------------
# Low-level Biot-Savart helpers
# ----------------------------------------------------------------------
def indefinite_integral_x(
    x: np.ndarray,
    x0: float,
    z: float,
    z0: float,
    direction: str = "ip",
) -> np.ndarray:
    """Indefinite x-integral of the Biot-Savart integrand for a rectangular wire.

    Parameters
    ----------
    x : ndarray
        Integration variable (m).
    x0 : float
        Field-point x-coordinate (m).
    z : float
        Integration variable for the rectangle's z-edge (m).
    z0 : float
        Field-point z-coordinate (m).
    direction : {'ip', 'oop'}, optional
        Component to compute:

        - ``'ip'``  : in-plane (x) component of the field.
        - ``'oop'`` : out-of-plane (z) component.

    Returns
    -------
    ndarray
        Indefinite integral evaluated at ``x``.
    """
    if direction == "ip":
        result = -(
            (x - x0) * np.log((x - x0) ** 2 + (z - z0) ** 2)
            + 2 * z * np.arctan((x - x0) / (z - z0))
            + 2 * z0 * np.arctan((z - z0) / (x - x0))
        )
    elif direction == "oop":
        result = (z - z0) * np.log((x - x0) ** 2 + (z - z0) ** 2) + 2 * (x - x0) * np.arctan(
            (z - z0) / (x - x0)
        )
    else:
        raise ValueError("direction must be 'ip' or 'oop'.")
    return result


def integrand(
    x_upper_boundary: float,
    x_lower_boundary: float,
    x0: float,
    z: float,
    z0: float,
    direction: str = "ip",
) -> np.ndarray:
    """Definite x-integral between two boundaries.

    Convenience wrapper around :func:`indefinite_integral_x` that evaluates
    ``F(x_upper) - F(x_lower)``.

    Parameters
    ----------
    x_upper_boundary, x_lower_boundary : float
        Integration limits in x (m).
    x0, z, z0 : float
        See :func:`indefinite_integral_x`.
    direction : {'ip', 'oop'}, optional
        Component selector.

    Returns
    -------
    ndarray
    """
    return indefinite_integral_x(
        x_upper_boundary, x0, z, z0, direction=direction
    ) - indefinite_integral_x(x_lower_boundary, x0, z, z0, direction=direction)


def B_BS_analytic(
    x0: np.ndarray,
    z0: float,
    thickness: float,
    width: float,
    I: float = 4.5e-3,
    direction: str = "ip",
) -> np.ndarray:
    r"""Analytical Biot-Savart field of a finite-cross-section conductor.

    Computes :math:`B_x` (``direction='ip'``) or :math:`B_z`
    (``direction='oop'``) at the field point :math:`(x_0, z_0)` above a
    rectangular conductor of given ``thickness`` × ``width`` carrying a
    uniform current ``I``. The conductor centre is at the origin.

    Parameters
    ----------
    x0 : ndarray
        Field-point x-coordinate(s) (m).
    z0 : float
        Field-point z-coordinate (m).
    thickness : float
        Conductor thickness (m).
    width : float
        Conductor width (m).
    I : float, optional
        Total current (A). Default 4.5 mA.
    direction : {'ip', 'oop'}, optional
        Component to compute.

    Returns
    -------
    ndarray
        Magnetic-flux-density component, in Tesla.
    """
    x_lower_boundary = -width / 2
    x_upper_boundary = +width / 2
    z_lower_boundary = -thickness / 2
    z_upper_boundary = +thickness / 2

    mu0 = 4 * np.pi * 1e-7
    J = I / (width * thickness)

    field = integrand(
        x_upper_boundary, x_lower_boundary, x0, z_upper_boundary, z0, direction=direction
    ) - integrand(
        x_upper_boundary, x_lower_boundary, x0, z_lower_boundary, z0, direction=direction
    )

    return mu0 / (4 * np.pi) * J * field


# ----------------------------------------------------------------------
# CPW class
# ----------------------------------------------------------------------
class CPW:
    """Coplanar-waveguide magnetic-field model.

    Geometry (top view, x is along the propagation direction)::

        |--ground--|--gap--|--signal--|--gap--|--ground--|
              I/2          I            I/2

    All conductors share the same ``thickness``.  The signal line is centred
    at the origin and carries current ``I``; the two ground lines each
    carry ``-I/2``.

    Parameters
    ----------
    current : float
        Total current through the signal line (A).
    signal_line : float
        Signal-line width (m).
    gap : float
        Gap between signal and each ground line (m).
    ground : float
        Ground-line width (m).
    thickness : float
        Common conductor thickness (m).

    Examples
    --------
    >>> cpw = CPW(current=10e-3, signal_line=2e-6, gap=1e-6,
    ...           ground=10e-6, thickness=200e-9)
    >>> cpw.get_Bdistribution(distance=210e-9)
    """

    def __init__(
        self,
        current: float,
        signal_line: float,
        gap: float,
        ground: float,
        thickness: float,
    ) -> None:
        self.I = current
        self.signal_line = signal_line
        self.gap = gap
        self.ground = ground
        self.thickness = thickness

    # ------------------------------------------------------------------
    # 1-D field profile vs lateral position
    # ------------------------------------------------------------------
    def get_Bdistribution(
        self,
        distance: float = 210e-9,
        number_of_points: int = 100_000,
        style: str = "Beam",
        **kw,
    ) -> None:
        r"""Plot :math:`h_\zeta(x)` and :math:`h_\xi(x)` at a fixed height.

        Parameters
        ----------
        distance : float, optional
            Height ``z₀`` above the CPW plane at which the field is sampled
            (m). Default ``210 nm``.
        number_of_points : int, optional
            Number of lateral samples along ``x``. Default ``100 000``.
        style : str, optional
            Matplotlib style sheet name. Default ``"Beam"``.
        **kw
            Forwarded to ``plt.plot``.

        Notes
        -----
        The plot also shows golden rectangles representing the ground and
        signal conductors, so the field profile can be read against the
        physical layout.
        """
        _safe_style(style)
        gold = (255 / 255, 215 / 255, 0 / 255, 100 / 100)
        z0 = distance
        x0 = np.linspace(
            -(self.signal_line + self.gap + self.ground),
            +(self.signal_line + self.gap + self.ground),
            number_of_points,
        )

        B_ip_CPW = (
            B_BS_analytic(x0, z0, self.thickness, self.signal_line, I=self.I)
            - B_BS_analytic(
                x0 - (self.signal_line / 2 + self.gap + self.ground / 2),
                z0, self.thickness, self.ground, I=self.I / 2,
            )
            - B_BS_analytic(
                x0 + (self.signal_line / 2 + self.gap + self.ground / 2),
                z0, self.thickness, self.ground, I=self.I / 2,
            )
        )
        B_oop_CPW = (
            B_BS_analytic(x0, z0, self.thickness, self.signal_line, I=self.I, direction="oop")
            - B_BS_analytic(
                x0 - (self.signal_line / 2 + self.gap + self.ground / 2),
                z0, self.thickness, self.ground, I=self.I / 2, direction="oop",
            )
            - B_BS_analytic(
                x0 + (self.signal_line / 2 + self.gap + self.ground / 2),
                z0, self.thickness, self.ground, I=self.I / 2, direction="oop",
            )
        )

        plt_rng = int((self.signal_line + self.gap + self.ground) * 1e6)
        rec_dim = number_of_points / plt_rng
        off_dim = self.signal_line / 4 * 1e6 * rec_dim
        ground_dim = self.ground / 2 * 1e6 * rec_dim
        gap_dim = self.gap / 2 * 1e6 * rec_dim
        signal_dim = self.signal_line / 2 * 1e6 * rec_dim

        fig, ax = plt.subplots()
        ax.plot(B_ip_CPW, label=r"$h_\zeta$", **kw)
        ax.plot(B_oop_CPW, label=r"$h_\xi$", **kw)
        ax.set_xticks(
            [0, number_of_points / 4, number_of_points / 2,
             3 * number_of_points / 4, number_of_points],
            [-plt_rng, -plt_rng / 2, 0, plt_rng / 2, plt_rng],
        )
        # Conductor rectangles
        for x_offset in (off_dim,
                         off_dim + ground_dim + gap_dim,
                         off_dim + 2 * gap_dim + signal_dim + ground_dim):
            w = signal_dim if x_offset == off_dim + ground_dim + gap_dim else ground_dim
            ax.add_patch(
                patches.Rectangle(
                    (x_offset, -0.005), w, 0.01,
                    linewidth=1, facecolor=gold, edgecolor="black",
                )
            )
        ax.set_xlabel(r"$d\,(\mathit{\mu} m)$")
        ax.set_ylabel(r"$\mathit{\mu}_0 h\,\, (mT)$")
        plt.legend()

    # ------------------------------------------------------------------
    # 2-D vector / contour map of the field
    # ------------------------------------------------------------------
    def get_Bspectrum(
        self,
        distance_range: List[float] = (-1e-6, 1e-6),
        number_points_XY: List[int] = (1000, 50),
        show_CPW: bool = True,
        style: str = "Beam",
        plot: str = "quiver",
        **kw,
    ) -> None:
        """Plot a 2-D map of the CPW magnetic field above the conductors.

        Parameters
        ----------
        distance_range : (float, float), optional
            ``(z_min, z_max)`` heights above the plane in metres.
            Default ``(-1, +1) µm``.
        number_points_XY : (int, int), optional
            Number of samples ``(N_x, N_z)``. Default ``(1000, 50)``.
        show_CPW : bool, optional
            If ``True`` (default), draw golden rectangles for the conductors
            on top of the map.
        style : str, optional
            Matplotlib style sheet. Default ``"Beam"``.
        plot : {'quiver', 'streamplot', 'contour', 'contourf'}, optional
            Visualisation type for the 2-D field.
        **kw
            Forwarded to the chosen matplotlib plotting function.
        """
        Z0 = np.linspace(distance_range[0], distance_range[1], number_points_XY[1])
        x0 = np.linspace(
            -(self.signal_line + self.gap + self.ground),
            +(self.signal_line + self.gap + self.ground),
            number_points_XY[0],
        )

        B_ip_CPW_map: List[np.ndarray] = []
        B_oop_CPW_map: List[np.ndarray] = []
        for z0 in Z0:
            B_ip = (
                B_BS_analytic(x0, z0, self.thickness, self.signal_line, I=self.I)
                - B_BS_analytic(
                    x0 - (self.signal_line / 2 + self.gap + self.ground / 2),
                    z0, self.thickness, self.ground, I=self.I / 2,
                )
                - B_BS_analytic(
                    x0 + (self.signal_line / 2 + self.gap + self.ground / 2),
                    z0, self.thickness, self.ground, I=self.I / 2,
                )
            )
            B_oop = (
                B_BS_analytic(x0, z0, self.thickness, self.signal_line,
                              I=self.I, direction="oop")
                - B_BS_analytic(
                    x0 - (self.signal_line / 2 + self.gap + self.ground / 2),
                    z0, self.thickness, self.ground, I=self.I / 2, direction="oop",
                )
                - B_BS_analytic(
                    x0 + (self.signal_line / 2 + self.gap + self.ground / 2),
                    z0, self.thickness, self.ground, I=self.I / 2, direction="oop",
                )
            )
            B_ip_CPW_map.append(B_ip)
            B_oop_CPW_map.append(B_oop)

        plt_rng = int((self.signal_line + self.gap + self.ground) * 1e6)
        rec_dim = number_points_XY[0] / plt_rng
        off_dim = self.signal_line / 4 * 1e6 * rec_dim
        ground_dim = self.ground / 2 * 1e6 * rec_dim
        gap_dim = self.gap / 2 * 1e6 * rec_dim
        signal_dim = self.signal_line / 2 * 1e6 * rec_dim

        _safe_style(style)
        gold = (255 / 255, 215 / 255, 0 / 255, 100 / 100)
        fig, ax = plt.subplots()

        B_ip_CPW_map = np.array(B_ip_CPW_map)
        B_oop_CPW_map = np.array(B_oop_CPW_map)
        x = np.arange(B_ip_CPW_map.shape[1])
        y = np.arange(B_ip_CPW_map.shape[0])
        X, Y = np.meshgrid(x, y)

        U = B_ip_CPW_map
        V = B_oop_CPW_map
        M = np.hypot(U, V)

        if plot == "quiver":
            ax.quiver(X, Y, U, V, M, units="xy", pivot="mid", scale_units="xy", **kw)
        elif plot == "streamplot":
            ax.streamplot(X, Y, U, V, **kw)
        elif plot == "contour":
            ax.contour(X, Y, M, **kw)
        elif plot == "contourf":
            ax.contourf(X, Y, M, **kw)
        else:
            raise ValueError(
                "plot must be one of 'quiver', 'streamplot', 'contour', 'contourf'."
            )

        ax.set_xticks(
            [0, number_points_XY[0] / 4, number_points_XY[0] / 2,
             3 * number_points_XY[0] / 4, number_points_XY[0]],
            [-plt_rng, -plt_rng / 2, 0, plt_rng / 2, plt_rng],
        )
        ax.set_yticks(
            [0, number_points_XY[1] / 4, number_points_XY[1] / 2,
             3 * number_points_XY[1] / 4, number_points_XY[1]],
            [distance_range[0] * 1e6, distance_range[0] * 1e6 / 2, 0,
             distance_range[1] * 1e6 / 2, distance_range[1] * 1e6],
        )

        if show_CPW:
            for x_offset, w in [
                (off_dim, ground_dim),
                (off_dim + ground_dim + gap_dim, signal_dim),
                (off_dim + 2 * gap_dim + signal_dim + ground_dim, ground_dim),
            ]:
                ax.add_patch(
                    patches.Rectangle(
                        (x_offset, number_points_XY[1] / 2 - 0.5),
                        w, 1,
                        linewidth=1, facecolor=gold, edgecolor="black",
                    )
                )

        ax.set_xlabel(r"$dx\,(\mu m)$")
        ax.set_ylabel(r"$dy\,(\mu m)$")

    # ------------------------------------------------------------------
    # k-space excitation spectrum
    # ------------------------------------------------------------------
    def get_Kspectrum(
        self,
        number_of_points: int = 1000,
        distance: float = 210e-9,
        IP_excitation: bool = True,
        OP_excitation: bool = True,
        style: str = "Beam",
        K_range: List[float] = (0, 5),
        **kw,
    ) -> None:
        r"""Plot the FFT of the CPW field — the spin-wave excitation spectrum.

        Parameters
        ----------
        number_of_points : int, optional
            Lateral samples used to evaluate the field before FFT.
            Default ``1000``.
        distance : float, optional
            Sampling height (m). Default ``210 nm``.
        IP_excitation, OP_excitation : bool, optional
            Toggle plotting of the in-plane (``h_ζ``) and out-of-plane
            (``h_ξ``) excitation efficiencies. Default both ``True``.
        style : str, optional
            Matplotlib style sheet. Default ``"Beam"``.
        K_range : (float, float), optional
            Wavenumber axis range (``µm⁻¹``). Default ``(0, 5)``.
        **kw
            Forwarded to ``plt.plot``.
        """
        _safe_style(style)
        z0 = distance
        x0 = np.linspace(
            -(self.signal_line + self.gap + self.ground),
            +(self.signal_line + self.gap + self.ground),
            number_of_points,
        )
        zero_pad = 6 * len(x0)
        dx = (x0[2] - x0[1]) * 1e6  # m -> µm

        B_ip_CPW = (
            B_BS_analytic(x0, z0, self.thickness, self.signal_line, I=self.I)
            - B_BS_analytic(
                x0 - (self.signal_line / 2 + self.gap + self.ground / 2),
                z0, self.thickness, self.ground, I=self.I / 2,
            )
            - B_BS_analytic(
                x0 + (self.signal_line / 2 + self.gap + self.ground / 2),
                z0, self.thickness, self.ground, I=self.I / 2,
            )
        )
        B_oop_CPW = (
            B_BS_analytic(x0, z0, self.thickness, self.signal_line, I=self.I, direction="oop")
            - B_BS_analytic(
                x0 - (self.signal_line / 2 + self.gap + self.ground / 2),
                z0, self.thickness, self.ground, I=self.I / 2, direction="oop",
            )
            - B_BS_analytic(
                x0 + (self.signal_line / 2 + self.gap + self.ground / 2),
                z0, self.thickness, self.ground, I=self.I / 2, direction="oop",
            )
        )

        fig, ax = plt.subplots()
        if IP_excitation:
            spectrum_CPW, k_ip_CPW, _ = FFT_1D(B_ip_CPW, dx, zero_pad=zero_pad)
            ax.plot(
                k_ip_CPW * 1e-6, np.abs(spectrum_CPW),
                label=r"{0:.0f}$\mu$m-{1:.0f}$\mu$m-{2:.0f}$\mu$m".format(
                    self.signal_line * 1e6, self.gap * 1e6, self.ground * 1e6
                ),
                **kw,
            )
            ax.set_xlabel(r"$k_\mathrm{\zeta}$ ($\mu$m$^{-1}$)")
            ax.set_ylabel(r"$|\mathrm{FFT}(\mu_0 h_\mathrm{\zeta})|$ (arb.u.)")
            ax.set_xlim(K_range[0], K_range[1])
            plt.legend()
        if OP_excitation:
            spectrum_CPW, k_op_CPW, _ = FFT_1D(B_oop_CPW, dx, zero_pad=zero_pad)
            ax.plot(
                k_op_CPW * 1e-6, np.abs(spectrum_CPW),
                label=r"{0:.0f}$\mu$m-{1:.0f}$\mu$m-{2:.0f}$\mu$m".format(
                    self.signal_line * 1e6, self.gap * 1e6, self.ground * 1e6
                ),
                **kw,
            )
            ax.set_xlabel(r"$k_\mathrm{\xi}$ ($\mu$m$^{-1}$)")
            ax.set_ylabel(r"$|\mathrm{FFT}(\mu_0 h_\mathrm{\xi})|$ (arb.u.)")
            ax.set_xlim(K_range[0], K_range[1])
            plt.legend()
        if IP_excitation and OP_excitation:
            ax.set_xlabel(r"$k_\mathrm{(\zeta,\xi)}$ ($\mu$m$^{-1}$)")
            ax.set_ylabel(r"$|\mathrm{FFT}(\mu_0 h_\mathrm{(\zeta,\xi)})|$ (arb.u.)")
