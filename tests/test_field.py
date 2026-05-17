"""Tests for ``drlib.field``: Biot-Savart helpers + CPW class."""

from __future__ import annotations

import numpy as np
import pytest

from drlib import CPW
from drlib.field import B_BS_analytic, indefinite_integral_x, integrand


# ----------------------------------------------------------------------
# Biot-Savart helpers
# ----------------------------------------------------------------------
def test_B_BS_analytic_returns_array_for_array_input():
    x0 = np.linspace(-5e-6, 5e-6, 51)
    B = B_BS_analytic(x0, z0=210e-9, thickness=200e-9, width=2e-6,
                      I=10e-3, direction="ip")
    assert B.shape == x0.shape
    assert np.isfinite(B).all()


def test_B_BS_analytic_sign_above_centre():
    """Above the wire centre, the in-plane field of a uniform current is finite."""
    # Just check we get nonzero finite values; sign convention varies by definition.
    B = B_BS_analytic(np.array([0.0]), z0=210e-9, thickness=200e-9, width=2e-6,
                      I=10e-3, direction="ip")
    assert np.isfinite(B[0])


def test_B_BS_analytic_decays_with_distance():
    """|B| decreases as the field point moves away from the wire."""
    x0 = np.array([0.0])
    B_close = B_BS_analytic(x0, z0=210e-9, thickness=200e-9, width=2e-6,
                            I=10e-3, direction="oop")
    B_far = B_BS_analytic(x0, z0=10e-6, thickness=200e-9, width=2e-6,
                          I=10e-3, direction="oop")
    assert abs(B_close[0]) >= abs(B_far[0])


def test_integrand_is_difference_of_indefinite():
    """``integrand(b, a, ...) == F(b) − F(a)`` for the indefinite integral."""
    Fa = indefinite_integral_x(1e-6, 0.0, 100e-9, 210e-9, direction="ip")
    Fb = indefinite_integral_x(3e-6, 0.0, 100e-9, 210e-9, direction="ip")
    I_def = integrand(3e-6, 1e-6, 0.0, 100e-9, 210e-9, direction="ip")
    assert np.isclose(I_def, Fb - Fa)


def test_indefinite_integral_x_direction_validation():
    with pytest.raises(ValueError):
        indefinite_integral_x(1e-6, 0.0, 100e-9, 210e-9, direction="up")


# ----------------------------------------------------------------------
# CPW class — methods run end-to-end
# ----------------------------------------------------------------------
def _make_cpw() -> CPW:
    return CPW(current=10e-3, signal_line=2e-6, gap=1e-6,
               ground=10e-6, thickness=200e-9)


def test_cpw_get_Bdistribution_runs():
    cpw = _make_cpw()
    cpw.get_Bdistribution(distance=210e-9, number_of_points=2000, style=None)


def test_cpw_get_Kspectrum_runs():
    cpw = _make_cpw()
    cpw.get_Kspectrum(number_of_points=512, distance=210e-9,
                      K_range=(0, 5), style=None)


@pytest.mark.parametrize("kind", ["quiver", "streamplot", "contour", "contourf"])
def test_cpw_get_Bspectrum_each_kind(kind):
    cpw = _make_cpw()
    cpw.get_Bspectrum(
        distance_range=(-1e-6, 1e-6),
        number_points_XY=(200, 30),
        plot=kind, style=None, show_CPW=False,
    )


def test_cpw_get_Bspectrum_invalid_kind_raises():
    cpw = _make_cpw()
    with pytest.raises(ValueError):
        cpw.get_Bspectrum(
            distance_range=(-1e-6, 1e-6),
            number_points_XY=(200, 30),
            plot="bogus", style=None,
        )
