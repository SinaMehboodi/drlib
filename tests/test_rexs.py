"""Tests for ``drlib.REXS2D``: m_z generation, FFT, peak locations, savefig."""

from __future__ import annotations

import numpy as np
import pytest

from drlib import REXS2D


# ----------------------------------------------------------------------
# Skyrmion lattice
# ----------------------------------------------------------------------
def test_skyrmion_lattice_mz_in_range_and_q_peak():
    """Skyrmion-lattice FFT has a peak near |q| = 2π/a."""
    a_nm = 60.0
    sim = REXS2D(N=128, L_um=1.0, pitch_nm=a_nm, a_nm=a_nm, K=1, seed=0,
                 use_envelope=False)
    mz = sim.build_skyrmion_mz()
    assert mz.shape == (128, 128)
    assert mz.min() >= -1.0 - 1e-9 and mz.max() <= 1.0 + 1e-9

    Iq, qx, qy = sim.compute_qspace()
    qx2, qy2 = np.meshgrid(qx, qy, indexing="xy")
    q = np.hypot(qx2, qy2)
    Iq_masked = Iq.copy()
    Iq_masked[q < 1e-3] = 0.0          # mask DC
    peak_q = q[np.unravel_index(np.argmax(Iq_masked), Iq.shape)]
    expected = 2 * np.pi / a_nm
    # Within ~25 % of the expected radial position (coarse grid).
    assert abs(peak_q - expected) / expected < 0.25


# ----------------------------------------------------------------------
# Helical state
# ----------------------------------------------------------------------
def test_helical_mz_peak_direction():
    """Helical FFT peak lies along the propagation direction."""
    pitch_nm = 60.0
    theta = 30.0
    sim = REXS2D(N=128, L_um=1.0, pitch_nm=pitch_nm, K=1, seed=1,
                 use_envelope=False)
    sim.build_helical_mz(theta_deg=theta, A=1.0)
    Iq, qx, qy = sim.compute_qspace()
    qx2, qy2 = np.meshgrid(qx, qy, indexing="xy")
    Iq_masked = Iq.copy()
    Iq_masked[(qx2 ** 2 + qy2 ** 2) < 1e-3] = 0.0
    iy, ix = np.unravel_index(np.argmax(Iq_masked), Iq.shape)
    qx_peak, qy_peak = qx[ix], qy[iy]
    theta_peak = np.degrees(np.arctan2(qy_peak, qx_peak)) % 180.0
    assert min(abs(theta_peak - theta),
               abs(theta_peak - theta - 180.0)) < 10.0


# ----------------------------------------------------------------------
# Coexistence
# ----------------------------------------------------------------------
def test_coexistence_runs_and_caches_angles():
    sim = REXS2D(N=128, L_um=2.0, pitch_nm=60.0, K=4, seed=2,
                 use_envelope=False)
    mz = sim.build_skyrmion_plus_helical_coexistence(
        f_sk=0.5,
        skyrmion_angles_deg=[0.0, 20.0],
        helix_angles_deg=[-30.0, 60.0],
    )
    assert mz.shape == (128, 128)
    assert isinstance(sim._angles_used_deg, dict)
    assert sim._angles_used_deg["f_sk"] == 0.5


# ----------------------------------------------------------------------
# compute_qspace raises before building m_z
# ----------------------------------------------------------------------
def test_compute_qspace_requires_mz():
    sim = REXS2D(N=64, L_um=1.0, pitch_nm=60.0)
    with pytest.raises(RuntimeError):
        sim.compute_qspace()


def test_plot_requires_mz():
    sim = REXS2D(N=64, L_um=1.0, pitch_nm=60.0)
    with pytest.raises(RuntimeError):
        sim.plot()


def test_savefig_requires_plot():
    sim = REXS2D(N=64, L_um=1.0, pitch_nm=60.0)
    sim.build_helical_mz()
    with pytest.raises(RuntimeError):
        sim.savefig("never_written.png")


# ----------------------------------------------------------------------
# savefig writes the file after plot
# ----------------------------------------------------------------------
def test_plot_and_savefig(tmp_path):
    sim = REXS2D(N=128, L_um=1.0, pitch_nm=60.0)
    sim.build_helical_mz(theta_deg=0.0)
    sim.plot(qmax_nm1=0.2, fig_size=(5, 2.5))
    out = tmp_path / "rexs.png"
    sim.savefig(str(out))
    assert out.is_file() and out.stat().st_size > 0


# ----------------------------------------------------------------------
# summary doesn't crash
# ----------------------------------------------------------------------
def test_summary_runs(capsys):
    sim = REXS2D(N=64, L_um=1.0, pitch_nm=60.0, K=1, seed=0)
    sim.build_skyrmion_mz()
    sim.summary()
    out = capsys.readouterr().out
    assert "pitch" in out
