"""
drlib.rexs
==========

The :class:`REXS2D` class generates a real-space out-of-plane magnetization
map :math:`m_z(x, y)` for a chosen magnetic texture (skyrmion lattice, helical
state, or a real-space coexistence of the two) and computes its
reciprocal-space intensity :math:`I(q_x, q_y)` via FFT — mimicking resonant
elastic x-ray scattering (REXS).

Units
-----
Real space:
    ``x``, ``y`` in micrometres (µm).
Reciprocal space:
    ``qx``, ``qy`` in nm^{-1} (scattering convention, ``|q| = 2π / period``).

Typical workflow::

    >>> sim = REXS2D(L_um=1.0, pitch_nm=60.0, a_nm=60.0,
    ...              K=2, angles_deg=[0, 15])
    >>> sim.build_skyrmion_mz()
    >>> sim.compute_qspace()
    >>> sim.plot(qmax_nm1=0.2)
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


class REXS2D:
    """2-D REXS-style simulator for magnetic textures.

    The class can build:

    1. **Skyrmion lattice (SkL)** — localised skyrmion "cores" placed on a
       hexagonal lattice (single- or multi-domain).
    2. **Helical (single-q) modulation** — single-domain or multi-domain,
       optionally with higher harmonics.
    3. **Spatial coexistence** of (1) and (2) using a Voronoi-like K-domain
       partition.

    Parameters
    ----------
    N : int, optional
        Pixels per axis (grid is ``N × N``). Default ``512``.
    L_um : float, optional
        Real-space field-of-view in µm. Default ``1.0``.
    pitch_nm : float, optional
        Magnetic-modulation period (nm). Sets the expected ``|q|`` of the
        first-order peak and the helix wavenumber.
    a_nm : float, optional
        Skyrmion-lattice constant (nm). Defaults to ``pitch_nm``.
    R0_frac, w_frac : float, optional
        Skyrmion core radius and domain-wall width as fractions of ``a``.
        Typical: ``R0_frac ∈ [0.15, 0.30]``, ``w_frac ∈ [0.05, 0.15]``.
    K : int, optional
        Number of (SkL or helix) domains. Default ``1``.
    angles_deg : list of float, optional
        Domain orientations (degrees). Must have length ``K`` if provided;
        otherwise random angles are drawn.
    domain_smooth_px : int, optional
        Gaussian smoothing (pixels) applied to the Voronoi masks.
    use_envelope : bool, optional
        If ``True``, apply a Gaussian real-space envelope before FFT
        (mimics finite-coherence broadening).
    Lc_um : float, optional
        Envelope coherence length (µm). Default ``2.0``.
    use_jitter : bool, optional
        If ``True``, jitter skyrmion-centre positions.
    jitter_frac : float, optional
        Jitter standard deviation as a fraction of ``a``. Default ``0.01``.
    seed : int, optional
        RNG seed for reproducibility.
    R0_nm, w_nm : float, optional
        Absolute skyrmion-core radius and wall width (nm). When given,
        they override ``R0_frac`` and ``w_frac`` (useful to decouple size
        from density).

    Notes
    -----
    All caches (``self._mz``, ``self._Iq``, ``self._qx_nm1``, ``self._qy_nm1``)
    are populated lazily by ``build_*`` and ``compute_qspace`` methods.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        N: int = 512,
        L_um: float = 1.0,
        pitch_nm: float = 60.0,
        a_nm: Optional[float] = None,
        R0_frac: float = 0.20,
        w_frac: float = 0.10,
        K: int = 1,
        angles_deg: Optional[List[float]] = None,
        domain_smooth_px: int = 12,
        use_envelope: bool = True,
        Lc_um: float = 2.0,
        use_jitter: bool = True,
        jitter_frac: float = 0.01,
        seed: int = 0,
        R0_nm: Optional[float] = None,
        w_nm: Optional[float] = None,
    ) -> None:
        # --- basic parameters ---
        self.N = int(N)
        self.L_um = float(L_um)
        self.pitch_nm = float(pitch_nm)
        self.a_nm = float(a_nm) if a_nm is not None else float(pitch_nm)

        # --- internal unit conversion (µm) ---
        self.L = self.L_um
        self.a = self.a_nm * 1e-3  # nm -> µm

        # --- skyrmion-shape parameters ---
        if R0_nm is None:
            self.R0 = float(R0_frac) * self.a
        else:
            self.R0 = float(R0_nm) * 1e-3
        if w_nm is None:
            self.w = float(w_frac) * self.a
        else:
            self.w = float(w_nm) * 1e-3

        # --- domain parameters ---
        self.K = int(K)
        self.domain_smooth_px = int(domain_smooth_px)
        self.angles_deg = angles_deg if angles_deg is not None else [0.0] * self.K

        # --- broadening / disorder ---
        self.use_envelope = bool(use_envelope)
        self.Lc = float(Lc_um)
        self.use_jitter = bool(use_jitter)
        self.jitter_frac = float(jitter_frac)

        # --- RNG ---
        self.rng = np.random.default_rng(seed)

        # --- caches ---
        self._grid_built = False
        self._mz: Optional[np.ndarray] = None
        self._mz0: Optional[np.ndarray] = None
        self._Iq: Optional[np.ndarray] = None
        self._qx_nm1: Optional[np.ndarray] = None
        self._qy_nm1: Optional[np.ndarray] = None
        self._angles_used_deg = None
        self._fig = None
        self._axs = None

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    def savefig(self, path: str, **kwargs) -> None:
        """Save the last figure produced by :meth:`plot`.

        Parameters
        ----------
        path : str
            Output file path.
        **kwargs
            Passed to ``matplotlib.figure.Figure.savefig`` (``dpi``,
            ``bbox_inches`` etc.).

        Raises
        ------
        RuntimeError
            If :meth:`plot` has not been called yet.
        """
        if self._fig is None:
            raise RuntimeError("No figure available. Call plot() before savefig().")
        self._fig.savefig(path, **kwargs)
        print(f"Figure saved to: {path}")

    # ------------------------------------------------------------------
    # Grid / utilities
    # ------------------------------------------------------------------
    def build_grid(self):
        """Build and cache the real-space grid ``(x, y, X, Y)`` in µm.

        Returns
        -------
        X, Y : ndarray
            2-D meshgrid arrays in µm.

        Notes
        -----
        ``dx_um = L_um / N`` is the real-space pixel size (µm/pixel).
        """
        N, L = self.N, self.L
        self.x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        self.y = np.linspace(-L / 2, L / 2, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx_um = L / N
        self._grid_built = True
        return self.X, self.Y

    @staticmethod
    def _rotate_xy(X: np.ndarray, Y: np.ndarray, ang_rad: float):
        """Rotate coordinates by ``ang_rad`` (radians)."""
        c, s = np.cos(ang_rad), np.sin(ang_rad)
        return c * X - s * Y, s * X + c * Y

    @staticmethod
    def _hann2d(N: int) -> np.ndarray:
        """Return a 2-D Hann window to suppress edge discontinuities in FFT."""
        wx = np.hanning(N)
        wy = np.hanning(N)
        return np.outer(wy, wx)

    @staticmethod
    def _single_skyrmion_mz(R: np.ndarray, R0: float, w: float) -> np.ndarray:
        r"""Radial skyrmion profile: :math:`m_z(R) = \tanh((R - R_0) / w)`.

        Near ``R = 0``, ``mz ≈ -1`` (core); far away, ``mz → +1`` (background).
        """
        return np.tanh((R - R0) / w)

    # ------------------------------------------------------------------
    # Skyrmion lattice construction
    # ------------------------------------------------------------------
    def _make_hex_centers(self, phase: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
        """Generate hexagonal lattice centres covering the ``L × L`` window.

        Parameters
        ----------
        phase : (float, float), optional
            Lattice-origin shift in µm. Useful to break artificial alignment.

        Returns
        -------
        centers : ndarray, shape (M, 2)
            Centre coordinates ``[cx, cy]`` in µm.
        """
        L, a = self.L, self.a
        a1 = np.array([a, 0.0])
        a2 = np.array([a / 2, a * np.sqrt(3) / 2])

        nmax = int(np.ceil((L / a) * 2))
        centers = []
        for n1 in range(-nmax, nmax + 1):
            for n2 in range(-nmax, nmax + 1):
                r = n1 * a1 + n2 * a2 + np.array(phase)
                if (abs(r[0]) <= L / 2 + a) and (abs(r[1]) <= L / 2 + a):
                    centers.append(r)
        return np.array(centers)

    def _make_domain_masks(self) -> np.ndarray:
        """Create ``K`` Voronoi-like domain masks (smoothed to soft boundaries).

        Returns
        -------
        masks : ndarray, shape (K, N, N)
            Per-domain weight maps that sum to 1 at each pixel.
        """
        if not self._grid_built:
            self.build_grid()
        X, Y = self.X, self.Y
        K = self.K

        seeds = np.column_stack([
            self.rng.uniform(X.min(), X.max(), size=K),
            self.rng.uniform(Y.min(), Y.max(), size=K),
        ])
        d2 = np.zeros((K, X.shape[0], X.shape[1]))
        for k in range(K):
            d2[k] = (X - seeds[k, 0]) ** 2 + (Y - seeds[k, 1]) ** 2

        labels = np.argmin(d2, axis=0)
        masks = np.zeros((K, X.shape[0], X.shape[1]), dtype=float)
        for k in range(K):
            masks[k] = (labels == k).astype(float)

        if self.domain_smooth_px > 0:
            try:
                from scipy.ndimage import gaussian_filter
                for k in range(K):
                    masks[k] = gaussian_filter(masks[k], sigma=self.domain_smooth_px)
                s = masks.sum(axis=0, keepdims=True)
                masks = masks / (s + 1e-12)
            except Exception as e:
                print("scipy not available for mask smoothing; using sharp domains. Error:", e)
        return masks

    def _make_single_domain_skyrmion_mz(self, ang_rad: float) -> np.ndarray:
        """Build a single-domain hexagonal skyrmion lattice ``m_z(x, y)``.

        Parameters
        ----------
        ang_rad : float
            In-plane lattice rotation (radians).

        Returns
        -------
        mz : ndarray, shape (N, N)
            Values in ``[-1, +1]``.
        """
        if not self._grid_built:
            self.build_grid()
        Xr, Yr = self._rotate_xy(self.X, self.Y, ang_rad)

        a = self.a
        phase = (self.rng.uniform(-a, a), self.rng.uniform(-a, a))
        centers = self._make_hex_centers(phase=phase)

        if self.use_jitter and len(centers) > 0:
            centers = centers + self.rng.normal(scale=self.jitter_frac * a, size=centers.shape)

        mz = np.ones_like(Xr)
        for cx, cy in centers:
            R = np.sqrt((Xr - cx) ** 2 + (Yr - cy) ** 2)
            mz = np.minimum(mz, self._single_skyrmion_mz(R, self.R0, self.w))

        return np.clip(mz, -1.0, 1.0)

    def build_skyrmion_mz(self) -> np.ndarray:
        """Build and cache the multi-domain skyrmion-lattice ``m_z(x, y)``.

        Returns
        -------
        mz : ndarray, shape (N, N)

        Notes
        -----
        * If ``len(angles_deg) != K``, random angles are drawn from
          ``Uniform(-30°, +30°)``.
        * ``K = 1`` yields a single domain with 6-fold FFT peaks.
        * ``K > 1`` yields several rotated 6-fold peak sets / broadening.
        """
        if not self._grid_built:
            self.build_grid()
        if len(self.angles_deg) != self.K:
            angles_deg = list(self.rng.uniform(-30, 30, size=self.K))
        else:
            angles_deg = list(self.angles_deg)

        angles = np.deg2rad(np.array(angles_deg))
        masks = self._make_domain_masks()

        mz_domains = np.stack(
            [self._make_single_domain_skyrmion_mz(angles[k]) for k in range(self.K)],
            axis=0,
        )
        mz = np.clip(np.sum(masks * mz_domains, axis=0), -1.0, 1.0)

        self._mz = mz
        self._angles_used_deg = angles_deg
        return mz

    # ------------------------------------------------------------------
    # Helical state construction
    # ------------------------------------------------------------------
    def build_helical_mz(
        self,
        theta_deg: float = 0.0,
        phase: float = 0.0,
        m0: float = 0.0,
        A: float = 1.0,
        alpha2: float = 0.0,
        alpha3: float = 0.0,
    ) -> np.ndarray:
        r"""Single-domain helical (single-``q``) ``m_z(x, y)``.

        .. math::

            m_z = m_0 + A\cos(\mathrm{arg}) + \alpha_2 \cos(2\,\mathrm{arg})
                + \alpha_3 \cos(3\,\mathrm{arg})

        with ``arg = q (cos θ · x + sin θ · y) + phase`` and
        ``q = 2π / pitch_nm`` (converted to µm⁻¹).

        Parameters
        ----------
        theta_deg : float
            Propagation direction (degrees).
        phase : float
            Initial phase (radians).
        m0 : float
            Constant offset.
        A : float
            Fundamental amplitude.
        alpha2, alpha3 : float
            Second- and third-harmonic amplitudes (give ±2q, ±3q peaks).

        Returns
        -------
        mz : ndarray, shape (N, N)
        """
        if not self._grid_built:
            self.build_grid()
        pitch_um = self.pitch_nm * 1e-3
        q_um1 = 2 * np.pi / pitch_um
        th = np.deg2rad(theta_deg)

        arg = q_um1 * (np.cos(th) * self.X + np.sin(th) * self.Y) + phase
        mz = m0 + A * np.cos(arg) + alpha2 * np.cos(2 * arg) + alpha3 * np.cos(3 * arg)
        mz = np.clip(mz, -1.0, 1.0)

        self._mz = mz
        self._angles_used_deg = [float(theta_deg)]
        return mz

    def build_helical_multidomain_mz(
        self,
        angles_deg: Optional[List[float]] = None,
        phase: float = 0.0,
        m0: float = 0.0,
        A: float = 1.0,
        phase_noise_std: float = 0.0,
        alpha2: float = 0.0,
        alpha3: float = 0.0,
    ) -> np.ndarray:
        """Multi-domain helical state using the same Voronoi-mask approach.

        Each domain ``k`` has its own propagation direction
        ``angles_deg[k]``, producing rotated ±q (and optional ±2q, ±3q)
        spots in q-space.

        Parameters
        ----------
        angles_deg : list of float, optional
            Propagation directions (deg). Must have length ``K``.
        phase, m0, A
            See :meth:`build_helical_mz`.
        phase_noise_std : float, optional
            Per-pixel random phase noise (rad) — broadens peaks.
        alpha2, alpha3 : float, optional
            Higher-harmonic amplitudes.

        Returns
        -------
        mz : ndarray, shape (N, N)
        """
        if not self._grid_built:
            self.build_grid()

        if angles_deg is None:
            if hasattr(self, "angles_deg") and len(self.angles_deg) == self.K:
                angles_deg = list(self.angles_deg)
            else:
                angles_deg = list(self.rng.uniform(-90, 90, size=self.K))
        elif len(angles_deg) != self.K:
            raise ValueError("angles_deg must have length K (number of domains).")

        masks = self._make_domain_masks()
        pitch_um = self.pitch_nm * 1e-3
        q_um1 = 2 * np.pi / pitch_um

        mz_domains = []
        for th_deg in angles_deg:
            th = np.deg2rad(th_deg)
            arg = q_um1 * (np.cos(th) * self.X + np.sin(th) * self.Y) + phase
            if phase_noise_std > 0:
                arg = arg + self.rng.normal(scale=phase_noise_std, size=arg.shape)
            mz_k = m0 + A * np.cos(arg) + alpha2 * np.cos(2 * arg) + alpha3 * np.cos(3 * arg)
            mz_domains.append(np.clip(mz_k, -1.0, 1.0))

        mz_domains = np.stack(mz_domains, axis=0)
        mz = np.clip(np.sum(masks * mz_domains, axis=0), -1.0, 1.0)
        self._mz = mz
        self._angles_used_deg = angles_deg
        return mz

    # ------------------------------------------------------------------
    # Coexistence state
    # ------------------------------------------------------------------
    def build_skyrmion_plus_helical_coexistence(
        self,
        f_sk: float = 0.5,
        skyrmion_angles_deg: Optional[List[float]] = None,
        helix_angles_deg: Optional[List[float]] = None,
        helix_phase: float = 0.0,
        helix_m0: float = 0.0,
        helix_A: float = 1.0,
        helix_phase_noise_std: float = 0.0,
        alpha2: float = 0.0,
        alpha3: float = 0.0,
    ) -> np.ndarray:
        """Real-space patches of skyrmion-lattice and helical phases.

        Uses the same ``K``-domain Voronoi masks: the first ``N_sk =
        round(f_sk * K)`` masks are filled with skyrmion lattices, the
        remaining ``K - N_sk`` with helices.

        Parameters
        ----------
        f_sk : float in [0, 1]
            Fraction of domains assigned to SkL.
        skyrmion_angles_deg : list of float, optional
            SkL orientations (deg), length must equal ``N_sk``.
        helix_angles_deg : list of float, optional
            Helix propagation directions (deg), length must equal
            ``K - N_sk``.
        helix_phase, helix_m0, helix_A, helix_phase_noise_std
            Helix parameters (see :meth:`build_helical_multidomain_mz`).
        alpha2, alpha3 : float
            Higher-harmonic amplitudes for the helix domains.

        Returns
        -------
        mz : ndarray, shape (N, N)
            Mixed-phase magnetisation map in ``[-1, +1]``.
        """
        if not self._grid_built:
            self.build_grid()

        masks = self._make_domain_masks()
        N_sk = int(np.clip(round(np.clip(f_sk, 0.0, 1.0) * self.K), 0, self.K))
        N_h = self.K - N_sk

        # SkL domains
        sk_domains = None
        sk_angles_used: List[float] = []
        if N_sk > 0:
            if skyrmion_angles_deg is None:
                skyrmion_angles_deg = list(self.rng.uniform(-30, 30, size=N_sk))
            if len(skyrmion_angles_deg) != N_sk:
                raise ValueError(f"skyrmion_angles_deg must have length N_sk={N_sk}.")
            sk_angles_used = list(skyrmion_angles_deg)
            sk_domains = np.stack(
                [self._make_single_domain_skyrmion_mz(np.deg2rad(th)) for th in skyrmion_angles_deg],
                axis=0,
            )

        # Helical domains
        hel_domains = None
        hel_angles_used: List[float] = []
        if N_h > 0:
            if helix_angles_deg is None:
                helix_angles_deg = list(self.rng.uniform(-90, 90, size=N_h))
            if len(helix_angles_deg) != N_h:
                raise ValueError(f"helix_angles_deg must have length N_h={N_h}.")
            hel_angles_used = list(helix_angles_deg)

            pitch_um = self.pitch_nm * 1e-3
            q_um1 = 2.0 * np.pi / pitch_um
            hel_list = []
            for th_deg in helix_angles_deg:
                th = np.deg2rad(th_deg)
                arg = q_um1 * (np.cos(th) * self.X + np.sin(th) * self.Y) + helix_phase
                if helix_phase_noise_std > 0:
                    arg = arg + self.rng.normal(scale=helix_phase_noise_std, size=arg.shape)
                mz_k = (
                    helix_m0
                    + helix_A * np.cos(arg)
                    + alpha2 * np.cos(2.0 * arg)
                    + alpha3 * np.cos(3.0 * arg)
                )
                hel_list.append(np.clip(mz_k, -1.0, 1.0))
            hel_domains = np.stack(hel_list, axis=0)

        # Combine
        mz = np.zeros((self.N, self.N), dtype=float)
        if N_sk > 0:
            mz += np.sum(masks[:N_sk] * sk_domains, axis=0)
        if N_h > 0:
            mz += np.sum(masks[N_sk:] * hel_domains, axis=0)
        mz = np.clip(mz, -1.0, 1.0)

        self._mz = mz
        self._angles_used_deg = {
            "f_sk": float(f_sk),
            "skyrmion_deg": sk_angles_used,
            "helical_deg": hel_angles_used,
        }
        return mz

    # ------------------------------------------------------------------
    # FFT / q-space
    # ------------------------------------------------------------------
    def compute_qspace(self):
        """Compute and cache ``I(qx, qy)`` from the currently stored ``m_z``.

        Processing steps:

        1. Subtract the mean (remove DC).
        2. (Optional) Apply Gaussian envelope ``exp(-(x²+y²)/(2 Lc²))``.
        3. Apply a 2-D Hann window to suppress edge discontinuities.
        4. FFT and form ``Iq = |FFT|²``, normalised so ``max(Iq) = 1``.

        Returns
        -------
        Iq : ndarray, shape (N, N)
            Normalised FFT intensity.
        qx_nm1, qy_nm1 : ndarray, shape (N,)
            Reciprocal-space axes in nm⁻¹.

        Raises
        ------
        RuntimeError
            If no ``m_z`` map has been built yet.
        """
        if self._mz is None:
            raise RuntimeError(
                "No m_z map built yet. "
                "Call build_skyrmion_mz() or build_helical_mz() first."
            )
        if not self._grid_built:
            self.build_grid()

        mz0 = self._mz - self._mz.mean()
        if self.use_envelope:
            env = np.exp(-(self.X ** 2 + self.Y ** 2) / (2 * self.Lc ** 2))
            mz0 *= env
        mz0 *= self._hann2d(self.N)

        self._mz0 = mz0
        F = np.fft.fftshift(np.fft.fft2(mz0))
        Iq = np.abs(F) ** 2
        Iq /= (Iq.max() + 1e-12)
        self._Iq = Iq

        # q axes: cycles/µm -> rad/nm
        qx_cyc_per_um = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dx_um))
        qy_cyc_per_um = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dx_um))
        self._qx_nm1 = 2 * np.pi * (qx_cyc_per_um / 1000.0)
        self._qy_nm1 = 2 * np.pi * (qy_cyc_per_um / 1000.0)
        return self._Iq, self._qx_nm1, self._qy_nm1

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(
        self,
        qmax_nm1: float = 0.2,
        L_max: float = 0.5,
        mask_dc: bool = True,
        dc_mask_px: int = 3,
        title_left: str = "m_z(x,y)",
        title_right: str = "FFT intensity (log10)",
        c_mapR: str = "Blues",
        c_mapQ: str = "Blues",
        cbar_ticks_R: Optional[List[float]] = None,
        cbar_ticks_Q: Optional[List[float]] = None,
        fig_size=(6, 3),
    ) -> None:
        """Show the real-space ``m_z`` and reciprocal-space ``I(q)`` side by side.

        Parameters
        ----------
        qmax_nm1 : float, optional
            Half-width of the displayed q-range (nm⁻¹). Pick ≈ 2 × the
            expected peak radius.
        L_max : float, optional
            Half-width of the real-space view in µm.
        mask_dc : bool, optional
            If ``True``, blank a small square around ``q = 0`` for clarity.
        dc_mask_px : int, optional
            Half-size of the DC-mask square (pixels).
        title_left, title_right : str, optional
            Plot titles.
        c_mapR, c_mapQ : str, optional
            Matplotlib colormaps for real-space and q-space images.
        cbar_ticks_R, cbar_ticks_Q : list of float, optional
            Override the colorbar ticks if non-``None``.
        fig_size : (float, float), optional
            Figure size in inches.
        """
        if self._mz is None:
            raise RuntimeError(
                "No m_z map built yet. "
                "Call build_skyrmion_mz() or build_helical_mz() first."
            )
        if self._Iq is None:
            self.compute_qspace()

        Iq_plot = self._Iq.copy()
        if mask_dc:
            c = self.N // 2
            Iq_plot[c - dc_mask_px:c + dc_mask_px + 1,
                    c - dc_mask_px:c + dc_mask_px + 1] = 0.0

        fig, axs = plt.subplots(1, 2, figsize=fig_size)
        self._fig = fig
        self._axs = axs

        im0 = axs[0].imshow(
            self._mz,
            extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
            origin="lower", aspect="equal",
            vmin=-1, vmax=1, cmap=c_mapR,
        )
        axs[0].set_title(title_left)
        axs[0].set_xlabel("x (µm)")
        axs[0].set_ylabel("y (µm)")
        axs[0].set_xlim(-L_max, L_max)
        axs[0].set_ylim(-L_max, L_max)
        axs[0].set_yticks([-L_max, 0, L_max])
        axs[0].set_xticks([-L_max, 0, L_max])
        cbarR = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        if cbar_ticks_R is not None:
            cbarR.set_ticks(cbar_ticks_R)

        im1 = axs[1].imshow(
            np.log10(Iq_plot + 1e-12),
            extent=[self._qx_nm1.min(), self._qx_nm1.max(),
                    self._qy_nm1.min(), self._qy_nm1.max()],
            origin="lower", aspect="equal",
            cmap=c_mapQ,
        )
        axs[1].set_title(title_right)
        axs[1].set_xlabel(r"$q_x$ (nm$^{-1}$)")
        axs[1].set_ylabel(r"$q_y$ (nm$^{-1}$)")
        axs[1].set_xlim(-qmax_nm1, qmax_nm1)
        axs[1].set_ylim(-qmax_nm1, qmax_nm1)
        axs[1].set_xticks([-qmax_nm1, 0, qmax_nm1])
        axs[1].set_yticks([-qmax_nm1, 0, qmax_nm1])
        cbarRQ = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        if cbar_ticks_Q is not None:
            cbarRQ.set_ticks(cbar_ticks_Q)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Convenience info
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print a short summary of the simulation parameters."""
        if not self._grid_built:
            self.build_grid()
        q_expected_nm1 = 2 * np.pi / self.pitch_nm
        print(f"N={self.N}, L={self.L_um:.3f} µm, dx={self.dx_um * 1000:.2f} nm")
        print(f"pitch={self.pitch_nm:.1f} nm  -> expected |q|≈2π/pitch={q_expected_nm1:.4f} nm^-1")
        print(f"a={self.a_nm:.1f} nm, R0={self.R0 * 1000:.2f} nm, w={self.w * 1000:.2f} nm")
        if self._angles_used_deg is not None:
            print("Angles used (deg):", self._angles_used_deg)
