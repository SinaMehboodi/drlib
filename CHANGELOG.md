# Changelog

All notable changes to `drlib` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [SemVer](https://semver.org/).

## [1.1.0] — 2026-05-17

### Added
- `drlib.io.load_mat_dataset(directory)` — lightweight loader for the
  flat four-`.mat`-file layout used by the bundled `PRJCT/Data/`
  reference dataset.
- `Spectrum.from_arrays(freq, field, mlin, mlin_ref, ...)` — build a
  `Spectrum` directly from in-memory NumPy arrays.
- `Spectrum.from_mat(directory, ...)` — convenience wrapper around
  `load_mat_dataset` + `from_arrays`.
- `compare_techniques(...)` — one-call comparison panel of the three
  FMR background-correction techniques (ΔS21 / derivative / dd).
- `Spectrum.__repr__` — informative ASCII-safe summary.
- `safe_style(name)` (in `drlib.utils`) — `plt.style.use` with a
  graceful fallback to `"default"` when the requested style sheet is
  missing.
- `Lorentzian` — real implementation of the complex Lorentzian line
  shape (was a stub that printed `"implement me"`).
- `tests/` — full pytest suite (64 tests) covering all public
  entry points; skips real-data tests automatically when
  `PRJCT/Data` is not on disk.
- `docs_assets/` — eight rendered figures referenced from
  `README.md` so GitHub previews work out of the box.

### Changed
- `Spectrum.get` no longer mutates `self.skip_freq` / `self.skip_field`,
  so calling it twice no longer compounds the down-sampling.
- `Spectrum.__add__`, `__sub__`, `__mul__`, `__truediv__` now produce a
  Spectrum via `__new__` instead of re-loading from disk — they work
  for spectra built via `from_arrays`/`from_mat`.
- `_process_signal` accepts a 1-D `freq` vector and broadcasts it to
  2-D, so the array-based constructors are easier to use.
- `Linecut.plot` honours tuple `pic_size` (e.g. `pic_size=(4, 3)`)
  instead of crashing in `set_size`.
- All `style="Beam"` callsites now go through `safe_style`, so plotting
  works on machines without the bespoke style sheet installed.
- `tutorial.ipynb` rewritten to run end-to-end against the bundled
  `PRJCT/Data/`; outputs (including figures) are embedded.
- `README.md` and `DATA_SHAPES.md` rewritten with the new API,
  inline figures, and a side-by-side comparison of the three
  constructors.

### Fixed
- Cyclic `import` risk in `linecut.py`/`spectrum.py` resolved by moving
  the shared style helper into `drlib.utils`.

## [1.0.0] — 2025

Initial modular release: `Spectrum`, `Linecut`, `CPW`, `REXS2D`,
`derivative`, `derivative_divide`, `FFT_1D`, `dS21`, `B_BS_analytic`.
