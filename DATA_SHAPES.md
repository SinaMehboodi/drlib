# Required data shapes & on-disk layout

This document lists, **for every public class and loader in `drlib`**, the
exact data shape (and where applicable, the on-disk folder structure) the
class needs to work.

If your data does not match any of the layouts below, the easiest fix is
to load it yourself and feed the resulting NumPy arrays into
`Spectrum.from_arrays` — you do **not** need to modify `drlib` itself.

---

## 1. `Spectrum(path=...)` — LabView VI folder layout

`Spectrum` reads LabView-generated `.mat` files. The constructor's `path`
argument must point at the **measurement-root folder**, which has to look
like this:

```
<path>/                                       ← you pass THIS to Spectrum
│
├── MagneticField.csv                         ← auto-generated on first load
│
└── measurement/
    │
    ├── MEAS/                                 ← used once to scrape field values
    │   ├── <scan_0>/
    │   │   └── <prefix>_<field>mT__freq.txt  ← field is parsed from the filename
    │   ├── <scan_1>/
    │   └── ...
    │
    └── store/
        │
        ├── MEAS/<S_param>/                   ← <S_param> = S11, S21, S12 or S22
        │   ├── freq.mat                      ← 2-D array under key "Data"
        │   └── MLIN.mat                      ← 2-D array under key "Data"
        │
        └── REF/<S_param>/                    ← reference baseline
            └── MLIN.mat
```

### Required ndarray shapes (after `loadmat(...)["Data"]`)

| File                         | NumPy shape          | Units / meaning                       |
|------------------------------|----------------------|---------------------------------------|
| `freq.mat["Data"]`           | `(N_freq, N_field)`  | Frequency, Hz                         |
| `MLIN.mat["Data"]` (MEAS)    | `(N_freq, N_field)`  | Linear magnitude of S-parameter       |
| `MLIN.mat["Data"]` (REF)     | `(N_freq, N_field)`  | Linear magnitude of reference         |
| `MagneticField.csv`          | `N_field` rows       | Magnetic field, mT (no header)        |

The class transposes `MLIN` internally, so its public attribute
`Spectrum.S21dd` has shape **`(N_field, N_freq)`**.

### Quickstart

```python
from drlib import Spectrum
spec = Spectrum(
    path=r"C:/data/2024-04-02_FMR_Run3",  # the folder shown above
    saturation=42,                        # set after a first look at spec.plot()
    skip_field=1, skip_freq=1,            # down-sampling, optional
    derivative_divide=True,
    S_param="S21",                        # which Sxy to load
)
```

---

## 1.bis  `Spectrum.from_mat(directory)` / `load_mat_dataset(directory)` — flat `.mat` folder

The lightweight alternative to the deep LabView layout. Used by the
bundled reference dataset at `PRJCT/Data/`. The directory contains
**four files only**:

```
<directory>/
├── freq.mat          ← (N_freq, N_field)  – frequency grid (Hz)
├── MLIN.mat          ← (N_freq, N_field)  – sample |S-parameter|
├── MLIN_REF.mat      ← (N_freq, N_field)  – reference baseline
└── sample_field.mat  ← (N_field, …)       – magnetic-field grid (mT)
```

### Required array shapes

| File / loader return value | NumPy shape         | Notes |
|----------------------------|---------------------|-------|
| `freq.mat` / `freq`        | `(N_freq, N_field)` | Frequency, Hz                                                                                          |
| `MLIN.mat` / `mlin`        | `(N_freq, N_field)` | Sample magnitude                                                                                       |
| `MLIN_REF.mat` / `mlin_ref`| `(N_freq, N_field)` or `(N_freq,)` | Reference baseline. 1-D arrays are tiled along the field axis on load.                |
| `sample_field.mat` / `field`| `(N_field,)`       | Magnetic-field grid (mT). If the file stores a tiled 2-D version, `load_mat_dataset` collapses it.    |

Either the `Data` key or the explicit variable name (`sample_field` etc.)
in the `.mat` is accepted.

### Quickstart

```python
from drlib import Spectrum, load_mat_dataset

freq, field, mlin, mlin_ref = load_mat_dataset(r"PRJCT/Data")
spec = Spectrum.from_mat(r"PRJCT/Data", saturation=200, derivative_divide=True)
# – or, equivalently –
spec = Spectrum.from_arrays(freq, field, mlin, mlin_ref,
                            saturation=200, derivative_divide=True)
```

### Choosing the background-correction mode

`Spectrum` (regardless of constructor) supports three processing modes,
selected via the `derivative_divide` / `derivative` flags:

| Flags                                       | Result          | Notes |
|---------------------------------------------|-----------------|-------|
| `derivative_divide=True`  (default)         | dd: `(∂S/∂H)/S` | Strongest background suppression, see Maier-Flaig 2017. |
| `derivative_divide=False, derivative=True`  | `∂S/∂H`         | Plain numerical derivative.                              |
| `derivative_divide=False, derivative=False` | `ΔS = S − S_ref`| Requires `mlin_ref`.                                     |

For a quick visual comparison of all three on the same data, use
`drlib.compare_techniques(freq, field, mlin, …)`.

---

## 2. `Linecut(cut=..., Spectrum=..., frequencies=...)`

`Linecut` does not read from disk — it slices an existing `Spectrum`.

| Argument        | Type                       | Shape / range                           |
|-----------------|----------------------------|-----------------------------------------|
| `cut`           | `int`                      | `0 ≤ cut < N_field`                     |
| `Spectrum`      | `drlib.Spectrum`           | Must already be loaded                  |
| `frequencies`   | `(float, float)`           | `(f_min, f_max)` in **Hz**              |

Resulting public attributes:

| Attribute       | Shape          | Notes                                |
|-----------------|----------------|--------------------------------------|
| `Freq_cut`      | `(M,)`         | Frequencies inside `frequencies`     |
| `S21dd_cut`     | `(M,)`         | Signal values along the cut          |
| `H0`            | scalar (mT)    | Bias field at the chosen cut         |

### Fit-parameter ordering

Every model (`dS21` for derivative-divide data, `Lorentzian` for
ΔS21 / derivative data) takes parameters in the same order, so the
same initial-guess vector can be reused when switching modes:

| Method        | Length | Order                                                                 |
|---------------|--------|-----------------------------------------------------------------------|
| `one_peak`    | 7      | `A, Psi, fres, Df, mod, Msat, H0`                                     |
| `two_peak`    | 14     | one-peak block repeated for `m1_`, `m2_`                              |
| `three_peak`  | 21     | one-peak block repeated for `m1_`, `m2_`, `m3_`                       |

---

## 3. `CPW(current, signal_line, gap, ground, thickness)`

Purely analytic — no on-disk data needed. All arguments are scalars in
**SI units**:

| Argument        | Units | Typical value |
|-----------------|-------|---------------|
| `current`       | A     | `10e-3`       |
| `signal_line`   | m     | `2e-6`        |
| `gap`           | m     | `1e-6`        |
| `ground`        | m     | `10e-6`       |
| `thickness`     | m     | `200e-9`      |

Methods return matplotlib plots only; their numerical outputs are
internally NumPy arrays of length `number_of_points`.

---

## 4. `REXS2D(N, L_um, pitch_nm, ...)`

Purely analytic — generates its own `m_z(x, y)`. No on-disk data needed.

| Argument        | Units / type            | Notes                                            |
|-----------------|-------------------------|--------------------------------------------------|
| `N`             | `int`                   | Pixels per side, grid is `N × N`                 |
| `L_um`          | µm                      | Real-space field of view                         |
| `pitch_nm`      | nm                      | Modulation period (helix) / SkL period reference |
| `a_nm`          | nm or `None`            | SkL lattice constant; default = `pitch_nm`       |
| `K`             | `int`                   | Number of domains                                |
| `angles_deg`    | `list[float]` of length `K` | Domain orientations (deg)                    |

Output caches (after `build_*` + `compute_qspace`):

| Attribute     | Shape         | Units        |
|---------------|---------------|--------------|
| `self._mz`    | `(N, N)`      | Dimensionless, in `[-1, +1]` |
| `self._Iq`    | `(N, N)`      | Normalised intensity         |
| `self._qx_nm1`, `self._qy_nm1` | `(N,)` | nm⁻¹                      |

---

## 5. Plain functions (no path needed)

| Function                    | Required input shape                                |
|-----------------------------|-----------------------------------------------------|
| `derivative(X, Y, Z, …)`    | All three `(N, M)`                                  |
| `derivative_divide(...)`    | Same as `derivative`                                |
| `FFT_1D(array, dx, ...)`    | `array.shape == (N,)`, `dx` is float (µm)           |
| `dS21(x, A, Psi, …, H0)`    | `x.shape == (N,)` (frequencies in **Hz**)           |
| `Lorentzian(x, A, Psi, …, H0)` | Same as `dS21`                                   |
| `B_BS_analytic(x0, z0, …)`  | `x0` scalar or `(N,)`; `z0` scalar                  |
| `compare_techniques(freq, field, mlin, mlin_ref, …)` | Same as `Spectrum.from_arrays` |

---

## Troubleshooting

* **`FileNotFoundError: freq.mat`** — your `path` is one folder too high
  or too low. The classical `Spectrum(path=...)` always appends
  `\measurement\store\MEAS\<S_param>\freq.mat`. If your layout is flat,
  use `Spectrum.from_mat(directory)` instead.

* **Magnetic-field axis looks wrong** — delete the auto-generated
  `MagneticField.csv` and let `Spectrum` rebuild it from the filenames.

* **lmfit complains about NaNs** — your data probably contains zeros
  where `derivative_divide` divides; reduce `modulation_amp`, switch to
  `derivative=True`, or use the plain ΔS21 path.

* **`OSError: 'Beam' style not available`** — the bespoke LaTeX style
  is not on your machine. Pass `style=None` to any plotting method, or
  call `drlib.safe_style("Beam")` which falls back to `"default"` silently.
