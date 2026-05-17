# Publish `drlib` to GitHub — GitHub Desktop + VS Code walk-through

You have **Git**, **GitHub Desktop**, and **VS Code** installed; this guide
uses just those three.  No command-line steps unless you want them.

If something goes sideways, the all-options reference is
[`GITHUB_PUBLISHING.md`](GITHUB_PUBLISHING.md).

---

## 1. Sign GitHub Desktop in to your GitHub account (once)

1. Open **GitHub Desktop**.
2. Menu: **File → Options → Accounts**.
3. Click **Sign in** next to *GitHub.com* and follow the browser flow.
4. Back in Options → **Git**, make sure the **Name** and **Email** are filled
   in (e.g. `Sina Mehboodi`, `sinamehboodi@gmail.com`).

Same for VS Code: install the built-in *GitHub Pull Requests* extension
and sign in (bottom-left **Accounts** icon → *Sign in with GitHub*). VS
Code and GitHub Desktop share the same git config, so you only do this
once per machine.

---

## 2. Add the local folder as a repository in GitHub Desktop

In GitHub Desktop:

1. **File → Add local repository…**
2. Browse to:
   ```
   F:\Sina_Data_Job2026\PRJCT\My LIbrary\DrLib
   ```
3. GitHub Desktop will say *"This directory does not appear to be a Git
   repository.  Would you like to create a repository here instead?"* —
   click **create a repository**.
4. A dialog appears.  Fill it in **exactly** like this:

   | Field | Value |
   |---|---|
   | **Name**           | `drlib` *(use `drlib-fmr` if `drlib` is already taken by another user — Desktop will tell you in step 5)* |
   | **Description**    | `Python toolkit for ferromagnetic-resonance (FMR) spectroscopy, coplanar-waveguide (CPW) field modelling, and 2-D REXS simulation of magnetic textures.` |
   | **Local path**     | (already filled by step 2 — don't change) |
   | **Initialize this repository with a README** | ☐ **leave unchecked** — you already have a README |
   | **Git ignore**     | leave **None** — you already have `.gitignore` |
   | **License**        | leave **None** — you already have `LICENSE` (MIT) |

5. Click **Create repository**.

GitHub Desktop now shows the changes view: a long list of files (every
file under `DrLib\`) staged for the first commit.

---

## 3. Sanity-check what is about to be committed

Scroll through the list on the left.  You should see:

* ✅  `README.md`, `LICENSE`, `CHANGELOG.md`, `DATA_SHAPES.md`,
   `tutorial.ipynb`, `pyproject.toml`, `requirements.txt`, `.gitignore`
* ✅  the whole `drlib/` package
* ✅  the whole `tests/` folder
* ✅  `docs_assets/*.png` (the README figures)
* ✅  `examples/data/*.mat` (~7 MB — the downsampled reference dataset)

You should **not** see:

* ❌  `__pycache__/`, `.pytest_cache/`
* ❌  any `MagneticField.csv`
* ❌  any `*.mat` outside `examples/data/`
* ❌  any `.env`, credential, or notebook checkpoint files

If anything sensitive appears, right-click it → **Ignore file** (it gets
added to `.gitignore`).

---

## 4. Write the first commit

At the bottom-left of GitHub Desktop:

| Field | Value |
|---|---|
| **Summary** *(top, ≤ 72 chars)* | `Initial public release: drlib 1.1.0` |
| **Description** *(optional, multi-line)* | ```\nFMR / CPW / REXS analysis toolkit.\n\n- Spectrum: load LabView .mat folder, flat .mat directory, or in-memory arrays\n- Linecut: 1/2/3-peak dS21 / Lorentzian fits via lmfit\n- CPW: analytical Biot-Savart field + k-spectrum\n- REXS2D: skyrmion / helical / coexistence textures\n- compare_techniques: side-by-side delta_S21 / derivative / dd plot\n- 64-test pytest suite, bundled downsampled reference dataset, runnable tutorial\n``` |

Click the blue **Commit to main** button.

---

## 5. Publish the repository to GitHub

A new banner at the top says **Publish repository**.  Click it.  In the
dialog:

| Field | Value |
|---|---|
| **Name** | `drlib` *(or `drlib-fmr`)* |
| **Description** | (same one-line text as in step 2) |
| **Keep this code private** | ☐ **uncheck** — we want public |
| **Organization** | leave as your personal account (or pick an org) |

Click **Publish repository**.

GitHub Desktop pushes the commit and opens the new repo on github.com in
your browser.  Done!

---

## 6. Polish the repo on github.com (5 minutes)

The web UI exposes a few knobs GitHub Desktop does not.  Open
`https://github.com/<your-username>/drlib` and:

1. **Description / website** — the gear ⚙️ icon at the top right of the
   **About** sidebar.  Paste:

   * Description:
     `Python toolkit for ferromagnetic-resonance (FMR) spectroscopy, coplanar-waveguide (CPW) field modelling, and 2-D REXS simulation of magnetic textures.`
   * Website: leave empty, or your ORCID / lab page.

2. **Topics** (still in the gear ⚙️ dialog).  Paste these one at a time:

   ```
   fmr  ferromagnetic-resonance  spectroscopy  coplanar-waveguide  cpw
   rexs  skyrmion  helical-magnet  lmfit  scientific-computing  physics  python
   ```

3. **Check that the README image links work.**  Scroll the landing page —
   you should see the three-technique comparison panel, the dd spectrum,
   the Linecut fit, the CPW field plot, and the REXS skyrmion lattice.
   If any image is broken, the corresponding file in `docs_assets/` was
   not committed; go back to GitHub Desktop, add it, commit, push.

4. **Open the tutorial on GitHub** (`tutorial.ipynb`).  GitHub renders
   notebooks inline; you should see every cell's output already embedded
   (figures included).

5. **Tag the release.**

   * GitHub Desktop → **Branch → Create tag** → enter `v1.1.0`,
     description `drlib 1.1.0`.
   * Then **Push origin** again to upload the tag.
   * Back on github.com → **Releases → Draft a new release** → pick the
     `v1.1.0` tag → paste the relevant section of `CHANGELOG.md` →
     **Publish release**.

---

## 7. Tell people how to install it

Anyone with Python can now run:

```bash
pip install git+https://github.com/<your-username>/drlib.git
```

Or, for a specific tagged release:

```bash
pip install git+https://github.com/<your-username>/drlib.git@v1.1.0
```

Add that snippet to the top of the README's *Installation* section if
you want it to be the very first thing visitors see.

---

## 8. Day-to-day workflow after the first push

You will keep editing the library; here is the routine:

### In GitHub Desktop

1. Edit files in VS Code.  Save.
2. Switch to GitHub Desktop — your changes appear in the changes view.
3. Type a one-line summary at the bottom-left
   (e.g. `Fix dS21 normalisation at low fields`), optional longer
   description.
4. Click **Commit to main**.
5. Click **Push origin** at the top.

### In VS Code (alternative — uses the *Source Control* sidebar)

1. Open the Source Control sidebar (Ctrl + Shift + G).
2. Stage files with the **+** icon (or all of them with **Stage All Changes**).
3. Type a commit message in the box at the top → **Ctrl+Enter** to commit.
4. Click the **…** menu → **Push** (or use the cloud icon in the status bar).

Both routes do the same thing under the hood, so use whichever you find
more comfortable.

---

## 9. (Optional) Run the test suite locally before every commit

```bash
cd "F:\Sina_Data_Job2026\PRJCT\My LIbrary\DrLib"
python -m pytest
```

Expected output: `64 passed in ~3 s` (≈ 13 s if `DRLIB_DATA_DIR`
points at the full-resolution `PRJCT/Data`).

If you want this to run automatically on GitHub for every push, see the
`.github/workflows/tests.yml` snippet at the bottom of
[`GITHUB_PUBLISHING.md`](GITHUB_PUBLISHING.md).

---

## 10. If something breaks

| Symptom | Fix |
|---|---|
| GitHub Desktop says *"failed to push some refs"* | Pull first (top-right **Fetch origin** → **Pull origin**) — someone (probably you on another machine) pushed first. |
| README images broken on the GitHub page | Open `.gitignore`, check that `!docs_assets/*.png` is present; commit any missing PNGs. |
| `pip install git+…` fails with *"setuptools could not find a version"* | Make sure your tag (`v1.1.0`) matches `version = "1.1.0"` in `pyproject.toml`, and that the tag was actually pushed (**Branch → Push tag** in GitHub Desktop). |
| Need to undo the very last commit (already pushed) | GitHub Desktop → **Branch → Revert commit** on the bad commit, then **Push origin**.  This creates a *new* commit that undoes the old one — safe for public branches. |
| Tutorial works locally but errors for cloners | Make sure `examples/data/*.mat` are actually committed (`git status` shows them as tracked).  If your `.gitignore` excludes `*.mat`, the negation pattern `!examples/data/*.mat` must come *after* the bare `*.mat`. |
