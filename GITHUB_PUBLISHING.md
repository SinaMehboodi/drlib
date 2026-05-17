# Publishing `drlib` to GitHub

This guide walks you from "code on your laptop" to "public GitHub
repository that anyone can `pip install` from" in eight short steps.

You only need:

* `git` (https://git-scm.com/downloads)
* a GitHub account (https://github.com/signup)
* the GitHub CLI **or** the GitHub web UI — both routes are shown

---

## 0. One-time setup (skip if you already have git configured)

```powershell
git config --global user.name  "Sina Mehboodi"
git config --global user.email "sinamehboodi@gmail.com"
```

Optionally install the GitHub CLI so you can create the repo from the
terminal (`winget install --id GitHub.cli` or
https://cli.github.com/manual/installation), then run `gh auth login`
once.

---

## 1. Decide where the repo's *root* will be

`drlib` lives in `F:\Sina_Data_Job2026\PRJCT\My LIbrary\DrLib\` and the
bundled reference dataset is one level up at
`F:\Sina_Data_Job2026\PRJCT\Data\`.  The README/tutorial reach the data
through `..\Data`, so for everything to work after a `git clone` you have
**three** clean options:

| Option | Repo root | Pros | Cons |
|--------|-----------|------|------|
| **A. recommended** — promote the package to the top | `DrLib\` itself | clean URL `<user>/drlib`, only ships the library | tutorial expects data in `..\Data`; cloners have to download the data themselves *or* set `$DRLIB_DATA_DIR` |
| **B.** bundle data inside the repo | `DrLib\`, with the dataset copied to `DrLib\examples\data\` | tutorial works after a bare clone, no extra env vars | data lives twice on your machine; adds ~16 MB of binary `.mat` to the repo |
| **C.** publish the whole `PRJCT\` folder | `PRJCT\` | preserves your exact layout | repo name has spaces (rename `My LIbrary` to `lib` first); ships extra analysis notebooks |

The steps below assume **Option A** (recommended) — if you pick B or C
the only thing that changes is *which directory* you `cd` into and what
you copy/move first.  Notes for B/C are inlined where needed.

> **Option B quick-start:** before step 2, run
> `mkdir DrLib\examples\data && copy "..\Data\*.mat" DrLib\examples\data\`
> and change `DATA_DIR` in `tutorial.ipynb` cell 2 to `r'examples/data'`.

---

## 2. Initialise the git repository

```powershell
cd "F:\Sina_Data_Job2026\PRJCT\My LIbrary\DrLib"
git init -b main
git status
```

You should see every file in `DrLib\` listed as untracked, **except**
those matched by `.gitignore` (e.g. `__pycache__`, `.pytest_cache/`,
arbitrary `*.mat` files outside `docs_assets/`).

---

## 3. Stage and commit the first snapshot

```powershell
# Add everything that .gitignore allows
git add .

# Sanity-check what you are about to commit (look for stray secrets / .env / large binaries)
git status

git commit -m "Initial public release: drlib 1.1.0"
```

Common things to *not* commit:

* `__pycache__\`, `.pytest_cache\`            — already in `.gitignore`
* `MagneticField.csv` (auto-generated)         — already in `.gitignore`
* the bundled reference dataset (`*.mat`)      — already in `.gitignore`; uncomment the `!docs_assets/*.png` block if you want the README figures committed (default: they ARE committed)
* personal credentials, `.env` files, `.npy` analysis caches

---

## 4. Create the GitHub repository

### Route A — GitHub CLI (one command)

```powershell
gh repo create drlib --public --source=. --remote=origin --description "FMR / CPW / REXS analysis toolkit" --homepage "https://github.com/sina-mehboodi/drlib"
```

That creates the repo *and* sets `origin` to the new URL.  Skip to step 5.

### Route B — GitHub web UI

1. Open https://github.com/new
2. Repository name: `drlib`
3. Visibility: **Public** (so others can `pip install` from it)
4. **Do not** check "Initialize this repository with a README" — you
   already have one.
5. Click **Create repository**.
6. Copy the HTTPS URL shown on the next page, then back in PowerShell:

```powershell
git remote add origin https://github.com/<your-username>/drlib.git
```

---

## 5. Push your first commit

```powershell
git push -u origin main
```

GitHub will ask you to authenticate the first time (a browser window
opens; sign in once and you are done forever).

After the push, open the repo in your browser:
`https://github.com/<your-username>/drlib`.  You should see:

* `README.md` rendered on the landing page, with the bundled figures
  (`docs_assets/*.png`) inlined.
* `tutorial.ipynb` viewable with all outputs already embedded.
* The folder tree (`drlib/`, `tests/`, `docs_assets/`, …).

---

## 6. Tag the release

A tag makes the release installable by version:

```powershell
git tag -a v1.1.0 -m "drlib 1.1.0"
git push origin v1.1.0
```

On the GitHub web UI, go to **Releases → Draft a new release**, pick the
`v1.1.0` tag, paste the contents of `CHANGELOG.md` for that version, and
publish.  This gives users a stable URL to cite.

---

## 7. Let other people install it

Once it's on GitHub, anyone with Python can install your library with one line:

```bash
pip install git+https://github.com/<your-username>/drlib.git
```

To install a specific tagged release:

```bash
pip install git+https://github.com/<your-username>/drlib.git@v1.1.0
```

Or, for development (editable) installs:

```bash
git clone https://github.com/<your-username>/drlib.git
cd drlib
pip install -e ".[dev]"
pytest          # 64 tests should all pass
```

---

## 8. (Optional) Continuous integration

Drop the file below at `.github\workflows\tests.yml` to make GitHub run
your test suite on every push and pull request:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python }}" }
      - run: pip install -e ".[dev]"
      - run: pytest -q
```

Real-data tests (the ones that need `PRJCT/Data`) are auto-skipped in
CI because the dataset is not present; the rest of the suite runs.

---

## 9. (Optional) Publish to PyPI

GitHub is enough for most academic use.  If you also want
`pip install drlib` (no GitHub URL needed), follow:

1. Register on https://pypi.org and https://test.pypi.org.
2. Install `build` and `twine`:
   ```powershell
   pip install build twine
   python -m build
   twine upload --repository testpypi dist/*    # try it on TestPyPI first
   twine upload dist/*                          # the real thing
   ```
3. The `drlib` name on PyPI may already be taken; pick a unique variant
   (e.g. `mwn-drlib`, `drlib-fmr`) and update `name = "..."` in
   `pyproject.toml` before re-building.

---

## Troubleshooting

* **"author email mismatch"** during `git commit` — fix with the
  `git config` lines in step 0.
* **`fatal: 'origin' does not appear to be a git repository`** — you
  haven't run `git remote add origin ...` yet (step 4 route B).
* **Large `.mat` files refused** by GitHub (`> 100 MB`) — either remove
  them from the repo and host them on Zenodo/OSF, or use
  [Git LFS](https://git-lfs.com/).
* **Tutorial figures missing on GitHub** — `docs_assets\` should be
  committed.  Confirm with `git ls-files docs_assets/`. If empty,
  check the `!docs_assets/*.png` line in `.gitignore`.
* **`pip install -e .` fails with "Multiple top-level packages"** —
  someone added a stray `tests/__init__.py` or similar; double-check
  the `[tool.setuptools.packages.find]` block in `pyproject.toml`.

---

## In short, the absolute minimum commands

```powershell
cd "F:\Sina_Data_Job2026\PRJCT\My LIbrary\DrLib"
git init -b main
git add .
git commit -m "Initial public release: drlib 1.1.0"
gh repo create drlib --public --source=. --remote=origin
git push -u origin main
git tag -a v1.1.0 -m "drlib 1.1.0" && git push origin v1.1.0
```

Done — your library is on GitHub.
