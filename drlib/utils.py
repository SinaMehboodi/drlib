"""
drlib.utils
===========

Small, self-contained helpers used throughout the package:

* :func:`timeit`           – decorator that prints how long a call takes.
* :func:`set_size`         – LaTeX-friendly figure-size calculator.
* :func:`read_measurement_txt` – read the first floating-point number on each
                                 line of a ``measurement.txt`` log file.

These helpers are pure-Python / NumPy and have no dependency on any of the
other drlib modules, so they can safely be imported from anywhere in the
package.
"""

from __future__ import annotations

import re
from functools import wraps
from time import time
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Matplotlib style helper
# ----------------------------------------------------------------------
def safe_style(name: Optional[str]) -> None:
    """``plt.style.use`` that silently falls back to ``"default"``
    when the requested style sheet is not installed.

    Useful so the package's plotting helpers work out of the box on a
    fresh machine that does not have the bespoke ``"Beam"`` LaTeX style
    installed.
    """
    if name is None:
        return
    try:
        plt.style.use(name)
    except (OSError, ValueError):
        plt.style.use("default")


# ----------------------------------------------------------------------
# Timing decorator
# ----------------------------------------------------------------------
def timeit(f: Callable) -> Callable:
    """Decorator that prints how long the wrapped function takes to run.

    Use as a plain decorator above any function or method::

        @timeit
        def slow_thing():
            ...

    Parameters
    ----------
    f : callable
        The function to wrap.

    Returns
    -------
    callable
        A wrapped version of ``f`` that prints ``func:'name' took: X.XXXX sec``
        every time it is called.
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result
    return wrap


# ----------------------------------------------------------------------
# Figure-size helper
# ----------------------------------------------------------------------
def set_size(width: float, fraction: float = 1.0) -> Tuple[float, float]:
    r"""Compute a (width, height) figure size in inches from a LaTeX width.

    Uses the golden ratio for the height so the figure is aesthetically
    pleasing and does not need to be rescaled inside the document.

    Parameters
    ----------
    width : float
        Document ``\textwidth`` or ``\columnwidth`` in **points** (pt).
        For a single-column article, ``246`` pt is typical; for a
        two-column article use the column width.
    fraction : float, optional
        Fraction of ``width`` the figure should occupy. Default ``1.0``.

    Returns
    -------
    (float, float)
        ``(figure_width_in, figure_height_in)`` in inches, ready to be
        passed to ``plt.figure(figsize=...)``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from drlib.utils import set_size
    >>> fig = plt.figure(figsize=set_size(width=246, fraction=1.0))
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # 1 inch = 72.27 pt (TeX point)
    inches_per_pt = 1 / 72.27

    # Golden ratio for aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    return (fig_width_in, fig_height_in)


# ----------------------------------------------------------------------
# LabView "measurement.txt" parser
# ----------------------------------------------------------------------
def read_measurement_txt(file_path: str) -> List[float]:
    """Read the first floating-point number from every line of *measurement.txt*.

    Some LabView VIs save acquisition parameters as a free-form text log
    (one parameter per line, value followed by units). This helper grabs the
    first ``\\d+\\.\\d+`` token on each line so the values can be re-used
    in post-processing.

    Parameters
    ----------
    file_path : str
        Directory **containing** the file ``measurement.txt``.
        The function appends ``\\measurement.txt`` automatically.

    Returns
    -------
    list of float
        One value per parsed line, in order of appearance.

    Notes
    -----
    Lines without a decimal number are skipped silently.
    """
    numbers: List[float] = []
    with open(file_path + r"\measurement.txt", "r") as file:
        for line in file:
            match = re.search(r"\d+\.\d+", line)
            if match:
                numbers.append(float(match.group(0)))
    return numbers
