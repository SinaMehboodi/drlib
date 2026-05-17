"""Tests for ``drlib.utils`` (set_size, timeit, safe_style, read_measurement_txt)."""

from __future__ import annotations

from drlib.utils import read_measurement_txt, safe_style, set_size, timeit


def test_set_size_returns_tuple_of_two_positive_floats():
    w, h = set_size(width=246, fraction=1.0)
    assert isinstance(w, float) and isinstance(h, float)
    assert w > 0 and h > 0
    # Golden-ratio aspect (height ≈ 0.618 × width)
    assert 0.55 < h / w < 0.65


def test_set_size_fraction_scales_width():
    w_full, h_full = set_size(246, 1.0)
    w_half, h_half = set_size(246, 0.5)
    assert abs(w_half * 2 - w_full) < 1e-9
    assert abs(h_half * 2 - h_full) < 1e-9


def test_timeit_wraps_and_returns(capsys):
    @timeit
    def f(x):
        return x + 1

    out = f(40)
    assert out == 41
    captured = capsys.readouterr().out
    assert "took" in captured


def test_safe_style_accepts_none_and_default():
    safe_style(None)        # no-op
    safe_style("default")   # always available


def test_safe_style_falls_back_silently_for_missing_style():
    # 'Beam' is not bundled with matplotlib; should not raise.
    safe_style("Beam_definitely_not_installed")


def test_read_measurement_txt(tmp_path):
    f = tmp_path / "measurement.txt"
    f.write_text(
        "field 3.14 mT\n"
        "freq 2.71 GHz\n"
        "no number here\n"
        "another 0.50 unit\n",
        encoding="utf-8",
    )
    nums = read_measurement_txt(str(tmp_path))
    assert nums == [3.14, 2.71, 0.50]
