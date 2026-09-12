from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paperlab.core import Tick
from paperlab.fly import frame
from paperlab.fly_trace import Assay, input_frames
from paperlab.fly_visual import apply_view, encoding
from paperlab.news import News


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "vendor/stonkfly"))
    news = News(enabled=False)
    def render(total, view="original", denomination=1):
        ticks = [Tick(1700000000+i*300, denomination*(1+total*i/99)*.995,
                      denomination*(1+total*i/99)*1.005, product="SYNTHETIC") for i in range(100)]
        original = frame(ticks, 99, news)
        return apply_view(original, ticks, 99, view)
    yield render
    news.db.close()


@pytest.mark.parametrize("small,large", [(.01,.2),(-.002,-.2)])
def test_fixed_returns_separates_documented_original_collisions(renderer, small, large):
    assert np.array_equal(renderer(small), renderer(large))
    a, b = renderer(small, "fixed_returns"), renderer(large, "fixed_returns")
    assert not np.array_equal(a, b)
    assert np.array_equal(a[:28], renderer(small)[:28])
    assert np.array_equal(a[140:], renderer(small)[140:])
    assert np.array_equal(a, renderer(small, "fixed_returns", denomination=1024))


def test_fixed_returns_ignores_future_and_old_history_and_marks_clipping():
    rgb = np.zeros((180,320,3), dtype=np.uint8)
    ticks = [SimpleNamespace(mid=1.) for _ in range(100)]
    ticks[-1] = SimpleNamespace(mid=1000.)
    expected = apply_view(rgb, ticks, 99, "fixed_returns")
    assert np.any(np.all(expected == (210,130,20), axis=2))
    extended = [SimpleNamespace(mid=1e-99)] + ticks + [SimpleNamespace(mid=1e99)]
    assert np.array_equal(expected, apply_view(rgb, extended, 100, "fixed_returns"))
    assert np.array_equal(rgb, np.zeros_like(rgb))  # Candidate does not mutate its source frame.
    with pytest.raises(ValueError): apply_view(rgb, [SimpleNamespace(mid=0.)], 0, "fixed_returns")
    with pytest.raises(ValueError): apply_view(rgb, ticks, 100, "fixed_returns")


def test_original_is_unchanged_and_amplitude_is_bounded(renderer):
    # The renderer fixture makes the vendored RGB adapter importable without loading a graph.
    rgb = np.arange(180*320*3, dtype=np.uint8).reshape(180,320,3)
    assert apply_view(rgb, [], 0, "original") is rgb
    for change in ({"amplitude":0}, {"amplitude":True}, {"probe_amplitude":2}, {"amplitude":float("nan")}):
        with pytest.raises(ValueError): Assay(**change)
    full = input_frames(Assay(view="fixed_returns", steps=1))[0]
    small = input_frames(Assay(view="fixed_returns", steps=1, amplitude=.05))[0]
    assert not np.array_equal(full, small)
    assert encoding("fixed_returns")["price_scale"] == "fixed_asinh_log_return_v1"
