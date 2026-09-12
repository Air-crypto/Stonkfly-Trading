"""Explicit sensory alternatives. No neural action, return target, or future data."""
import math

import numpy as np
from PIL import Image, ImageDraw

VIEWS = ("original", "price_only", "blank", "fixed_returns")
LOG_KNEE = .01
MAX_LOG_RETURN = math.log(100)


def encoding(view):
    if view not in VIEWS:
        raise ValueError("Unknown visual input")
    if view != "fixed_returns":
        return {"view": view, "price_scale": "upstream_window_range" if view != "blank" else "none"}
    return {"view": view, "price_scale": "fixed_asinh_log_return_v1",
            "reference": "first price in the trailing 100-observation window",
            "log_return_knee": LOG_KNEE, "maximum_absolute_log_return": MAX_LOG_RETURN,
            "formula": "y = 85 - 51 * asinh(log(mid/reference)/0.01) / asinh(log(100)/0.01)",
            "bounds": "Clipped at 0.01x and 100x reference; amber dots mark clipped observations",
            "interpretation": "Fixed monotone sensory transform; no suggested action or P&L input"}


def apply_view(rgb, ticks, index, view):
    """Preserve the original adapter byte-for-byte unless an alternative is requested."""
    if view not in VIEWS:
        raise ValueError("Unknown visual input")
    if view == "original":
        return rgb
    if view == "blank":
        rgb[:] = 128
        return rgb
    if view == "price_only":
        rgb[:28] = (235, 240, 249)
        rgb[140:] = (235, 240, 249)
        return rgb
    if type(index) is not int or not 0 <= index < len(ticks):
        raise ValueError("Use an existing observation index")
    history = np.asarray([t.mid for t in ticks[max(0, index-99):index+1]], dtype=float)
    if not len(history) or not np.isfinite(history).all() or (history <= 0).any():
        raise ValueError("Fixed returns need a finite, positive observed price history")
    # Difference of logs avoids overflow when two valid prices have an extreme ratio.
    logs = np.log(history)
    returns = logs-logs[0]
    position = np.arcsinh(np.clip(returns, -MAX_LOG_RETURN, MAX_LOG_RETURN)/LOG_KNEE)
    position /= math.asinh(MAX_LOG_RETURN/LOG_KNEE)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 28, 319, 139), fill=(235, 240, 249))
    for x in range(12, 310, 30):
        draw.line((x, 34, x, 136), fill=(200, 212, 233))
    for y in (34, 59, 85, 111, 136):
        draw.line((10, y, 308, y), fill=(200, 212, 233))
    draw.line((10, 85, 308, 85), fill=(150, 166, 190))
    points = [(12+i*294/max(1, len(history)-1), 85-51*float(v)) for i, v in enumerate(position)]
    for a, b in zip(points, points[1:]):
        draw.line((*a, *b), fill=(0, 101, 183) if b[1] <= a[1] else (197, 37, 78), width=3)
    for (x, y), value in zip(points, returns):
        color = (210, 130, 20) if abs(value) > MAX_LOG_RETURN+1e-12 else (27, 39, 81)
        draw.rectangle((x-1, y-1, x+1, y+1), fill=color)
    return np.asarray(image, dtype=np.uint8).copy()
