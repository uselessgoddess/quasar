"""Metric plots, drawn the same way the blueprints are.

Matplotlib would be the obvious answer and is the wrong one here. It is a 30 MB
wheel plus numpy for what amounts to drawing lines into a framebuffer, it does
not match the renderer next door, and its defaults — anti-aliased curves, soft
grids, a white page — are the opposite of what this harness is trying to show.
:class:`~quasar_factorio.render.Raster` already writes PNG out of `zlib` and
`struct`, so the plots cost one more module and no dependency at all.

Same rules as the blueprint renderer: near-black ground, hairline grid, hard
edges, no anti-aliasing, the 5x7 bitmap font, and every number that matters
printed as text somewhere on the panel rather than left to be read off an axis.
A plot that has to be measured with a ruler has thrown away the number it was
given.

The input is whatever `quasar train` printed. Parsing stdout rather than asking
for a metrics file is deliberate: the log always exists, it exists for runs that
crashed, and it exists for runs that happened before this module did.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .render import CHUNK, FAINT, GROUND, PAPER, RGB, RULE, Raster, text_width

#: Series colours, in the order they get handed out. Picked to stay apart on a
#: near-black ground and to survive being drawn one pixel wide.
INKS: tuple[RGB, ...] = (
    (238, 238, 238),
    (232, 128, 48),
    (86, 168, 210),
    (110, 200, 120),
    (226, 74, 58),
    (196, 182, 64),
    (168, 130, 220),
)

#: Panel geometry. The margins are the widest label the font can print in the
#: space, not a guess: five characters at scale 1 is 36 px.
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 42, 10, 22, 16
GRID_LINES = 4


@dataclass(frozen=True)
class Series:
    """One line on a chart, as `(x, y)` in data units."""

    label: str
    points: tuple[tuple[float, float], ...]
    color: RGB = PAPER
    #: Draw as a filled area down to the axis instead of a line. For anything
    #: cumulative, where the area is the quantity and the line is an edge.
    area: bool = False


@dataclass
class Training:
    """What a training log said, one list per metric.

    Every list is `(step, value)`. Validation is sparser than training by design
    — `eval_every` is usually 50 or 100 steps — so the lists are different
    lengths and are never zipped.
    """

    steps: int = 0
    plan: str = ""
    loss: list[tuple[float, float]] = field(default_factory=list)
    valid: list[tuple[float, float]] = field(default_factory=list)
    perplexity: list[tuple[float, float]] = field(default_factory=list)
    bits_per_byte: list[tuple[float, float]] = field(default_factory=list)
    lr: list[tuple[float, float]] = field(default_factory=list)
    throughput: list[tuple[float, float]] = field(default_factory=list)
    tflops: list[tuple[float, float]] = field(default_factory=list)

    @property
    def final(self) -> float | None:
        """The last validation loss, which is the number the run is judged on."""
        return self.valid[-1][1] if self.valid else None


_STEP = re.compile(
    r"optimizer step (?P<step>\d+)/(?P<total>\d+) \| loss (?P<loss>[\d.]+) \| "
    r"lr (?P<lr>[\d.e+-]+) \| (?P<throughput>[\d]+) tok/s \| (?P<tflops>[\d.]+) TFLOP/s"
)
_VALID = re.compile(
    r"(?:valid|final): loss (?P<loss>[\d.]+) \| ppl (?P<ppl>[\d.]+) \| bpb (?P<bpb>[\d.]+)"
)
_PLAN = re.compile(r"training plan: (?P<plan>.+)")


def read_training(text: str) -> Training:
    """Parse `quasar train` stdout.

    A `valid:` line carries no step of its own, so it is attributed to the last
    training step seen — which is where it was measured. The evaluation printed
    before the first training line lands at step 0, and that one is worth
    keeping: it is the only untrained score in the run.
    """
    run = Training()
    step = 0.0
    for line in text.splitlines():
        if plan := _PLAN.search(line):
            run.plan = plan["plan"].strip()
            continue
        if found := _STEP.search(line):
            step = float(found["step"])
            run.steps = max(run.steps, int(found["total"]))
            run.loss.append((step, float(found["loss"])))
            run.lr.append((step, float(found["lr"])))
            run.throughput.append((step, float(found["throughput"])))
            run.tflops.append((step, float(found["tflops"])))
            continue
        if found := _VALID.search(line):
            # `final:` repeats the last evaluation; keeping both would draw a
            # flat tail onto the end of every curve.
            point = (step, float(found["loss"]))
            if run.valid and run.valid[-1] == point:
                continue
            run.valid.append(point)
            run.perplexity.append((step, float(found["ppl"])))
            run.bits_per_byte.append((step, float(found["bpb"])))
    return run


def chart(
    series: Sequence[Series],
    *,
    title: str,
    width: int = 480,
    height: int = 200,
    log: bool = False,
    floor: float | None = None,
) -> Raster:
    """One panel: framed, gridded, labelled, with the last value spelled out.

    `log` puts the y axis in log space, which is what a loss curve wants — the
    interesting part of training is the last 10% of the drop, and a linear axis
    spends nine tenths of its height on the first epoch.
    """
    canvas = Raster(width, height, GROUND)
    canvas.frame(0, 0, width, height, RULE)
    canvas.text(6, 7, title[: (width - 12) // 6], PAPER)

    drawn = [line for line in series if line.points]
    plot = (PAD_LEFT, PAD_TOP, width - PAD_LEFT - PAD_RIGHT, height - PAD_TOP - PAD_BOTTOM)
    if not drawn:
        canvas.text(PAD_LEFT, height // 2, "no data", FAINT)
        return canvas

    xs = [x for line in drawn for x, _ in line.points]
    ys = [y for line in drawn for _, y in line.points]
    low, high = (floor if floor is not None else min(ys)), max(ys)
    if log:
        # A log axis cannot show a zero, and a metric that touches zero is
        # usually a metric that started there.
        positive = [y for y in ys if y > 0]
        low, high = (min(positive), max(positive)) if positive else (1.0, 1.0)
    if high - low < 1e-12:
        low, high = low - 0.5, high + 0.5

    _grid(canvas, plot, low, high, log)
    _axis(canvas, plot, min(xs), max(xs))
    for line in drawn:
        _draw(canvas, plot, line, min(xs), max(xs), low, high, log)
    _legend(canvas, plot, drawn, width)
    return canvas


def bars(
    rows: Sequence[tuple[str, float]],
    *,
    title: str,
    width: int = 480,
    height: int = 200,
    top: float | None = None,
) -> Raster:
    """A horizontal bar per row, value printed at the end of the bar.

    For the categorical half of the metrics — the grader's verdicts, the mixture
    of kinds — where a line chart would be a lie about continuity.
    """
    canvas = Raster(width, height, GROUND)
    canvas.frame(0, 0, width, height, RULE)
    canvas.text(6, 7, title[: (width - 12) // 6], PAPER)
    if not rows:
        canvas.text(PAD_LEFT, height // 2, "no data", FAINT)
        return canvas

    ceiling = top if top is not None else max((value for _, value in rows), default=1.0)
    ceiling = ceiling if ceiling > 0 else 1.0
    label = max(text_width(name[:16]) for name, _ in rows) + 8
    span = width - label - 58
    pitch = max(7, (height - PAD_TOP - 6) // len(rows))

    for index, (name, value) in enumerate(rows):
        y = PAD_TOP + index * pitch
        if y + 7 > height - 4:
            break
        canvas.text(6, y, name[:16], FAINT)
        canvas.box(label, y, span, 7, RULE)
        filled = round(span * max(0.0, min(1.0, value / ceiling)))
        canvas.box(label, y, filled, 7, INKS[index % len(INKS)])
        canvas.text(label + span + 6, y, _number(value), PAPER)
    return canvas


def board(panels: Sequence[Raster], *, columns: int = 2, gap: int = 8) -> Raster:
    """Lay panels out on one sheet, row-major, left-aligned in equal columns."""
    if not panels:
        return Raster(1, 1, GROUND)
    columns = max(1, min(columns, len(panels)))
    rows = math.ceil(len(panels) / columns)
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    sheet = Raster(gap + columns * (cell_w + gap), gap + rows * (cell_h + gap), GROUND)
    for index, panel in enumerate(panels):
        x = gap + (index % columns) * (cell_w + gap)
        y = gap + (index // columns) * (cell_h + gap)
        sheet.blit(panel, x, y)
    return sheet


def training_panels(run: Training, *, width: int = 480, height: int = 200) -> list[Raster]:
    """The four panels worth looking at after a run.

    Loss against validation loss is the plot; the other three are there to
    answer the questions that come immediately after "is it learning" — was the
    schedule what I asked for, did the card stay busy, and how far did
    perplexity actually move.
    """
    size = {"width": width, "height": height}
    return [
        chart(
            [
                Series("train", tuple(run.loss), INKS[0]),
                Series("valid", tuple(run.valid), INKS[1]),
            ],
            title="LOSS (LOG)",
            log=True,
            **size,
        ),
        chart(
            [Series("ppl", tuple(run.perplexity), INKS[2])],
            title="PERPLEXITY (LOG)",
            log=True,
            **size,
        ),
        chart([Series("lr", tuple(run.lr), INKS[3])], title="LEARNING RATE", floor=0.0, **size),
        chart(
            [Series("tok/s", tuple(run.throughput), INKS[5], area=True)],
            title="THROUGHPUT TOK/S",
            floor=0.0,
            **size,
        ),
    ]


def _grid(canvas: Raster, plot: tuple[int, int, int, int], low: float, high: float, log: bool):
    x, y, w, h = plot
    for index in range(GRID_LINES + 1):
        fraction = index / GRID_LINES
        at = y + h - round(fraction * h)
        canvas.box(x, at, w, 1, CHUNK if index == 0 else RULE)
        value = _unscale(fraction, low, high, log)
        canvas.text(x - text_width(_number(value)) - 4, at - 3, _number(value), FAINT)


def _axis(canvas: Raster, plot: tuple[int, int, int, int], first: float, last: float) -> None:
    x, y, w, h = plot
    canvas.box(x, y + h, w, 1, CHUNK)
    canvas.text(x, y + h + 4, _number(first), FAINT)
    end = _number(last)
    canvas.text(x + w - text_width(end), y + h + 4, end, FAINT)


def _draw(
    canvas: Raster,
    plot: tuple[int, int, int, int],
    series: Series,
    left: float,
    right: float,
    low: float,
    high: float,
    log: bool,
) -> None:
    x, y, w, h = plot
    span = right - left or 1.0

    def place(point: tuple[float, float]) -> tuple[int, int]:
        column = x + round((point[0] - left) / span * (w - 1))
        return column, y + h - round(_scale(point[1], low, high, log) * (h - 1))

    plotted = [place(point) for point in series.points]
    for (x0, y0), (x1, y1) in zip(plotted, plotted[1:], strict=False):
        if series.area:
            for column in range(x0, x1 + 1):
                at = y0 + round((y1 - y0) * (column - x0) / max(1, x1 - x0))
                canvas.box(column, at, 1, y + h - at, tuple(c // 3 for c in series.color))
        _segment(canvas, x0, y0, x1, y1, series.color)
    if len(plotted) == 1:
        canvas.box(plotted[0][0], plotted[0][1], 2, 2, series.color)
    if plotted:
        # The endpoint gets a marker: on a 200-px panel the last few steps of a
        # converged run are a flat line, and the reader wants to know where it
        # stops rather than where the axis does.
        last_x, last_y = plotted[-1]
        canvas.box(last_x - 1, last_y - 1, 3, 3, series.color)


def _segment(canvas: Raster, x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
    """Bresenham. One pixel wide, no anti-aliasing, on purpose."""
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        canvas.dot(x0, y0, color)
        if x0 == x1 and y0 == y1:
            return
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _legend(canvas: Raster, plot: tuple[int, int, int, int], series: Iterable[Series], width: int):
    cursor = width - PAD_RIGHT
    for line in reversed(list(series)):
        if not line.label:
            continue
        cursor -= text_width(line.label) + 12
        canvas.box(cursor, 8, 5, 5, line.color)
        canvas.text(cursor + 8, 7, line.label, FAINT)
        cursor -= 2


def _scale(value: float, low: float, high: float, log: bool) -> float:
    """Value to `0..1` up the panel, clamped so an outlier cannot draw outside."""
    if log:
        value, low, high = (math.log10(max(v, 1e-12)) for v in (value, low, high))
    return max(0.0, min(1.0, (value - low) / (high - low or 1.0)))


def _unscale(fraction: float, low: float, high: float, log: bool) -> float:
    if log:
        low, high = math.log10(max(low, 1e-12)), math.log10(max(high, 1e-12))
        return 10 ** (low + fraction * (high - low))
    return low + fraction * (high - low)


def _number(value: float) -> str:
    """Four characters of significance, whatever the magnitude.

    Axis labels are drawn in a 5x7 font in a 42-px margin, so `0.00012345` and
    `123456789` both have to become something a human reads at a glance.
    """
    magnitude = abs(value)
    # Below 0.01 the plain spelling needs seven characters, which is wider than
    # the margin it has to fit in — `0.00225` would be drawn off the panel.
    if magnitude and (magnitude < 1e-2 or magnitude >= 1e7):
        exponent = math.floor(math.log10(magnitude))
        return f"{value / 10**exponent:.1f}E{exponent}"
    for limit, suffix in ((1e6, "M"), (1e3, "K")):
        if magnitude >= limit:
            return f"{value / limit:.1f}{suffix}"
    if magnitude >= 100:
        return f"{value:.0f}"
    return f"{value:.3g}" if magnitude else "0"
