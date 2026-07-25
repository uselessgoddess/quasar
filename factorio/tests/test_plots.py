"""The plots are read, not returned, so the tests read them back.

Two things can go wrong here and neither shows up in a return type. The parser
can silently agree with a log format the trainer no longer prints, leaving every
panel empty; and a label can be drawn a few pixels outside the panel it belongs
to, which looks like a smudge on the sheet next door. So the log fixtures below
are copied verbatim out of a real run, and the drawing tests count pixels.
"""

import struct

import pytest

from quasar_factorio import plots, render

#: Verbatim stdout of `quasar train nano --steps 20`, TUI suppressed. The
#: leading blank on the `valid:` lines and the trailing fields after `bpb` are
#: part of what the parser has to tolerate, so they are kept.
SMOKE = """training plan: 20 optimizer steps | 0.00B tokens | 2048 tokens/step
  valid: loss 2.9885 | ppl 19.86 | bpb 0.6583 | 40960 tokens
optimizer step 20/20 | loss 2.2270 | lr 3.00e-4 | 185 tok/s | 0.00 TFLOP/s | 0.00B tokens | ETA 0.0h
  valid: loss 2.0839 | ppl 8.04 | bpb 0.4590 | 40960 tokens
final: loss 2.0839 | ppl 8.04 | bpb 0.4590 | 40960 tokens
"""

LONGER = """training plan: 400 optimizer steps | 0.01B tokens | 16384 tokens/step
  valid: loss 6.2000 | ppl 492.75 | bpb 1.3600 | 112581 tokens
optimizer step 100/400 | loss 3.1000 | lr 3.00e-4 | 21000 tok/s | 4.20 TFLOP/s | 0.00B tokens
  valid: loss 2.9000 | ppl 18.17 | bpb 0.6400 | 112581 tokens
optimizer step 200/400 | loss 1.9000 | lr 2.10e-4 | 21500 tok/s | 4.30 TFLOP/s | 0.00B tokens
optimizer step 300/400 | loss 1.4000 | lr 9.00e-5 | 21400 tok/s | 4.28 TFLOP/s | 0.01B tokens
  valid: loss 1.3000 | ppl 3.67 | bpb 0.2900 | 112581 tokens
optimizer step 400/400 | loss 1.1000 | lr 1.00e-6 | 21600 tok/s | 4.32 TFLOP/s | 0.01B tokens
  valid: loss 1.0500 | ppl 2.86 | bpb 0.2300 | 112581 tokens
final: loss 1.0500 | ppl 2.86 | bpb 0.2300 | 112581 tokens
"""


def pixel(canvas: render.Raster, x: int, y: int) -> tuple[int, int, int]:
    at = (y * canvas.width + x) * 3
    return tuple(canvas.pixels[at : at + 3])


def count(canvas: render.Raster, color: tuple[int, int, int]) -> int:
    packed = bytes(color)
    return sum(canvas.pixels[at : at + 3] == packed for at in range(0, len(canvas.pixels), 3))


# --- reading the log -------------------------------------------------------


def test_a_real_training_log_parses_into_every_metric_it_printed():
    run = plots.read_training(SMOKE)
    assert run.steps == 20
    assert run.plan == "20 optimizer steps | 0.00B tokens | 2048 tokens/step"
    assert run.loss == [(20.0, 2.2270)]
    assert run.lr == [(20.0, 3.00e-4)]
    assert run.throughput == [(20.0, 185.0)]
    assert run.tflops == [(20.0, 0.0)]


def test_the_evaluation_before_the_first_step_is_kept_at_step_zero():
    """It is the only untrained score in the run, and the drop from it is the point."""
    run = plots.read_training(SMOKE)
    assert run.valid[0] == (0.0, 2.9885)
    assert run.perplexity[0] == (0.0, 19.86)
    assert run.bits_per_byte[0] == (0.0, 0.6583)


def test_the_final_line_does_not_duplicate_the_last_evaluation():
    run = plots.read_training(SMOKE)
    assert run.valid == [(0.0, 2.9885), (20.0, 2.0839)]
    assert run.final == 2.0839


def test_validation_is_attributed_to_the_step_it_was_measured_at():
    run = plots.read_training(LONGER)
    assert [step for step, _ in run.valid] == [0.0, 100.0, 300.0, 400.0]
    assert [step for step, _ in run.loss] == [100.0, 200.0, 300.0, 400.0]


def test_a_log_with_nothing_in_it_parses_to_nothing_rather_than_failing():
    run = plots.read_training("compiling quasar v0.1.0\nerror: no such preset\n")
    assert (run.steps, run.plan, run.loss, run.valid) == (0, "", [], [])
    assert run.final is None


# --- numbers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value,shown",
    [
        (0.0, "0"),
        (2.9885, "2.99"),
        (19.86, "19.9"),
        (185.0, "185"),
        (21600.0, "21.6K"),
        (3.0e-4, "3.0E-4"),
        (1.2e9, "1.2E9"),
    ],
)
def test_numbers_are_spelled_the_way_the_panels_show_them(value, shown):
    assert plots._number(value) == shown


@pytest.mark.parametrize("low,high", [(1e-6, 3e-4), (0.0, 21600.0), (1.05, 6.2), (0.0, 1.0)])
@pytest.mark.parametrize("log", [False, True])
def test_every_axis_label_fits_inside_the_margin_it_is_drawn_in(low, high, log):
    """A label wider than `PAD_LEFT` is drawn at a negative x and clipped to a smear."""
    for index in range(plots.GRID_LINES + 1):
        value = plots._unscale(index / plots.GRID_LINES, max(low, 1e-6) if log else low, high, log)
        label = plots._number(value)
        assert render.text_width(label) + 4 <= plots.PAD_LEFT, f"{value} -> {label!r}"


# --- drawing ---------------------------------------------------------------


def test_a_chart_is_a_png_with_the_geometry_it_was_asked_for():
    canvas = plots.chart(
        [plots.Series("loss", ((0.0, 6.2), (400.0, 1.05)), plots.INKS[1])],
        title="LOSS",
        width=320,
        height=160,
    )
    assert (canvas.width, canvas.height) == (320, 160)
    blob = canvas.png()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", blob[16:24]) == (320, 160)


def test_a_series_is_drawn_in_its_own_ink_across_the_whole_panel():
    canvas = plots.chart(
        [plots.Series("loss", tuple((float(x), 6.2 - x / 100) for x in range(401)), plots.INKS[1])],
        title="LOSS",
    )
    columns = {
        (at // 3) % canvas.width
        for at in range(0, len(canvas.pixels), 3)
        if canvas.pixels[at : at + 3] == bytes(plots.INKS[1])
    }
    # It reaches both ends of the plotting area rather than bunching at one edge.
    assert min(columns) <= plots.PAD_LEFT + 1
    assert max(columns) >= canvas.width - plots.PAD_RIGHT - 2


def test_an_empty_series_says_so_instead_of_drawing_an_empty_frame():
    canvas = plots.chart([plots.Series("loss", ())], title="LOSS")
    assert count(canvas, render.FAINT) > 0
    assert count(canvas, render.PAPER) > 0  # the title is still there
    assert count(canvas, plots.INKS[1]) == 0


def test_a_log_axis_survives_a_metric_that_touches_zero():
    """`0.00 TFLOP/s` is what a CPU run prints, and log10(0) is not a number."""
    canvas = plots.chart(
        [plots.Series("tflops", ((0.0, 0.0), (10.0, 0.0), (20.0, 4.3)))],
        title="TFLOPS",
        log=True,
    )
    assert count(canvas, render.PAPER) > 0


def test_a_single_point_still_leaves_a_mark():
    canvas = plots.chart([plots.Series("valid", ((20.0, 2.08),), plots.INKS[2])], title="VALID")
    assert count(canvas, plots.INKS[2]) >= 4


def test_a_bar_is_as_long_as_its_share_of_the_ceiling():
    def ink(value: float) -> int:
        return count(plots.bars([("a", 0.0), ("b", value)], title="GRADE", top=1.0), plots.INKS[1])

    assert ink(0.0) == 0
    assert 0 < ink(0.25) < ink(0.5) < ink(1.0)


def test_a_bar_past_the_ceiling_is_clamped_rather_than_drawn_off_the_panel():
    assert count(plots.bars([("a", 0.0), ("b", 4.0)], title="G", top=1.0), plots.INKS[1]) == count(
        plots.bars([("a", 0.0), ("b", 1.0)], title="G", top=1.0), plots.INKS[1]
    )


def test_more_rows_than_fit_are_dropped_rather_than_drawn_over_the_frame():
    canvas = plots.bars([(f"row {n}", 1.0) for n in range(60)], title="MANY", height=120)
    bottom = [pixel(canvas, x, canvas.height - 2) for x in range(canvas.width)]
    assert set(bottom) <= {render.GROUND, render.RULE}


def test_the_board_tiles_panels_row_major_in_equal_cells():
    panels = [plots.chart([], title=f"P{n}", width=100, height=50) for n in range(3)]
    sheet = plots.board(panels, columns=2, gap=8)
    assert (sheet.width, sheet.height) == (8 + 2 * 108, 8 + 2 * 58)
    # Each panel's top-left frame pixel lands on its own cell origin.
    for index, (x, y) in enumerate([(8, 8), (116, 8), (8, 66)]):
        assert pixel(sheet, x, y) == render.RULE, index


def test_a_board_with_no_panels_is_a_pixel_not_a_crash():
    assert (plots.board([]).width, plots.board([]).height) == (1, 1)


def test_the_training_board_covers_every_metric_the_trainer_prints():
    run = plots.read_training(LONGER)
    panels = plots.training_panels(run, width=240, height=120)
    assert len(panels) == 4
    assert all((panel.width, panel.height) == (240, 120) for panel in panels)
    # None of them fell back to "no data": each has ink that is not the frame.
    for panel in panels:
        assert count(panel, render.FAINT) > 0
        assert sum(count(panel, ink) for ink in plots.INKS) > 0
