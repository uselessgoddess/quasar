"""The commands, exercised the way a run uses them.

Everything below goes through `main(argv)` rather than the functions it calls,
because the failures worth catching here live in the wiring: an argument that
never reaches the builder, an output written to the wrong path, an exception
class that escapes the handler and turns a typo into a traceback.

The corpus is built once for the whole module — it is the slowest thing here and
nothing mutates it.
"""

import json

import pytest

from quasar_factorio import grammar, prototypes, synth
from quasar_factorio.cli import main

DATA = prototypes.load()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    assert main(["build", str(out), "--count", "16", "--variants", "2", "--prompts", "4"]) == 0
    return out


@pytest.fixture
def document(tmp_path):
    """One generated document on disk, which is what most commands take."""
    import random

    blueprint, spec = synth.GENERATORS["smelter-column"](random.Random(3), DATA)
    path = tmp_path / "design.txt"
    path.write_text(grammar.serialise(blueprint, DATA, spec))
    return path


# --- the parser ------------------------------------------------------------


def test_no_command_prints_help_rather_than_a_traceback(capsys):
    assert main([]) == 2
    assert "quasar-factorio" in capsys.readouterr().out


def test_a_missing_file_is_an_error_message_not_a_traceback(capsys, tmp_path):
    assert main(["render", str(tmp_path / "nope.txt"), str(tmp_path / "out.png")]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_a_document_that_is_not_one_is_an_error_message(capsys, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("<bp> <e> not-an-entity x00 y00 d0 </bp>\n")
    assert main(["render", str(bad), str(tmp_path / "out.png")]) == 1
    assert "error:" in capsys.readouterr().err


# --- build -----------------------------------------------------------------


def test_build_writes_everything_quasar_train_needs(corpus):
    assert (corpus / "tokenizer.json").is_file()
    assert (corpus / "train" / "meta.json").is_file()
    assert (corpus / "valid" / "meta.json").is_file()
    assert list((corpus / "train").glob("*.bin"))


def test_the_manifest_agrees_with_what_was_asked_for(corpus):
    manifest = json.loads((corpus / "manifest.json").read_text())
    assert manifest["designs"] == 16
    # Every generator is held to a perfect score by its own tests, so a single
    # rejection here is a bug rather than an acceptable loss.
    assert manifest["rejected"] == 0
    assert manifest["vocab_size"] == 739


def test_the_held_out_prompts_carry_what_a_grader_needs(corpus):
    records = [json.loads(line) for line in (corpus / "prompts.jsonl").read_text().splitlines()]
    # `--prompts` is a ceiling: prompts come from held-out designs, and at
    # `VALID_EVERY` = 25 a 16-design corpus has one to give.
    assert 1 <= len(records) <= 4
    for record in records:
        assert record["prompt"].startswith("<bp>")
        # The reference is the document the prompt was cut from: without it a
        # generation can be graded but not compared.
        assert record["reference"].startswith(record["prompt"])
        assert record["kind"] in synth.GENERATORS


# --- looking at it ---------------------------------------------------------


def test_preview_draws_one_card_per_design(corpus, tmp_path):
    out = tmp_path / "preview.png"
    assert main(["preview", str(out), "--corpus", str(corpus), "--count", "4"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_preview_can_draw_from_the_generators_without_a_corpus(tmp_path):
    out = tmp_path / "fresh.png"
    assert main(["preview", str(out), "--count", "2", "--kind", "balancer"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_heatmap_is_drawn_over_many_designs(tmp_path):
    out = tmp_path / "heat.png"
    assert main(["heatmap", str(out), "--count", "8", "--scale", "4"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("suffix,magic", [(".png", b"\x89PNG\r\n\x1a\n"), (".svg", b"<svg")])
def test_render_picks_its_format_from_the_suffix(document, tmp_path, suffix, magic):
    out = tmp_path / f"design{suffix}"
    assert main(["render", str(document), str(out)]) == 0
    assert out.read_bytes().startswith(magic)


def test_export_prints_a_blueprint_string_the_game_would_take(document, capsys):
    assert main(["export", str(document), "--label", "smelter"]) == 0
    printed = capsys.readouterr().out.strip()
    # Version byte, then base64 of a zlib-compressed json object.
    assert printed.startswith("0")
    assert len(printed) > 32


# --- grading ---------------------------------------------------------------


def test_a_corpus_document_grades_as_valid(document, tmp_path, capsys):
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"text": document.read_text()}) + "\n")
    summary = tmp_path / "summary.json"

    assert main(["grade", str(samples), "--json", str(summary)]) == 0

    scores = json.loads(summary.read_text())
    assert scores["samples"] == 1
    assert (scores["parse_rate"], scores["valid_rate"]) == (1.0, 1.0)
    assert "valid" in capsys.readouterr().out


def test_a_generation_that_is_not_a_blueprint_is_graded_not_crashed_on(tmp_path, capsys):
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"text": "<bp> <e> steel-chest x99 y99"}) + "\n")
    summary = tmp_path / "summary.json"

    sheet = tmp_path / "s.png"
    assert main(["grade", str(samples), "--json", str(summary), "--sheet", str(sheet)]) == 0

    scores = json.loads(summary.read_text())
    assert scores["parse_rate"] == 0.0
    # The sheet is still drawn: a failure is the frame worth looking at.
    assert sheet.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "WHY IT FAILED" in capsys.readouterr().out


@pytest.mark.parametrize("order", ["best", "worst", "given"])
def test_the_sheet_can_be_ordered_three_ways_and_given_keeps_the_prompt_order(
    document, tmp_path, order
):
    """`given` is what makes one sheet per checkpoint into a timelapse: sorted
    frames reshuffle between checkpoints and nothing can be followed."""
    good, bad = document.read_text(), "<bp> not a blueprint </bp>"
    samples = tmp_path / "samples.jsonl"
    samples.write_text("".join(json.dumps({"text": t}) + "\n" for t in (bad, good)))

    sheet = tmp_path / f"{order}.png"
    assert main(["grade", str(samples), "--sheet", str(sheet), "--order", order]) == 0
    assert sheet.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_sheet_order_changes_which_design_lands_in_the_first_cell(document, tmp_path):
    good, bad = document.read_text(), "<bp> not a blueprint </bp>"
    samples = tmp_path / "samples.jsonl"
    samples.write_text("".join(json.dumps({"text": t}) + "\n" for t in (bad, good)))

    drawn = {}
    for order in ("best", "worst", "given"):
        sheet = tmp_path / f"{order}.png"
        assert main(["grade", str(samples), "--sheet", str(sheet), "--order", order]) == 0
        drawn[order] = sheet.read_bytes()

    # `given` put the failure first, as the file did; `best` put the design there.
    assert drawn["given"] == drawn["worst"]
    assert drawn["best"] != drawn["given"]


def test_grading_an_empty_file_says_so(tmp_path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    assert main(["grade", str(empty)]) == 1
    assert "no samples" in capsys.readouterr().err


def test_a_sampler_that_named_the_field_something_else_is_still_read(tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"completion": "<bp> </bp>"}) + "\n")
    summary = tmp_path / "summary.json"
    assert main(["grade", str(samples), "--json", str(summary)]) == 0
    assert json.loads(summary.read_text())["samples"] == 1


# --- plotting --------------------------------------------------------------


LOG = """training plan: 200 optimizer steps | 0.00B tokens | 16384 tokens/step
  valid: loss 6.2000 | ppl 492.75 | bpb 1.3600 | 112581 tokens
optimizer step 100/200 | loss 2.1000 | lr 3.00e-4 | 21000 tok/s | 4.20 TFLOP/s
  valid: loss 1.9000 | ppl 6.69 | bpb 0.4200 | 112581 tokens
optimizer step 200/200 | loss 1.2000 | lr 1.00e-6 | 21500 tok/s | 4.30 TFLOP/s
  valid: loss 1.1000 | ppl 3.00 | bpb 0.2400 | 112581 tokens
final: loss 1.1000 | ppl 3.00 | bpb 0.2400 | 112581 tokens
"""


def test_plot_draws_a_sheet_from_a_training_log(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text(LOG)
    out = tmp_path / "metrics.png"

    assert main(["plot", str(log), str(out)]) == 0

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    printed = capsys.readouterr().out
    assert "1.1000" in printed and "200 optimizer steps" in printed


def test_graded_checkpoints_add_panels_and_are_ordered_by_their_filenames(
    document, tmp_path, capsys
):
    log = tmp_path / "train.log"
    log.write_text(LOG)
    text = document.read_text()
    for step in (200, 100):  # out of order on the command line, on purpose
        (tmp_path / f"samples-{step:06d}.jsonl").write_text(json.dumps({"text": text}) + "\n")

    plain = tmp_path / "plain.png"
    graded = tmp_path / "graded.png"
    assert main(["plot", str(log), str(plain)]) == 0
    assert (
        main(
            [
                "plot",
                str(log),
                str(graded),
                "--grade",
                str(tmp_path / "samples-000200.jsonl"),
                str(tmp_path / "samples-000100.jsonl"),
            ]
        )
        == 0
    )

    assert len(graded.read_bytes()) > len(plain.read_bytes())
    assert "graded checkpoints" in capsys.readouterr().out


def test_a_log_with_no_metrics_in_it_is_an_error(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("error: unknown preset `nanno`\n")
    assert main(["plot", str(log), str(tmp_path / "out.png")]) == 1
    assert "no training metrics" in capsys.readouterr().err
