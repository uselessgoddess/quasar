"""One command for every step between "no corpus" and "look at what it built".

The commands are deliberately small and file-based rather than a pipeline
object: each one reads something off disk, writes something to disk, and prints
what it did. That is what makes the loop debuggable — every intermediate is
sitting there to be looked at, and any step can be re-run on its own without
rebuilding the ones before it.

    quasar-factorio build corpus --count 20000 --real data/blueprints.jsonl
    quasar-factorio preview corpus/preview.png --count 12
    quasar-factorio heatmap corpus/occupancy.png --count 400
    quasar-factorio grade runs/nano/samples.jsonl --sheet runs/nano/sheet.png
    quasar-factorio plot runs/nano/train.log runs/nano/metrics.png --grade runs/nano/*.jsonl
    quasar-factorio export sample.txt

Printing is brutalist for the same reason the renderer is: fixed-width columns,
no colour, no spinner, nothing that survives a pipe into a log file as garbage.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
from collections.abc import Iterator, Sequence

from . import blueprint as bp
from . import dataset, grammar, plots, prototypes, render, synth, validate

# Rule width for the printed tables. 72 so that two of them fit side by side in
# a 150-column terminal and neither wraps in an 80-column one.
RULE = 72


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "run"):
        parser.print_help()
        return 2
    try:
        return args.run(args)
    except (OSError, ValueError, bp.BlueprintError, grammar.ParseError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quasar-factorio",
        description="Build, look at and grade a Factorio blueprint corpus.",
    )
    sub = parser.add_subparsers()

    build = sub.add_parser("build", help="write a corpus quasar train can read")
    build.add_argument("out", type=pathlib.Path)
    build.add_argument("--count", type=int, default=20_000, help="designs to draw")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--variants", type=int, default=4, help="forms kept per design")
    build.add_argument("--valid-every", type=int, default=dataset.VALID_EVERY)
    build.add_argument("--prompts", type=int, default=256, help="held-out eval prompts")
    build.add_argument(
        "--real",
        type=pathlib.Path,
        help="also mix in human blueprints from a cache written by tools/fetch_blueprints.py",
    )
    build.add_argument(
        "--real-limit",
        type=int,
        default=0,
        help="cap the human half; 0 takes everything the cache yields",
    )
    build.set_defaults(run=_build)

    preview = sub.add_parser("preview", help="a contact sheet of graded designs")
    preview.add_argument("out", type=pathlib.Path)
    _source(preview)
    preview.add_argument("--columns", type=int, default=4)
    preview.set_defaults(run=_preview)

    heat = sub.add_parser("heatmap", help="where entities land, over many designs")
    heat.add_argument("out", type=pathlib.Path)
    _source(heat, count=400)
    heat.add_argument("--scale", type=int, default=8)
    heat.set_defaults(run=_heatmap)

    grade = sub.add_parser("grade", help="score generations the way the game would")
    grade.add_argument("samples", type=pathlib.Path, help="jsonl, or - for stdin")
    grade.add_argument("--sheet", type=pathlib.Path, help="also draw a contact sheet")
    grade.add_argument("--json", type=pathlib.Path, help="also write the summary as json")
    grade.add_argument("--columns", type=int, default=4)
    grade.add_argument(
        "--order",
        choices=("best", "worst", "given"),
        default="best",
        help="sheet order; `given` keeps the prompt order, which is what makes "
        "one sheet per checkpoint comparable frame to frame",
    )
    grade.set_defaults(run=_grade)

    plot = sub.add_parser("plot", help="metric panels from a training log")
    plot.add_argument("log", type=pathlib.Path, help="quasar train stdout, or - for stdin")
    plot.add_argument("out", type=pathlib.Path)
    plot.add_argument(
        "--grade",
        type=pathlib.Path,
        nargs="*",
        default=(),
        metavar="SAMPLES.JSONL",
        help="also plot the grader; several files draw the curve over training",
    )
    plot.add_argument("--columns", type=int, default=2)
    plot.set_defaults(run=_plot)

    draw = sub.add_parser("render", help="draw one document, by suffix: .png or .svg")
    draw.add_argument("document", type=pathlib.Path, help="a text file, or - for stdin")
    draw.add_argument("out", type=pathlib.Path)
    draw.add_argument("--scale", type=int, default=16)
    draw.set_defaults(run=_render)

    export = sub.add_parser("export", help="a blueprint string to paste into the game")
    export.add_argument("document", type=pathlib.Path, help="a text file, or - for stdin")
    export.add_argument("--label", default="")
    export.set_defaults(run=_export)

    return parser


def _source(parser: argparse.ArgumentParser, count: int = 12) -> None:
    """Where blueprints come from: a built corpus, or freshly drawn ones."""
    parser.add_argument("--corpus", type=pathlib.Path, help="read from a built corpus instead")
    parser.add_argument("--split", default="train", choices=("train", "valid"))
    parser.add_argument("--count", type=int, default=count)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", help="only this generator")


# --- commands --------------------------------------------------------------


def _build(args) -> int:
    data = prototypes.load()
    harvest: collections.Counter[str] = collections.Counter()
    extra = None
    if args.real:
        from . import real

        extra = real.designs(
            args.real,
            data=data,
            limit=args.real_limit,
            variants=args.variants,
            seed=args.seed,
            counts=harvest,
        )
    stats = dataset.build(
        args.out,
        args.count,
        seed=args.seed,
        variants=args.variants,
        data=data,
        extra=extra,
        valid_every=args.valid_every,
        prompts=args.prompts,
    )
    _rule("CORPUS", str(args.out))
    _rows(
        [
            ("distinct layouts", stats.designs),
            ("documents", stats.documents),
            ("layouts redrawn", stats.repeats),
            ("duplicates dropped", stats.duplicates),
            ("rejected", stats.rejected),
            ("train tokens", stats.train_tokens),
            ("valid tokens", stats.valid_tokens),
            ("vocab", stats.vocab_size),
            ("longest document", stats.longest),
        ]
    )
    _rule("MIXTURE")
    total = sum(stats.kinds.values()) or 1
    for kind, seen in sorted(stats.kinds.items(), key=lambda item: -item[1]):
        _bar(kind, seen / total, f"{seen}")
    if harvest:
        # What the human half threw away and why. "3,000 designs came in" says
        # nothing about whether the cache is worth extending; "1,900 were Space
        # Age" says to fetch a different slice of it.
        _rule("HARVEST", str(args.real))
        _rows(sorted(harvest.items(), key=lambda item: -item[1]))
    if stats.rejected:
        # The generators are held to a perfect score by their own tests, so this
        # is a bug in one of them rather than a tolerable loss.
        print(f"\nWARNING: {stats.rejected} documents failed the grader and were dropped")
    return 0


def _preview(args) -> int:
    data = prototypes.load()
    cards = []
    for label, text, blueprint in _blueprints(args, data):
        report = validate.grade(text, data)
        cards.append(
            render.card(
                blueprint,
                data,
                title=label,
                lines=_caption(report),
                accent=render.GOOD if report.valid else render.ALERT,
            )
        )
    if not cards:
        raise ValueError("nothing to preview")
    _write(args.out, render.sheet(cards, columns=args.columns).png())
    print(f"{len(cards)} designs -> {args.out}")
    return 0


def _heatmap(args) -> int:
    data = prototypes.load()
    blueprints = [blueprint for _, _, blueprint in _blueprints(args, data)]
    counts = render.trim(render.occupancy(blueprints, data))
    title = f"OCCUPANCY {len(blueprints)} DESIGNS {len(counts[0])}X{len(counts)}"
    _write(args.out, render.heatmap(counts, scale=args.scale, title=title).png())
    print(f"{len(blueprints)} designs -> {args.out}")
    return 0


def _grade(args) -> int:
    data = prototypes.load()
    samples = list(_samples(args.samples))
    if not samples:
        raise ValueError(f"no samples in {args.samples}")
    reports = [(sample, validate.grade(sample["text"], data)) for sample in samples]
    summary = validate.summarise([report for _, report in reports])

    _rule("GRADE", str(args.samples))
    _rows([("samples", summary.samples)])
    for name, value in [
        ("parses", summary.parse_rate),
        ("valid", summary.valid_rate),
        ("spec honoured", summary.spec_rate),
        ("powered", summary.mean_powered),
        ("inserters connected", summary.mean_connected),
        ("belts lead somewhere", summary.mean_belts),
        ("quality", summary.mean_quality),
    ]:
        _bar(name, value, f"{value:.3f}")
    _rows([("mean entities", f"{summary.mean_entities:.1f}")])

    # Reported apart from the block above because it answers a different
    # question. Everything above is legality; this is whether items move.
    _rule("ITEM FLOW", f"{summary.ported} of {summary.samples} declared ports")
    flow_rows = [
        ("machines fed", summary.mean_fed),
        ("machines working", summary.mean_working),
        ("flow score", summary.mean_flow),
    ]
    if summary.ported:
        flow_rows = [
            ("outputs delivered", summary.mean_delivers),
            *flow_rows,
            ("inside the zone", summary.zone_rate),
        ]
    for name, value in flow_rows:
        _bar(name, value, f"{value:.3f}")
    _rows(
        [
            ("designs leaking items", f"{summary.leak_rate:.3f}"),
            ("belts over two items", f"{summary.mixed_rate:.3f}"),
        ]
    )

    if summary.errors:
        _rule("WHY IT FAILED")
        worst = max(summary.errors.values())
        for message, seen in summary.errors.items():
            _bar(message[:32], seen / worst, f"{seen}")

    if args.json:
        _write(args.json, (json.dumps(summary.to_dict(), indent=2) + "\n").encode())
    if args.sheet:
        _sheet(args.sheet, reports, data, columns=args.columns, order=args.order)
        print(f"\n{args.sheet}")
    return 0


def _plot(args) -> int:
    """Everything a run produced, on one sheet.

    The grader panels are optional but are the point: a loss curve says the
    model is fitting the corpus, and only the validity curve says it is
    learning to build something the game would accept.
    """
    run = plots.read_training(_read(args.log))
    if not (run.loss or run.valid):
        raise ValueError(f"no training metrics in {args.log}")
    panels = plots.training_panels(run)

    graded = [(_step_of(path), _summary_of(path)) for path in args.grade]
    if graded:
        panels.extend(_grade_panels(sorted(graded)))

    _write(args.out, plots.board(panels, columns=args.columns).png())
    _rule("PLOT", str(args.out))
    _rows(
        [
            ("plan", run.plan or "?"),
            ("logged steps", len(run.loss)),
            ("evaluations", len(run.valid)),
            ("final valid loss", f"{run.final:.4f}" if run.final is not None else "-"),
            ("graded checkpoints", len(graded)),
        ]
    )
    return 0


def _grade_panels(graded: Sequence[tuple[float, validate.Summary]]) -> list[render.Raster]:
    """Grader panels: the verdict now, and — with more than one file — over time."""
    # `flow` last and deliberately in the same panel as the rest: the point of
    # the analysis this came out of is that the legality metrics saturate while
    # the flow metric still has somewhere to go, and putting them on separate
    # charts would hide exactly that comparison.
    metrics = (
        ("valid", plots.INKS[3], lambda s: s.valid_rate),
        ("parses", plots.INKS[0], lambda s: s.parse_rate),
        ("spec", plots.INKS[2], lambda s: s.spec_rate),
        ("quality", plots.INKS[1], lambda s: s.mean_quality),
        ("flow", plots.INKS[5], lambda s: s.mean_flow),
    )
    step, last = graded[-1]
    panels = [
        plots.bars(
            [(name, read(last)) for name, _, read in metrics],
            title=f"GRADE AT STEP {step:.0f} ({last.samples} SAMPLES)",
            top=1.0,
        )
    ]
    if len(graded) > 1:
        panels.append(
            plots.chart(
                [
                    plots.Series(name, tuple((at, read(s)) for at, s in graded), color)
                    for name, color, read in metrics
                ],
                title="GRADE OVER TRAINING",
                floor=0.0,
            )
        )
    if last.errors:
        worst = sorted(last.errors.items(), key=lambda item: -item[1])[:6]
        panels.append(plots.bars([(m[:16], n) for m, n in worst], title="WHY IT FAILED"))
    return panels


def _step_of(path: pathlib.Path) -> float:
    """The step a samples file belongs to, taken from the digits in its name.

    `samples-000400.jsonl` is what the training loop writes per checkpoint, so
    the step is already in the filename and does not need a sidecar.
    """
    digits = "".join(char if char.isdigit() else " " for char in path.stem).split()
    return float(digits[-1]) if digits else 0.0


def _summary_of(path: pathlib.Path) -> validate.Summary:
    data = prototypes.load()
    reports = [validate.grade(sample["text"], data) for sample in _samples(path)]
    if not reports:
        raise ValueError(f"no samples in {path}")
    return validate.summarise(reports)


def _render(args) -> int:
    data = prototypes.load()
    text = _read(args.document)
    blueprint, spec = grammar.parse(text, data)
    report = validate.grade(text, data)
    title = spec.kind.upper() if spec else "BLUEPRINT"
    if args.out.suffix == ".svg":
        _write(
            args.out,
            render.svg(
                blueprint, data, scale=args.scale, title=title, lines=_caption(report)
            ).encode(),
        )
    else:
        accent = render.GOOD if report.valid else render.ALERT
        card = render.card(blueprint, data, title=title, lines=_caption(report), accent=accent)
        _write(args.out, card.png())
    print(f"{report.entities} entities, {report.width}x{report.height} -> {args.out}")
    return 0


def _export(args) -> int:
    """The end of the whole pipeline: something to paste into the game.

    Printed to stdout on its own line, so `quasar-factorio export x.txt | xclip`
    does the obvious thing.
    """
    data = prototypes.load()
    blueprint, spec = grammar.parse(_read(args.document), data)
    label = args.label or (spec.kind if spec else "")
    print(bp.to_string(blueprint, data, label=label))
    return 0


# --- shared plumbing -------------------------------------------------------


def _blueprints(args, data) -> Iterator[tuple[str, str, bp.Blueprint]]:
    """`(label, document, blueprint)` from a corpus or from the generators."""
    if args.corpus:
        yield from _from_corpus(args.corpus / args.split, args.count, data, args.kind)
        return
    for index in range(args.count):
        rng = random.Random(args.seed * 1_000_003 + index)
        if args.kind:
            blueprint, spec = synth.GENERATORS[args.kind](rng, data)
        else:
            blueprint, spec = synth.sample(rng, data)
        yield spec.kind, grammar.serialise(blueprint, data, spec), blueprint


def _from_corpus(directory, count, data, kind) -> Iterator[tuple[str, str, bp.Blueprint]]:
    from . import shards, tokenizer

    encoder = tokenizer.Encoder(data)
    tokens, meta = shards.read(directory)
    seen = 0
    for ids in shards.documents(tokens, meta.eos):
        text = encoder.decode(ids)
        blueprint, spec = grammar.parse(text, data)
        if kind and (spec is None or spec.kind != kind):
            continue
        yield (spec.kind if spec else "?"), text, blueprint
        seen += 1
        if seen >= count:
            return


def _samples(path: pathlib.Path) -> Iterator[dict]:
    """Generations, from jsonl or from a plain file of one document per line.

    Tolerant on purpose: this reads whatever the sampler happened to write, and
    a run that produced 500 samples should not be unreadable because the field
    was called `completion` rather than `text`.
    """
    for line in _read(path).splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            yield {"text": line}
            continue
        record = json.loads(line)
        for field in ("text", "completion", "generation", "sample", "output"):
            if field in record:
                yield {**record, "text": record[field]}
                break
        else:
            raise ValueError(f"no generated text in {sorted(record)}")


def _sheet(out, reports, data, *, columns: int, order: str) -> None:
    """A contact sheet of the generations.

    Sorted rather than sampled by default: the interesting frames of a training
    run are the extremes, and a random twelve from five hundred shows neither.
    `given` is the exception — a timelapse needs the same prompt in the same
    cell in every frame, or the whole sheet reshuffles between checkpoints and
    nothing can be followed.
    """

    def rank(item):
        return (item[1].valid, item[1].quality())

    ranked = reports
    if order != "given":
        ranked = sorted(reports, key=rank, reverse=order == "best")
    cards = []
    for sample, report in ranked[: columns * 3]:
        # Unknown entities are dropped by the renderer, which is exactly the case
        # worth looking at, so parse failures still get a card where they can.
        try:
            blueprint, _ = grammar.parse(sample["text"], data)
        except grammar.ParseError:
            blueprint = bp.Blueprint(entities=[])
        cards.append(
            render.card(
                blueprint,
                data,
                title=sample.get("kind", report.spec_kind or "SAMPLE"),
                lines=_caption(report),
                accent=render.GOOD if report.valid else render.ALERT,
            )
        )
    _write(out, render.sheet(cards, columns=columns).png())


def _caption(report: validate.Report) -> list[str]:
    """Up to three lines of grader verdict, which is all a card has room for.

    The third line only appears for a design that declared ports, and it says
    the one thing a reader cannot see by looking at the picture: whether the
    item the module promised actually comes out of it.
    """
    if not report.parsed:
        return [f"PARSE FAIL AT {report.error_at}", report.error[:38].upper()]
    head = "VALID" if report.valid else "INVALID"
    if report.overlaps:
        head += f" OVERLAP {report.overlaps}"
    elif not report.fits:
        head += " OFF GRID"
    elif report.illegal_recipes:
        head += f" BAD RECIPE {report.illegal_recipes}"
    lines = [
        f"{head}  {report.entities} ENT  {report.width}X{report.height}",
        f"PWR {report.powered:.2f}  INS {report.connected_inserters:.2f}"
        f"  BLT {report.belts_lead_somewhere:.2f}",
    ]
    if report.ported:
        starved = f" MISSING {report.missing[0][:14].upper()}" if report.missing else ""
        lines.append(f"OUT {report.delivers:.2f}  FED {report.fed:.2f}{starved}")
    return lines


def _read(path: pathlib.Path) -> str:
    return sys.stdin.read() if str(path) == "-" else path.read_text()


def _write(path: pathlib.Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


def _rule(title: str, note: str = "") -> None:
    head = f"== {title} "
    if note:
        head += f"[{note}] "
    print(f"\n{head}{'=' * max(0, RULE - len(head))}")


def _rows(rows) -> None:
    for name, value in rows:
        print(f"{name:<24}{value:>12}")


def _bar(name: str, fraction: float, value: str) -> None:
    """A fraction as a bar, because a column of decimals hides the shape."""
    width = RULE - 24 - 8
    filled = max(0, min(width, round(fraction * width)))
    print(f"{name[:23]:<24}{'#' * filled}{'.' * (width - filled)}{value:>8}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
