#!/usr/bin/env python3
"""Collect human-made blueprints from the open web into a local cache.

The generators in `synth.py` saturate: 32,000 draws produce 6,344 distinct
designs, and six of the ten generators are exhausted by a few hundred. A model
trained to the Chinchilla budget on that corpus sees each design twenty times
over, which is where a small model stops learning Factorio and starts learning
the templates. Real blueprints are the only source of layouts nobody wrote a
generator for.

Where they come from
--------------------

`factorioprints.com` is a Firebase app with a public read API, which is what
makes this a hundred lines rather than a scraper:

    GET /blueprintSummaries.json?shallow=true   -> {id: true, ...}   (17,780)
    GET /blueprints/<id>.json                   -> one record

Ids are Firebase push keys, so sorting them sorts by upload date: `--order new`
is the 2.0/Space Age end, `--order old` is the 0.15-1.1 end. The 1.1 end is
worth more here, because the prototype table this harness carries is 1.1's — a
Space Age blueprint arrives as a handful of recognisable belts around entities
that do not exist, and `real.py` throws it out. Hence the default.

`factorio.school` hosts a second collection behind
`/api/blueprintSummaries/page/N`; it answered 502 for the whole of this work, so
it is named here rather than implemented. The record shape it returns is close
enough that a second `Source` would be the whole change.

What is written
---------------

One JSON object per line — id, title, author, tags, favourites, date and the
blueprint string — plus a manifest carrying the source, the count and the
sha256 of the cache file. The cache itself is *not* committed: it is other
people's work, it is hundreds of megabytes, and the manifest is enough to say
whether two runs used the same bytes.

    python factorio/tools/fetch_blueprints.py --count 2000
    python factorio/tools/fetch_blueprints.py --count 4000   # resumes

Re-running extends the cache rather than refetching it, so an interrupted crawl
costs nothing, and `--count` is a target for the cache rather than a batch size.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

SOURCE = "https://facorio-blueprints.firebaseio.com"
SUMMARIES = f"{SOURCE}/blueprintSummaries.json?shallow=true"
RECORD = SOURCE + "/blueprints/{id}.json"

# Where the cache lands. Under `factorio/data/`, which is gitignored.
DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "blueprints.jsonl"

# Politeness. Eight in flight against a Firebase read endpoint is nothing, and
# the whole crawl is a few hundred megabytes served from a CDN — but it is
# somebody else's bill, so it stays low and backs off rather than hammering.
JOBS = 8
RETRIES = 4
BACKOFF = 1.5

# A blueprint string longer than this is a book of a whole base. Books are worth
# walking — a third of the site is books and they hold the tileable blocks this
# corpus wants — but past a megabyte the decompression alone costs more than the
# designs are worth.
MAX_STRING = 1 << 20


def get(url: str, *, timeout: float, retries: int = RETRIES) -> bytes:
    """One GET, with backoff on the failures that are worth retrying.

    Timeouts and 5xx are the server asking for a moment; a 404 is an answer,
    and retrying it four times is just noise on somebody else's logs.
    """
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if attempt + 1 < retries:
            time.sleep(BACKOFF**attempt)
    raise OSError(f"gave up on {url} after {retries} attempts")


def ids(*, timeout: float) -> list[str]:
    """Every blueprint id the site knows about, oldest first."""
    payload = json.loads(get(SUMMARIES, timeout=timeout))
    return sorted(payload)


def record(identifier: str, *, timeout: float) -> dict | None:
    """One cache line, or `None` if there is no usable blueprint string in it.

    Everything the site stores that this harness cannot use — descriptions,
    image ids, the favourites map, which is thousands of user ids on a popular
    print — is dropped here rather than cached and ignored later.
    """
    body = json.loads(get(RECORD.format(id=identifier), timeout=timeout))
    if not isinstance(body, dict):
        return None
    string = body.get("blueprintString")
    if not isinstance(string, str) or not string.strip():
        return None
    if len(string) > MAX_STRING:
        return None
    author = body.get("author") or {}
    return {
        "id": identifier,
        "source": "factorioprints",
        "title": str(body.get("title") or ""),
        "author": str(author.get("displayName") or body.get("authorId") or ""),
        "tags": [str(tag) for tag in body.get("tags") or []],
        "favourites": int(body.get("numberOfFavorites") or 0),
        "created": str(body.get("createdDate") or ""),
        "string": string.strip(),
    }


def cached(path: pathlib.Path) -> set[str]:
    """Ids already in the cache, so a second run is a continuation.

    A half-written last line — the crawl was killed mid-flush — is tolerated by
    skipping it; the id it belonged to is simply refetched.
    """
    if not path.exists():
        return set()
    seen = set()
    for line in path.read_text(errors="replace").splitlines():
        try:
            seen.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            continue
    return seen


def fetch(
    path: pathlib.Path,
    wanted: list[str],
    *,
    jobs: int,
    timeout: float,
    log=sys.stderr,
) -> int:
    """Fetch `wanted` into `path`, appending one line each. Returns the count.

    Written as they arrive rather than collected and dumped: a crawl of ten
    thousand records takes twenty minutes, and losing all of it to a broken pipe
    at minute nineteen is not a tradeoff worth the tidier code.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    with path.open("a") as sink, futures.ThreadPoolExecutor(jobs) as pool:
        pending = {pool.submit(record, i, timeout=timeout): i for i in wanted}
        for done in futures.as_completed(pending):
            try:
                body = done.result()
            except (OSError, ValueError) as error:
                failed += 1
                print(f"  {pending[done]}: {error}", file=log)
                continue
            if body is None:
                continue
            sink.write(json.dumps(body, sort_keys=True) + "\n")
            written += 1
            if written % 200 == 0:
                sink.flush()
                print(f"  {written}/{len(wanted)}", file=log)
    if failed:
        print(f"{failed} records could not be fetched", file=log)
    return written


def manifest(path: pathlib.Path) -> dict:
    """What a corpus built from this cache can cite.

    The digest is of the file, not of the records: the point is to be able to
    say two training runs read the same bytes, and that is a property of the
    file the loader opened.
    """
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return {
        "source": SOURCE,
        "records": lines,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "third-party blueprints; cached, not committed",
    }


def _spread(every: list[str], count: int) -> list[str]:
    """An even stride across the whole site, then the rest behind it.

    A capped crawl that takes a prefix takes one era: the oldest end is 0.15,
    where a third of the strings predate the format this harness reads, and the
    newest end is Space Age, whose entities are not in the 1.1 prototype table.
    A stride samples every year the site has been up, so raising `--count` later
    keeps everything already cached and fills in between it.
    """
    if count <= 0 or count >= len(every):
        return every
    step = len(every) // count
    sampled = every[::step]
    return sampled + [identifier for identifier in every if identifier not in set(sampled)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=2_000, help="records to hold in total")
    parser.add_argument(
        "--order",
        choices=("old", "new", "spread"),
        default="spread",
        help="which end of the site to crawl; ids sort by upload date, and "
        "`spread` takes an even stride across all of it",
    )
    parser.add_argument("--jobs", type=int, default=JOBS)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    print(f"listing {SUMMARIES}", file=sys.stderr)
    every = ids(timeout=args.timeout)
    if args.order == "new":
        every.reverse()
    elif args.order == "spread":
        every = _spread(every, args.count)
    have = cached(args.out)
    wanted = [identifier for identifier in every if identifier not in have]
    wanted = wanted[: max(0, args.count - len(have))]
    print(f"{len(every)} known, {len(have)} cached, fetching {len(wanted)}", file=sys.stderr)

    if wanted:
        fetch(args.out, wanted, jobs=args.jobs, timeout=args.timeout)
    if not args.out.exists():
        print("nothing fetched", file=sys.stderr)
        return 1

    summary = manifest(args.out)
    args.out.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{args.out}: {summary['records']} records, {summary['bytes'] / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
