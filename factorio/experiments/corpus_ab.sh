#!/bin/sh
# Wall clock of a corpus build, one revision against another.
#
# `corpus_cost.py` answers "where does the time go" inside one checkout. This
# answers the other question — "is this revision faster than that one" — and it
# has to be a shell script because the two revisions cannot be imported into the
# same Python process.
#
# It prints the build statistics next to every timing on purpose. A corpus build
# that got faster while producing fewer documents did not get faster, and the
# only way to see that is to have both numbers on the same line.
#
# Run:  git worktree add --detach /tmp/wt-main upstream/main
#       factorio/experiments/corpus_ab.sh /tmp/wt-main main
#       factorio/experiments/corpus_ab.sh . branch
set -eu

tree=${1:?usage: corpus_ab.sh <tree> [label] [runs]}
label=${2:-$(basename "$tree")}
runs=${3:-3}
count=${COUNT:-1500}
seed=${SEED:-7}

for run in $(seq 1 "$runs"); do
    out=$(mktemp -d)
    (
        cd "$tree/factorio" &&
            PYTHONPATH=src python3 -c '
import sys, time

from quasar_factorio import cli

started = time.perf_counter()
cli.main(["build", "--count", sys.argv[2], "--seed", sys.argv[3], sys.argv[1]])
print(f"ELAPSED {time.perf_counter() - started:.1f}")
' "$out" "$count" "$seed"
    ) >"$out/build.log" 2>&1
    awk -v label="$label" -v run="$run" '
        /distinct layouts/ { designs = $3 }
        /^documents/       { documents = $2 }
        /train tokens/     { tokens = $3 }
        /^ELAPSED/         { spent = $2 }
        END { printf "%-12s run%-3d %6.1f s  %5d designs %6d docs %8d tokens\n",
                     label, run, spent, designs, documents, tokens }
    ' "$out/build.log"
    rm -rf "$out"
done
