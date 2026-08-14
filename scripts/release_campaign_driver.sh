#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# release_campaign_driver.sh — campaign driver for the luhdm (alpha x M_DM x lambda)
# data-release cube. Runs ONE build_release.py invocation per lambda-shard
# (crash isolation + per-shard timing). noatm pass first (full product in ~1 h),
# then atm with tags-first so the tag/massless slices land early for the +4 h
# V1 launch-abort gate. NO `set -e`: a SHARD_FAIL never aborts the campaign;
# just relaunch this script and it retries only the missing shards (build_release
# skips shards that already exist).
#
# Shards are written to  $HOME/release_shards/{atm,noatm}  — deliberately OUTSIDE
# any rsync --delete target, so a code push can never wipe compute results.
#
# ── OPS (run these from the LOCAL machine) ──────────────────────────────────
#
# 0. Name the compute node once; every command below uses it. Any ssh alias or
#    hostname works — the campaign itself is host-agnostic.
#      export REMOTE_HOST="${REMOTE_HOST:-remote-node}"
#
# 1. Push code (NEVER touches ~/release_shards; --delete only prunes
#    the luhdm/ and scripts/ trees under code/luhdm/):
#      rsync -az --delete --exclude __pycache__ luhdm scripts "$REMOTE_HOST":code/luhdm/
#
# 2. (optional) timing probe BEFORE the real launch — smallest-lambda ils on a
#    tiny grid to sanity-check the sub-2 um ODE cost extrapolation:
#      ssh "$REMOTE_HOST" 'cd code/luhdm && .venv/bin/python scripts/build_release.py \
#          --pass atm --il-start 0 --il-end 3 --n-a 6 --m-tier 60 \
#          --shard-dir /tmp/probe_shards --workers 80'
#
# 3. Launch the full campaign under nohup and detach:
#      ssh "$REMOTE_HOST" 'cd code/luhdm && nohup bash scripts/release_campaign_driver.sh \
#          > release_build.log 2>&1 &'
#
# 4. Local watch/pull loop — mirror shards back every ~2.5 min until DONE:
#      until ssh "$REMOTE_HOST" "grep -q '] DONE$' code/luhdm/release_build.log"; do
#          rsync -az "$REMOTE_HOST":release_shards/ ~/release_shards/
#          sleep 150
#      done
#      rsync -az "$REMOTE_HOST":release_shards/ ~/release_shards/   # final sync
#
# 5. Safe-kill the campaign on the node (the [b]racket trick keeps pkill from
#    matching its own command line):
#      ssh "$REMOTE_HOST" 'pkill -f "[b]uild_release"'
# ─────────────────────────────────────────────────────────────────────────────

set -u   # NOT -e: SHARD_FAIL must not abort the run

# REPO_DIR override lets the campaign run from a relocated repo copy
# (e.g. /stor1 when the root filesystem is full).
cd "${REPO_DIR:-$HOME/code/luhdm}"

PY="${PYTHON_BIN:-.venv/bin/python}"

# Label for the log banner. The campaign runs wherever it is started; this is
# only what it calls itself when `hostname` is unavailable, and it is the same
# knob the OPS block above uses to address the node from the local machine.
REMOTE_HOST="${REMOTE_HOST:-remote-node}"

OUT="${SHARD_OUT:-$HOME/release_shards}"
WORKERS="${WORKERS:-$(nproc)}"
mkdir -p "$OUT/atm" "$OUT/noatm"

ts() { date '+%F %T'; }

# run_shard PASS IL — one build_release invocation, timestamped sentinels.
#
# Env knobs (all optional; unset reproduces the historical capped campaign):
#   BCAP=0.1     impact-parameter cap [m]; BCAP=none runs UNCAPPED (v7-quick)
#   LSET=v7quick reduced lambda axis (20 um / 200 um / 2 mm + 200 m validation)
#   QMIN=1000    analysis window [GeV]; unset => build_release's config.Q_THRESH
#   KNL=isotropic-folded  projection kernel; unset => the planar-signed default
#   NA=44        coupling grid points; unset => build_release's own default
#   NMC_HI=100000  two-tier MC: cells whose base extremeness lands in
#                [0.90, 1.0) are re-evaluated on a second table with this many
#                trials (seed+1), so the exclusion boundary stops jittering
#                while bulk cells stay at the base n_mc. Unset => single tier,
#                shards byte-identical to the historical campaign.
#   MTIER / DATA_DIR / PASSES  as before
run_shard() {
    local pass="$1" il="$2"
    local t0 wall
    local extra=()
    # BCAP=none (or empty) => omit the flag entirely => uncapped cross section.
    case "${BCAP-0.1}" in
        none|None|NONE|"") ;;
        *) extra+=(--b-constrained-max "${BCAP-0.1}") ;;
    esac
    [ -n "${QMIN:-}" ] && extra+=(--q-min "$QMIN")
    [ -n "${KNL:-}" ] && extra+=(--projection-kernel "$KNL")
    [ -n "${NA:-}" ] && extra+=(--n-a "$NA")
    [ -n "${NMC_HI:-}" ] && extra+=(--n-mc-hi "$NMC_HI")
    t0=$(date +%s)
    echo "[$(ts)] SHARD_START pass=$pass il=$il workers=$WORKERS"
    if "$PY" scripts/build_release.py \
            --pass "$pass" \
            --il-start "$il" --il-end "$((il + 1))" \
            --shard-dir "$OUT/$pass" \
            --order tags-first \
            --lambda-set "${LSET:-full}" \
            --data-dir "${DATA_DIR:-notebooks}" \
            --m-tier "${MTIER:-119}" \
            "${extra[@]}" \
            --workers "$WORKERS"; then
        wall=$(( $(date +%s) - t0 ))
        echo "[$(ts)] SHARD_DONE pass=$pass il=$il wall=${wall}s"
    else
        echo "[$(ts)] SHARD_FAIL pass=$pass il=$il"
    fi
}

echo "[$(ts)] campaign start  host=$(hostname 2>/dev/null || echo "$REMOTE_HOST")  workers=$WORKERS  out=$OUT"

# noatm first (cheap, full product), then atm tags-first (tag/massless slices
# early for the V1 gate; remaining lambda smallest-first surfaces ODE blowups).
for pass in ${PASSES:-noatm atm}; do
    mkdir -p "$OUT/$pass"
    echo "[$(ts)] pass start: $pass"
    # il execution order comes straight from build_release (single source of
    # truth for the lambda axis + massless virtual index).
    ils=$("$PY" scripts/build_release.py --pass "$pass" --shard-dir "$OUT/$pass" \
              --order tags-first --lambda-set "${LSET:-full}" --print-order)
    if [ -z "$ils" ]; then
        echo "[$(ts)] SHARD_FAIL pass=$pass il=NONE (empty print-order)"
        continue
    fi
    for il in $ils; do
        run_shard "$pass" "$il"
    done
    echo "[$(ts)] PASS_DONE pass=$pass"
done

echo "[$(ts)] DONE"
