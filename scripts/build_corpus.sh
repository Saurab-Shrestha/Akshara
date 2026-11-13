#!/usr/bin/env bash
# End-to-end v2 corpus build: corpus -> tokenizer -> packed .bin -> private HF dataset.
#
#   nohup caffeinate -is scripts/build_corpus.sh > logs/build_v5.log 2>&1 &
#
# Env overrides: TOTAL_CHARS (default 12e9 ≈ 3.4B tokens), NEPALI_SHARE (0.20),
# OUT (data/corpus_v5), HF_REPO (Saurab0/akshara-corpus-v5), SAMPLE_BYTES (1e9:
# tokenizer sample, sized for a 16 GB machine).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TOTAL_CHARS=${TOTAL_CHARS:-12e9}
NEPALI_SHARE=${NEPALI_SHARE:-0.20}
OUT=${OUT:-data/corpus_v5}
HF_REPO=${HF_REPO:-Saurab0/akshara-corpus-v5}
SAMPLE_BYTES=${SAMPLE_BYTES:-1e9}
PY=venv/bin/python
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "1/4 corpus ($TOTAL_CHARS chars) -> $OUT"
# HF streams can hang forever in an SSL read. prepare_data.py resumes per source,
# so on a stall (no file under $OUT written for STALL_MIN minutes) kill and rerun.
STALL_MIN=${STALL_MIN:-15}
for attempt in $(seq 1 30); do
    $PY -u scripts/prepare_data.py --total_chars "$TOTAL_CHARS" --out_dir "$OUT" --nepali_share "$NEPALI_SHARE" &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
        sleep 60
        if [[ -z $(find "$OUT" -type f -mmin -"$STALL_MIN" 2>/dev/null | head -1) ]]; then
            echo "[watchdog] no output for ${STALL_MIN} min, restarting (attempt $attempt)"
            kill $pid; sleep 5; kill -9 $pid 2>/dev/null || true
        fi
    done
    if wait $pid; then break; fi
    [[ $attempt == 30 ]] && { echo "[watchdog] giving up"; exit 1; }
done

# The tokenizer is committed and every checkpoint depends on its ids, so it is
# only retrained when missing or when RETRAIN_TOKENIZER=1.
if [[ ! -f tokenizer/akshara-bpe/tokenizer.json || ${RETRAIN_TOKENIZER:-0} == 1 ]]; then
    step "2/4 tokenizer (48k) -> tokenizer/akshara-bpe"
    $PY -u scripts/train_tokenizer.py --input "$OUT/train.jsonl" --vocab_size 49152 \
        --sample_bytes "$SAMPLE_BYTES" --out_dir tokenizer/akshara-bpe --compare Qwen/Qwen3.5-0.8B
else
    step "2/4 tokenizer: reusing tokenizer/akshara-bpe"
fi

step "3/4 pack -> $OUT/{train,val}.bin"
$PY -u scripts/tokenize_corpus.py --input "$OUT/val.jsonl" --tokenizer tokenizer/akshara-bpe
$PY -u scripts/tokenize_corpus.py --input "$OUT/train.jsonl" --tokenizer tokenizer/akshara-bpe

step "4/4 upload -> $HF_REPO (private)"
$PY -u - "$HF_REPO" "$OUT" <<'EOF'
import sys
from huggingface_hub import HfApi
repo, out = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for local, remote in [(f"{out}/train.bin", "train.bin"), (f"{out}/val.bin", "val.bin"),
                      (f"{out}/val.jsonl", "val.jsonl")]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type="dataset",
                    commit_message=f"add {remote}")
    print("uploaded", remote)
api.upload_folder(folder_path="tokenizer/akshara-bpe", path_in_repo="tokenizer", repo_id=repo,
                  repo_type="dataset", commit_message="add tokenizer")
print("uploaded tokenizer/")
EOF

step "done"
