#!/usr/bin/env bash
# Download the public benchmark datasets used by examples/validation.toml into datasets/
# (gitignored). Run once; it is safe to run again. The harness itself never downloads data.
#
#   scripts/fetch_datasets.sh
#
#   datasets/gsm8k/test.jsonl              GSM8K test split (1,319 problems)
#   datasets/humaneval/HumanEval.jsonl.gz  HumanEval (164 problems)
#   datasets/bfcl/mix.jsonl                BFCL v4: 20 simple, 10 parallel, 10 irrelevance
#   datasets/bfcl/mix_answers.jsonl        possible answers for those items
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/datasets"
bfcl="https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"
mkdir -p "$out/gsm8k" "$out/humaneval" "$out/bfcl/raw"

fetch() {  # url destination
  echo "  $2"
  curl -fsSL --retry 2 "$1" -o "$2.tmp" && mv "$2.tmp" "$2"
}

echo "Downloading into $out"
fetch https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
      "$out/gsm8k/test.jsonl"
fetch https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
      "$out/humaneval/HumanEval.jsonl.gz"
for name in simple_python parallel irrelevance; do
  fetch "$bfcl/BFCL_v4_$name.json" "$out/bfcl/raw/BFCL_v4_$name.json"
  [ "$name" = irrelevance ] || fetch "$bfcl/possible_answer/BFCL_v4_$name.json" \
      "$out/bfcl/raw/answers_$name.json"
done

# One BFCL file with a fixed mix of categories (first N of each, upstream order).
python3 - "$out/bfcl" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
take = {"simple_python": 20, "parallel": 10, "irrelevance": 10}
questions, answers = [], []
for name, count in take.items():
    rows = [json.loads(line) for line in (base / "raw" / f"BFCL_v4_{name}.json").read_text().splitlines() if line.strip()]
    questions += rows[:count]
    if name != "irrelevance":
        wanted = {row["id"] for row in rows[:count]}
        answers += [json.loads(line) for line in (base / "raw" / f"answers_{name}.json").read_text().splitlines()
                    if line.strip() and json.loads(line)["id"] in wanted]
(base / "mix.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
(base / "mix_answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
print(f"  {base / 'mix.jsonl'}: {len(questions)} items, {len(answers)} answers")
PY
echo "Done. scripts/validate.sh now runs the benchmarks on these files."
