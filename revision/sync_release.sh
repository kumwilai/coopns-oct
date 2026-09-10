#!/usr/bin/env bash
# Re-sync the public release bundle from the working copies. Run this from the
# project root immediately before publishing, then inspect `git -C revision/release status`.
#
#   bash revision/sync_release.sh
#
# It copies the modules byte for byte, rewrites the data lists with paths relative
# to the bundle root, byte compiles everything, and greps the bundle for anything
# identifying. It never touches weights or images. Result files are copied only
# if COPY_RESULTS=1, because they must be the final ones.
set -eu
cd "$(dirname "$0")/.."
R=revision/release
PY="${PY:-/home/kumwilai/osmnx-env/bin/python}"
ROOT_MODULES="train_v8_cooperative.py neuro_symbolic_corrector_v8_cooperative.py neuro_symbolic_corrector_v8_enhanced.py neuro_symbolic_corrector_v8.py uncertainty_guided_correction.py clinical_enhancement_module.py cnr_preserving_correction.py validate_crossdataset.py eval_pku37_test.py run_duke17_loo.py"
REV_FILES="ablation_runner.py classical_baselines.py diagnostics.py fig_architecture.py fig_predicates.py fig_studies.py fig_subjective.py figstyle.py style_check.py select_config.py make_tables.py swinir_reference_check.py ALLOCATION_FIX.md"
for f in $ROOT_MODULES; do cp -p "$f" "$R/$f"; done
for b in nafnet dncnn swinir kbnet; do cp -p sota/models/${b}_7m.py $R/sota/models/; done
for f in $REV_FILES; do cp -p revision/$f $R/revision/$f; done
cp -p ALLOCATION_FIX.md $R/ALLOCATION_FIX.md 2>/dev/null || cp -p revision/ALLOCATION_FIX.md $R/ALLOCATION_FIX.md
$PY - <<'PYEOF'
import json, os
R="revision/release"; ROOT=os.getcwd()+"/"
for src in ["pku37_oct_dataset/pku37_real_train.jsonl","pku37_oct_dataset/pku37_real_val.jsonl",
            "pku37_oct_dataset/pku37_real_test.jsonl","revision/pku37_subset40.jsonl",
            "duke_sota_datasets/Duke17_Eval/duke17_eval.jsonl","duke_sota_datasets/Duke17_Eval/duke2013_synth_eval.jsonl"]:
    with open(src) as f, open(os.path.join(R,src),"w") as g:
        for line in f:
            if not line.strip(): continue
            e=json.loads(line)
            for k,v in e.items():
                if isinstance(v,str) and v.startswith(ROOT): e[k]=v[len(ROOT):]
                elif isinstance(v,str) and v.startswith("/"): raise SystemExit(f"absolute path outside project in {src}: {v}")
            g.write(json.dumps(e)+"\n")
print("data lists rewritten")
PYEOF
if [ "${COPY_RESULTS:-0}" = 1 ]; then
  cp -p outputs/revision/*.json $R/outputs/revision/ && echo "result files copied"
fi
$PY -m py_compile $(find $R -name '*.py' -not -path '*/.git/*') && echo "byte compile ok"
find $R -name __pycache__ -not -path '*/.git/*' -prune -exec rm -r {} +
echo "identity grep (must print nothing):"
grep -rnIE '/home/|kumwilai|research/oct|coopns-slr|github-coopns' --exclude-dir=.git $R | grep -vE '@torch|@staticmethod|@property' || true
echo "bundle files that differ from the working copies (must print nothing):"
for f in $ROOT_MODULES; do cmp -s "$f" "$R/$f" || echo "  $f"; done
for f in $REV_FILES; do cmp -s "revision/$f" "$R/revision/$f" || echo "  revision/$f"; done
echo "done. Now: git -C $R status, and remember the remote URL must not carry a user name."
