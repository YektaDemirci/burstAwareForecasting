#!/usr/bin/env bash
# Trains and tests the final comparison: vanilla Informer and the proposed model, on the paper's self-similar series and
# three fresh realizations of its generator (self-similar 2-4), at H=1 and H=12 with seeds 0-3. Run from the repo root:
#   nohup bash final_study/run_final.sh > final_study/run.log 2>&1 &
# Finished runs are skipped, so it is safe to re-run after an interruption. A failed run is listed at the end and does
# not stop the others. Scoring runs at the end, on whatever finished. Results: final_study/results/results_<PREFIX>.txt
# Optional settings: DATASETS, HORIZONS, SEEDS, MODELS, BATCH, ACCUM, EPOCHS, PATIENCE, PREFIX, CKPT (checkpoint folder),
# SCORE=0 (skip scoring). For example, H=1 only: HORIZONS=1 bash final_study/run_final.sh
# Patience 10 with 10 epochs means no run stops early. The best-validation checkpoint is still the one kept.
DATASETS=${DATASETS:-"paper ss2 ss3 ss4"}
HORIZONS=${HORIZONS:-"1 12"}
SEEDS=${SEEDS:-"0 1 2 3"}
MODELS=${MODELS:-"baseline proposed"}
BATCH=${BATCH:-64}
ACCUM=${ACCUM:-2}
EPOCHS=${EPOCHS:-10}
PATIENCE=${PATIENCE:-10}
PREFIX=${PREFIX:-fin}
CKPT=${CKPT:-./checkpoints/}
LOGS=final_study/logs
mkdir -p "$LOGS" final_study/results
failed=()

# paper: the series used in the paper, with its burst list. ss2-4: data/fresh/self_similar_<n>.csv (simulator seeds 11, 12, 18).
data_flags() {
    case "$1" in
        paper) echo "--root_path ./data/ETT/ --data_path 1.04_750_10ms.csv --peak_path bursts177.csv --freq 10L" ;;
        ss*)   echo "--root_path ./data/fresh/ --data_path self_similar_${1#ss}.csv --peak_path self_similar_${1#ss}_peaks.csv --freq 10L" ;;
        *)     echo "unknown dataset $1" >&2; return 1 ;;
    esac
}

model_flags() {
    case "$1" in
        proposed) echo "--model_mode proposed --peak_feature causal --peak_target causal --pos_weight 5" ;;
        baseline) echo "--model_mode baseline --peak_feature none" ;;
    esac
}

for h in $HORIZONS; do
    for ds in $DATASETS; do
        flags=$(data_flags "$ds") || { failed+=("$ds: unknown dataset"); continue; }
        for seed in $SEEDS; do
            for m in $MODELS; do
                parr="${PREFIX}_${ds}_${m}_s${seed}"; log="$LOGS/${parr}_h${h}"
                if [ -f "$log.done" ]; then echo "skip  $parr H=$h"; continue; fi
                echo "start $parr H=$h  $(date '+%a %H:%M:%S')"
                if python -u main_informer.py --model informer --data custom $flags --features S \
                    --seq_len 128 --label_len 64 --pred_len "$h" --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1 \
                    --batch_size "$BATCH" --grad_accum "$ACCUM" --train_epochs "$EPOCHS" --patience "$PATIENCE" --seed "$seed" --parr "$parr" \
                    --checkpoints "$CKPT" $(model_flags "$m") > "$log.log" 2>&1; then
                    touch "$log.done"; echo "done  $parr H=$h  $(date '+%a %H:%M:%S')"
                else
                    echo "FAILED $parr H=$h, see $log.log"; failed+=("$parr H=$h")
                fi
            done
        done
    done
done

if [ "${SCORE:-1}" != 0 ]; then
    python -u final_study/evaluate_final.py --prefix "$PREFIX" --ckpt_root "$CKPT" | tee "final_study/results/results_${PREFIX}.txt"
    [ ${PIPESTATUS[0]} -eq 0 ] || failed+=("scoring")
fi
[ ${#failed[@]} -eq 0 ] && echo "All runs finished." || { echo "Failed:"; printf '  %s\n' "${failed[@]}"; exit 1; }
