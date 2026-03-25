#!/bin/bash
# Depth Recurrence Training — Aalind Tiwari
# Run on 8×H100 on RunPod

set -e

export RECUR_LAYERS="4,5"
export TTT_ENABLED=1
export TTT_UNTIE=0

# Hyperparameters (from PR#549 + PR#686)
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3.0
export BIGRAM_VOCAB_SIZE=1536
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export LATE_QAT=1
export LATE_QAT_THRESHOLD=0.15
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0
export MUON_WD=0.04
export ADAM_WD=0.04
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3500
export ITERATIONS=9000
export MAX_WALLCLOCK_SECONDS=600
export EVAL_STRIDE=64
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EMA_ENABLED=1
export EMA_DECAY=0.997
export SWA_ENABLED=1
export SWA_EVERY=50
export GRAD_CLIP_NORM=0.3
export VAL_BATCH_SIZE=524288
export VOCAB_SIZE=1024

SEED=${1:-1337}
export SEED

echo "=== Depth Recurrence Training ==="
echo "Seed: $SEED"
echo "RECUR_LAYERS: $RECUR_LAYERS"
echo "TTT: enabled"
echo ""

DATA_PATH=${DATA_PATH:-"./data/datasets/fineweb10B_sp1024/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"./data/tokenizers/fineweb_1024_bpe.model"}

torchrun --standalone --nproc_per_node=8 train_gpt.py \
  --data_path $DATA_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --run_id "aalind_recur_${SEED}"
