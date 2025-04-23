#!/bin/bash

# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
 EXP_NAME=Baseline_Orig_Repo

set -ex

# Distributed training variables
LAUNCHER_TYPE=${HL_LAUNCHER_TYPE:-mpirun}
#EDITTTTT TODO
DATA_DIR=${HL_DATA_DIR_ROOT:-/software/data/datasets/c4_full_mlperf_12012023/preprocessed_c4_spm}
DATA_CACHE_DIR=${HL_DATA_CACHE_DIR:-}
# DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-redpajama}
#EDITTTTT TODO
TOKENIZER_MODEL=${HL_TOKENIZER_MODEL:-}
TRANSFORMER_IMPL=${HL_TRANSFORMER_IMPL:-local}
# Parallelism variables
NUM_NODES=${HL_NUM_NODES:-1}
ASYNCH=${HL_ASYNCH:-1} 
DP=${HL_DP:-1}
TP=${HL_TP:-8}
PP=${HL_PP:-1}
MICRO_BATCH_SIZE=${HL_MICRO_BATCH:-16} # batch_size
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
OUTPUT_DIR_PREFIX=${HL_RESULTS_DIR_PREFIX:-.}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
DIST_CKPT_FORMAT=${HL_DIST_CKPT_FORMAT:-torch_dist}
USE_DISTRIBUTED_OPTIMIZER=${HL_USE_DISTRIBUTED_OPTIMIZER:-1}
USE_DIST_CKPT=${HL_USE_DIST_CKPT:-1}
LOAD_DIR=${HL_LOAD_DIR:-}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
VERIFY_CKPT=${HL_VERIFY_CKPT:-0}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH_FILE:-}
HOSTSFILE=${HL_HOSTSFILE:-}
CKP_ACT=${HL_CKP_ACT:-0}
RECOMPUTE_NUM_LAYERS=${HL_RECOMPUTE_NUM_LAYERS:-1}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
DEVICES_PER_NODE=${HL_DEVICES_PER_NODE:-8}
SEQ_PARALLEL=${HL_SEQ_PARALLEL:-0}
OPTIMIZER=${HL_OPTIMIZER:-fusedadamw}
DROPOUT=${HL_DROPOUT:-0.0}
EVAL_ITERS=${HL_EVAL_ITERS:-100}
EVAL_INTERVAL=${HL_EVAL_INTERVAL:-1000}
USE_FUSED_SDPA=${HL_USE_FUSED_SDPA:-1}
TRAIN_ITERS=${HL_TRAIN_ITERS:-600000}
USE_FUSED_SDPA_WITH_RECOMPUTE=${HL_USE_FUSED_SDPA_WITH_RECOMPUTE:-0}
USE_FAST_SOFTMAX=${HL_USE_FAST_SOFTMAX:-1}
USE_FUSED_RMSNORM=${HL_USE_FUSED_RMSNORM:-1}
PROFILE_TYPE=${HL_PROFILE_TYPE:-} # provide either of pt, pt-full, hltv
PROFILE_STEP_START=${HL_PROFILE_STEP_START:-3}
PROFILE_STEP_END=${HL_PROFILE_STEP_END:-4}
PROFILE_RANKS=${HL_PROFILE_RANKS:-"0"} # "0 1 4 7"
REDIRECT_LOGS=${HL_REDIRECT_LOGS:-0}
DETERMINISTIC_MODE=${HL_DETERMINISTIC_MODE:-1}
FP8=${HL_FP8:-0}
FP8_FORMAT=${HL_FP8_FORMAT:-hybrid} # hybrid or e5m2
FP8_MARGIN=${HL_FP8_MARGIN:-0}
FP8_AMAX_COMPUTE_ALGO=${HL_FP8_AMAX_COMPUTE_ALGO:-max} # max or most_recent
USE_TORCH_COMPILE=${HL_USE_TORCH_COMPILE:-0}
USE_TORCH_COMPILED_AUTOGRAD=${HL_USE_TORCH_COMPILED_AUTOGRAD:-0}
USE_LAZY_MODE=${HL_USE_LAZY_MODE:-1}
SKIP_TRAIN=${HL_SKIP_TRAIN:-0}
NUM_WORKERS=${HL_NUM_WORKERS:-2}
NUM_LAYERS=24 # must be divisible by PP
HIDDEN_SIZE=1536
NHEADS=16 # must be divisible by TP
FFN_HIDDEN_SIZE=$(($HIDDEN_SIZE * 4))
MAX_SEQ_LEN=2048
GLOBAL_BATCH_SIZE=256
LR=2.5e-4
MIN_LR=2.5e-5
TOKENIZER_TYPE=GPT2BPETokenizer

FP8_COVERAGE=${HL_FP8_COVERAGE:-"mlp_row_parallel=False attention=False"}
TRAIN_DATA_PATH="0.125 ${DATA_DIR}/c4_en_0_c4_spm_text_document \
                 0.125 ${DATA_DIR}/c4_en_1_c4_spm_text_document \
                 0.125 ${DATA_DIR}/c4_en_2_c4_spm_text_document \
                 0.125 ${DATA_DIR}/c4_en_3_c4_spm_text_document \
                 0.125 ${DATA_DIR}/c4_en_4_c4_spm_text_document \
                 0.125 ${DATA_DIR}/c4_en_5_c4_spm_text_document \
                 0.125 ${DATA_DIR}/c4_en_6_c4_spm_text_document \
                 0.125 ${DATA_DIR}/c4_en_7_c4_spm_text_document"

VALID_DATA_PATH="${DATA_DIR}/c4_en_validation_c4_spm_text_document"

if [[ -z "${MEGATRON_LM_ROOT}" ]]; then
    MEGATRON_LM_ROOT=$(realpath "$(dirname "$0")"/../../)
fi

if [[ $((NUM_NODES*DEVICES_PER_NODE)) -ne $((DP*TP*PP)) ]]; then
    echo "NUM_NODES*DEVICES_PER_NODE != DP*TP*PP"
    exit 1
fi

if [[ "${TRANSFORMER_IMPL}" = "local" && "${FP8}" -eq 1 ]]; then
    echo "fp8 is not supported with local transformer implementation"
    exit 1
fi

if [[ $(( NUM_LAYERS % PP )) -ne 0 ]]; then
    echo 'HL_NUM_LAYERS must be divisible by PP'
    exit 1
fi

# Paths
SRC_PATH="${MEGATRON_LM_ROOT}/pretrain_gpt.py"

if [[ -z "${TOKENIZER_MODEL}" ]]; then
    TOKENIZER_MODEL="${DATA_DIR}/tokenizer.model"
fi

NUM_DEVICES=$((DEVICES_PER_NODE*NUM_NODES))

RUNTIME=$(date +"%Y%m%d_%H%M")
# Experiment name
if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="default"
fi
# output paths
if [[ -z "${OUTPUT_DIR}" ]]; then
    data_type="bf16"
    if [[ "${FP8}" -eq 1 ]]; then
        data_type="fp8"
    fi
    OUTPUT_DIR=${OUTPUT_DIR_PREFIX}/out/gpt3_1.3b/${data_type}_${TRANSFORMER_IMPL}_${EXP_NAME}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_ffn${FFN_HIDDEN_SIZE}_gb${GLOBAL_BATCH_SIZE}_mb${MICRO_BATCH_SIZE}_sp${SEQ_PARALLEL}_D${DP}_T${TP}_P${PP}_devices${NUM_DEVICES}_${RUNTIME}
fi
if [[ -z "${CHECKPOINTS_DIR}" ]]; then
    CHECKPOINTS_DIR=${OUTPUT_DIR}/checkpoints
fi
if [[ -z "${LOAD_DIR}" ]]; then
    LOAD_DIR=${CHECKPOINTS_DIR}
fi

if [[ -z "${TENSORBOARD_DIR}" ]]; then
    TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
fi
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINTS_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

if [[ "${NUM_NODES}" -ne "1" ]] && [[ -z "${HOSTSFILE}" ]]; then
    HOSTSFILE=${MEGATRON_LM_ROOT}/examples/hostsfile
    if [[ -f "${HOSTSFILE}" ]]; then
        cat /dev/null > "${HOSTSFILE}"
    fi
    etc_mpi_hostfile="/etc/mpi/hostfile"
    if [[ ! -f ${etc_mpi_hostfile} ]]; then
        echo "${etc_mpi_hostfile} not available, set HL_HOSTSFILE"
        exit 1
    fi
    cat ${etc_mpi_hostfile} | xargs -I{} echo {} slots=8 >> "${HOSTSFILE}"
fi

# Setting the environment variables
PT_HPU_GPU_MIGRATION=1

# Set training command
CMD=""
if [[ "${LAUNCHER_TYPE}" = "mpirun" ]]; then
    CMD="${CMD} mpirun"
    CMD="${CMD} --allow-run-as-root"
    CMD="${CMD} -n ${NUM_DEVICES}"
    CMD="${CMD} --bind-to none"
    CMD="${CMD} -x PT_HPU_GPU_MIGRATION=${PT_HPU_GPU_MIGRATION}"
    if [[ "${NUM_NODES}" -ne "1" ]]; then
        CMD="${CMD} -hostfile ${HOSTSFILE}"
        CMD="${CMD} -x MASTER_ADDR=$(head -n 1 "${HOSTSFILE}" | sed -n s/[[:space:]]slots.*//p)"
    else
        CMD="${CMD} -x MASTER_ADDR=localhost"
    fi
    CMD="${CMD} -x MASTER_PORT=12345"
elif [[ "${LAUNCHER_TYPE}" = "torchrun" ]]; then
    if [[ "${NUM_NODES}" -ne "1" ]]; then
        echo "NUM_NODES greater than 1 not supported by torchrun"
        exit 1
    fi
    export PT_HPU_GPU_MIGRATION=${PT_HPU_GPU_MIGRATION}
    CMD="${CMD} torchrun"
    CMD="${CMD} --nnodes ${NUM_NODES}"
    CMD="${CMD} --nproc-per-node ${DEVICES_PER_NODE}"
    CMD="${CMD} --no-python"
    CMD="${CMD} --node-rank 0"
    CMD="${CMD} --master-addr localhost"
    CMD="${CMD} --master-port 12345"
else
    echo "Unsupported launcher type = ${LAUNCHER_TYPE}"
    exit 2
fi

if [ "$USE_LAZY_MODE" = "0" ]; then
    CMD="${CMD} -x PT_HPU_LAZY_MODE=0"
fi

CMD="${CMD} \
    python ${SRC_PATH} \
    --transformer-impl ${TRANSFORMER_IMPL} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --distributed-backend nccl \
    --seq-length ${MAX_SEQ_LEN} \
    --max-position-embeddings ${MAX_SEQ_LEN} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NHEADS} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --attention-dropout ${DROPOUT} \
    --hidden-dropout ${DROPOUT} \
    --weight-decay 1e-1 \
    --init-method-std 0.015
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr ${LR} \
    --lr-decay-style cosine \
    --lr-warmup-fraction .001 \
    --min-lr ${MIN_LR} \
    --use-torch-compile=${USE_TORCH_COMPILE} \
    --use-torch-compiled-autograd=${USE_TORCH_COMPILED_AUTOGRAD} \
    --use-fused-sdpa-with-recompute ${USE_FUSED_SDPA_WITH_RECOMPUTE} \
    --use-fused-sdpa ${USE_FUSED_SDPA} \
    --use-fused-rmsnorm ${USE_FUSED_RMSNORM} \
    --use-fast-softmax ${USE_FAST_SOFTMAX} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --log-interval ${LOG_INTERVAL} \
    --log-throughput \
    --optimizer ${OPTIMIZER} \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --use-mcore-models \
    --exit-interval ${EXIT_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --load ${LOAD_DIR} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --train-data-path ${TRAIN_DATA_PATH} \
    --valid-data-path ${VALID_DATA_PATH} \
    --num-workers ${NUM_WORKERS} \
    --asynch_p ${ASYNCH} \
    --bf16
    "

if [[ "${SEQ_PARALLEL}" -eq 1 ]]; then
    CMD="${CMD} --sequence-parallel"
fi

if [[ "${LLAMA_VER}" = "2" ]]; then
     CMD="${CMD} --no-check-for-nan-in-loss-and-grad"
fi

if [[ "${USE_FUSED_SDPA}" = "1" || "${USE_FUSED_SDPA_WITH_RECOMPUTE}" = "1" ]]; then
    CMD="${CMD} --no-create-attention-mask-in-dataloader"
fi

if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
    CMD="${CMD} --skip-train"
fi

if [[ "${CKP_ACT}" -eq 1 ]]; then
    CMD="${CMD} --recompute-granularity=full"
    CMD="${CMD} --recompute-method uniform"
    CMD="${CMD} --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
elif [[ "${CKP_ACT}" -eq 2 ]]; then
    CMD="${CMD} --recompute-granularity selective"
elif [[ "${CKP_ACT}" -ne 0 ]]; then
    echo "incorrect HL_CKP_ACT=${CKP_ACT} is set"
    exit 1
fi

if [[ "${USE_DISTRIBUTED_OPTIMIZER}" -eq 1 ]]; then
    CMD="${CMD} --use-distributed-optimizer"
fi

if [[ "${DETERMINISTIC_MODE}" -eq 1 ]]; then
    CMD="${CMD} --deterministic-mode"
fi

# fp8 args
if [[ "${TRANSFORMER_IMPL}" = "transformer_engine" && "${FP8}" -eq 1 ]]; then

    FP8_MEASURE_INTERVAL=${HL_FP8_MEASURE_INTERVAL:-$(( GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE / DP ))}
    FP8_AMAX_HISTORY_LEN=${HL_FP8_AMAX_HISTORY_LEN:-$((( GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE / DP + 15 ) / 16 ))}
    FP8_AMAX_REDUCE=${HL_FP8_AMAX_REDUCE:-1}

    CMD="${CMD} --fp8-interval ${FP8_MEASURE_INTERVAL}"
    CMD="${CMD} --fp8-margin ${FP8_MARGIN}"
    CMD="${CMD} --fp8-amax-compute-algo ${FP8_AMAX_COMPUTE_ALGO}"
    CMD="${CMD} --fp8-amax-history-len ${FP8_AMAX_HISTORY_LEN}"
    CMD="${CMD} --fp8-format ${FP8_FORMAT}"
    CMD="${CMD} --fp8-coverage ${FP8_COVERAGE}"

    if [[ "${FP8_AMAX_REDUCE}" -eq 1 ]]; then
        CMD="${CMD} --fp8-amax-reduce"
    fi
fi

# handle kill switch file argument
if [[ -n "${KILL_SWITCH_FILE}" ]]; then
    CMD="${CMD} --kill-switch-file ${KILL_SWITCH_FILE}"
fi

if [[ -n "${PROFILE_TYPE}" ]]; then
    CMD="${CMD} --profile-type ${PROFILE_TYPE}"
    CMD="${CMD} --profile-step-start ${PROFILE_STEP_START}"
    CMD="${CMD} --profile-step-end ${PROFILE_STEP_END}"
    CMD="${CMD} --profile-ranks ${PROFILE_RANKS}"
fi

if [[ "${CHECKPOINT_SAVE}" -eq 1 ]]; then
    CMD="${CMD} --save ${CHECKPOINTS_DIR}"
    CMD="${CMD} --save-interval ${SAVE_INTERVAL}"
    CMD="${CMD} --dist-ckpt-format ${DIST_CKPT_FORMAT}"
    if [[ "${USE_DIST_CKPT}" -eq 1 ]]; then
        CMD="${CMD} --use-dist-ckpt"
    fi
    if [[ "${VERIFY_CKPT}" -eq 1 ]]; then
        CMD="${CMD} --verify-checkpoint"
        CMD="${CMD} --verify-checkpoint-model-type LLAMA"
    fi
fi

if [[ "${TOKENIZER_TYPE}" = "GPTSentencePieceTokenizer" || "${TOKENIZER_TYPE}" = "Llama3Tokenizer" || "${TOKENIZER_TYPE}" = "HuggingFaceTokenizer" ]]; then
    CMD="${CMD} --tokenizer-type ${TOKENIZER_TYPE}"
    CMD="${CMD} --tokenizer-model ${TOKENIZER_MODEL}"
elif [[ "${TOKENIZER_TYPE}" = "GPT2BPETokenizer" ]]; then
    CMD="${CMD} --tokenizer-type GPT2BPETokenizer"
    CMD="${CMD} --vocab-file ${DATA_DIR}/vocab.json"
    CMD="${CMD} --merge-file ${DATA_DIR}/merges.txt"
else
    echo "incorrect HL_TOKENIZER_TYPE=${TOKENIZER_TYPE} is set"
    exit 1
fi

if [[ -n "${DATA_CACHE_DIR}" ]]; then
    CMD="${CMD} --data-cache-path ${DATA_CACHE_DIR}"
fi

if [[ "${REDIRECT_LOGS}" -eq 1 ]]; then
    ${CMD} 2>&1 | tee "${OUTPUT_DIR}"/log_"${EXP_NAME}"_"${RUNTIME}".txt
else
    ${CMD}
fi
