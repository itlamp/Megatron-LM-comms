#!/bin/bash
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_file="tp_8_experiments/TP_8_experiment_log_${timestamp}.txt"
touch "$log_file"
alias ssh='ssh -F ~/.ssh/config'

export HL_LAUNCHER_TYPE=mpirun  
export HL_DATA_DIR_ROOT= # Set 
export HL_DATA_CACHE_DIR= # Set
export HL_TOKENIZER_MODEL= # Set
export HL_TRANSFORMER_IMPL=local
export HL_LOG_INTERVAL=10
export HL_SAVE_INTERVAL=1000
export HL_CKP_ACT=0  
export HL_DEVICES_PER_NODE=8
export HL_LLAMA_VER=2 
export HL_LLAMA_MODEL_SIZE=7
export HL_DROPOUT=0  
export HL_SEQ_PARALLEL=0
export HL_NUM_WORKERS=32
export HL_USE_FUSED_RMSNORM=1 
export HL_OPTIMIZER=fusedadamw  
export HL_EVAL_ITERS=100
export HL_EVAL_INTERVAL=1000
export HL_TRAIN_ITERS=500000
export HL_KILL_SWITCH_FILE=$session_data_kill_switch_path  
export HL_RESULTS_DIR=$session_data_logs_path  
export HL_TENSORBOARD_DIR=$session_data_tensorboard_path  
export HL_NUM_NODES=1
export HL_MICRO_BATCH=8
export HL_CP=1
export HL_PP=1
export HL_DP=1
export HL_TP=8
export HL_ASYNCH=0.5

bash examples/llama/pretrain_llama.sh