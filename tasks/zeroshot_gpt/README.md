# ZeroShot llama 2 & 3.1 wikitext and lambada Evaluation

## pre-requisites

```bash
cd /path/to/Megatron-LM/
pip install -r examples/llama/requirements.txt
```

## [LAMBADA] ZeroShot llama 3.1 evaluation

```bash
TASK="LAMBADA"
export MEGATRON_LM_ROOT=/path/to/Megatron-LM/
export PYTHONPATH=$MEGATRON_LM_ROOT:$PYTHONPATH

export MASTER_ADDR=localhost
export MASTER_PORT=12345


VALID_DATA=/path/to/lambada_test.jsonl
CHECKPOINT_PATH=/path/to/mlm/llama3.1/checkpoint/

# params needs to be specified if checkpoint doesnt already has it.

# Common task args for llama 3.1
COMMON_TASK_ARGS="--seq-length 8192 \
                  --max-position-embeddings 8192 \
                  --bf16"

# torchrun for llama 3.1
PT_HPU_GPU_MIGRATION=1 torchrun --nproc-per-node=8 $MEGATRON_LM_ROOT/tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model /path/to/llama3/tokenizer/ \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 4 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng \
       --overlapping-eval 1024 \
       --attention-softmax-in-fp32 \
       --ckpt-format torch \
       --use-checkpoint-args \
       --use-mp-args-from-checkpoint-args
```

## [LAMBADA] ZeroShot llama 2  evaluation

```bash

TASK="LAMBADA"
export MEGATRON_LM_ROOT=/path/to/Megatron-LM/
export PYTHONPATH=$MEGATRON_LM_ROOT:$PYTHONPATH

export MASTER_ADDR=localhost
export MASTER_PORT=12345


VALID_DATA=/path/to/lambada_test.jsonl
CHECKPOINT_PATH=/path/to/mlm/llama2/checkpoint/

# params needs to be specified if checkpoint doesnt already has it.
# Common task args for llama 2
COMMON_TASK_ARGS="--seq-length 4096 \
                  --bf16"

# torchrun for llama 2
PT_HPU_GPU_MIGRATION=1 torchrun --nproc-per-node=8 $MEGATRON_LM_ROOT/tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model /path/to/tokenizer.model \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 4 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng \
       --overlapping-eval 1024 \
       --attention-softmax-in-fp32 \
       --ckpt-format torch \
       --use-checkpoint-args \
       --use-mp-args-from-checkpoint-args

```

## [wikitext] ZeroShot llama 3.1 evaluation

```bash

TASK="WIKITEXT103"
export MEGATRON_LM_ROOT=/path/to/Megatron-LM/
export PYTHONPATH=$MEGATRON_LM_ROOT:$PYTHONPATH

export MASTER_ADDR=localhost
export MASTER_PORT=12345

VALID_DATA=/path/to/wiki.valid.tokens
CHECKPOINT_PATH=/path/to/llama3/checkpoint/

COMMON_TASK_ARGS="--seq-length 8192 \
                  --max-position-embeddings 8192 \
                  --bf16"

PT_HPU_GPU_MIGRATION=1 torchrun --nproc-per-node=8 $MEGATRON_LM_ROOT/tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model /path/to/llama3/tokenizer/ \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 4 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng \
       --overlapping-eval 1024 \
       --attention-softmax-in-fp32 \
       --ckpt-format torch \
       --use-checkpoint-args \
       --use-mp-args-from-checkpoint-args

```

## [wikitext] ZeroShot llama 2 evaluation

```bash

TASK="WIKITEXT103"
export MEGATRON_LM_ROOT=/path/to/Megatron-LM/
export PYTHONPATH=$MEGATRON_LM_ROOT:$PYTHONPATH

export MASTER_ADDR=localhost
export MASTER_PORT=12345

VALID_DATA=/path/to/wiki.valid.tokens
CHECKPOINT_PATH=/path/to/llama2/checkpoint/

COMMON_TASK_ARGS="--seq-length 4096 \
                  --bf16"

PT_HPU_GPU_MIGRATION=1 torchrun --nproc-per-node=8 $MEGATRON_LM_ROOT/tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model /path/to/tokenizer.model \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 4 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng \
       --overlapping-eval 1024 \
       --attention-softmax-in-fp32 \
       --ckpt-format torch \
       --use-checkpoint-args \
       --use-mp-args-from-checkpoint-args

```