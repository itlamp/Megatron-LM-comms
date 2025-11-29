# LLM training with Partial Activation Syncronization

This directory provides training scripts for the paper [Tensor-Parallelism with Partially Synchronized Activations](https://arxiv.org/abs/2506.19645), and enables training CAAT-Net models using partial activation synchronization on Intel® Gaudi® 2 & Gaudi® 3 AI accelerators. This implementation is based on https://github.com/NVIDIA/Megatron-LM, and is a fork of Intel's https://github.com/HabanaAI/Megatron-LM. 


## Setup
Please follow the instructions provided in the [Intel Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the `$PYTHON` environment variable. 

This repository is meant to be executed on the Intel Gaudi 1.22.0 software release. 

## Prerequisites
When creating Docker container, set the shared memory size as 10 GB through the Docker run command:
  ```bash
  --shm-size=10g
  ```

## Clone Intel Gaudi Megatron-LM
In the docker container corresponding to the 1.22.0 software version, clone this repository. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.

```bash
git clone https://github.com/itlamp/Megatron-LM-comms.git
```
Set the required environment variables as shown below:
```
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PYTHONPATH=$MEGATRON_LM_ROOT:$PYTHONPATH
```
## Install Megatron-LM Requirements
* In the Docker container, go to the Megatron-LM directory:
  ```bash
  cd $MEGATRON_LM_ROOT
  ```

* Install the required packages using pip:
  ```bash
  pip install -r megatron/core/requirements.txt
  
  # For 1.20.1 release, also run
  pip install -r megatron/core/requirements_1.20.txt
  ```

## Training Scripts

Attached are commands to launch training of the 130M, 1B and 7B parameter Llama-based models detailed in the paper. 

### Running Llama models in Megatron-LM

First, see examples/llama/README.md for more information on prerequisites for running Llama on Gaudi. These are identical to the requirements in the upstream [HabanaAI/Megatron-LM](https://github.com/HabanaAI/Megatron-LM) repository.

### Configuring CAAT-Net
The files examples/llama/run_llama_<model_size>.sh set the arguments necessary for training. Synchronization factor $p$ is controlled by the environment variable HL_ASYNCH, and the tensor-parallel dimension is controled by HL_TP. The defaults are $p$=0.5 and tensor-parallel dimension 8.

Furthermore, one should set the dataset and tokenizer path inside these files before execution.

### 130M model
```
cd $MEGATRON_LM_ROOT
bash examples/llama/run_llama_130M.sh
```

### 1B model
```
cd $MEGATRON_LM_ROOT
bash examples/llama/run_llama_1B.sh
```

### 7B model
```
cd $MEGATRON_LM_ROOT
bash examples/llama/run_llama_7B.sh
```

For speedup measurments, switch to the speedup branch (Currently not available, will be published shortly):

``` 
git checkout speedup
```

## Megatron-LM
First introduced in 2019, Megatron ([1](https://arxiv.org/pdf/1909.08053), [2](https://arxiv.org/pdf/2104.04473), and [3](https://arxiv.org/pdf/2205.05198)) sparked a wave of innovation in the AI community, enabling researchers and developers to utilize the underpinnings of this library to further LLM advancements.
