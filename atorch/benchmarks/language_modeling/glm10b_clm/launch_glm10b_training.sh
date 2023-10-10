#!/bin/bash
pip install -U -r requirements.txt
DATASET_DIR=/path/to/wikitext-2-raw-v1
PRETRAINED_MODEL_DIR=/path/to/glm-10b

NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS=$((NUM_GPUS_PER_NODE * WORLD_SIZE))
PER_DEVICE_TRAIN_BATCH_SIZE=21
TOTAL_TRAIN_BATCH_SIZE=$((NUM_GPUS_PER_NODE * WORLD_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE))

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_GID_INDEX=3
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

python -m atorch.distributed.launch \
    --nnodes="$WORLD_SIZE" \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    glm10b_clm.py \
    --model_name_or_path $PRETRAINED_MODEL_DIR \
    --dataset_path $DATASET_DIR \
    --num_train_epochs 3 \
    --block_size 512 \
    --total_train_batch_size $TOTAL_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --seed 42 \
    --preprocessing_num_workers 6 \
    --dataloader_num_workers 0 \
    --output_dir /tmp/test-clm \
    --trust_remote_code \
    --ignore_mismatched_sizes \
    --skip_atorch_autoacc_dryrun \
    2>&1 | tee log_glm10b_"${WORLD_SIZE}"n"${NUM_GPUS}"g.txt 
