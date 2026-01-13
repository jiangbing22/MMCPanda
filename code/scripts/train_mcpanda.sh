#!/bin/bash

# 设置相关路径
DATA_PATH="./data/mosi/label.csv"          # MOSI CSV 路径 (请确认你的文件位置)
IMAGE_ROOT="./data/mosi/raw/video"         # MOSI 图片帧文件夹
AUDIO_ROOT="./data/mosi/raw/audio"         # MOSI 音频文件夹
SAVE_PATH="./checkpoints/mcpanda_mosi_finetune"

# 模型 Checkpoint 路径 (请修改为你本地的实际路径)
IMAGEBIND_PATH="./pretrained_ckpt/imagebind_huge.pth" 
VICUNA_PATH="./pretrained_ckpt/vicuna_ckpt/7b_v0"  # 或者 13b_v0
DELTA_PATH="./pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt"

# 训练参数
EPOCHS=5
BATCH_SIZE=4
LR=1e-4

mkdir -p $SAVE_PATH

# 运行训练
python code/train_MCPanda.py \
    --data_path "$DATA_PATH" \
    --image_root_path "$IMAGE_ROOT" \
    --audio_root_path "$AUDIO_ROOT" \
    --save_path "$SAVE_PATH" \
    --imagebind_ckpt_path "$IMAGEBIND_PATH" \
    --vicuna_ckpt_path "$VICUNA_PATH" \
    --delta_ckpt_path "$DELTA_PATH" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --max_tgt_len 128 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --gradient_accumulation_steps 2 \
    --gamma 1.5