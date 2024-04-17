#!/usr/bin/bash
OUTPUT_DIR='./logs_modified_attention'
DATA_PATH='./data/UCF-101/all_files.csv'
CUDA_VISIBLE_DEVICES=1 /xingjiandiao/miniconda3/envs/pytorch/bin/python run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.75 \
        --model pretrain_videomae_small_patch16_224 \
        --decoder_depth 4 \
        --lr 3e-4 \
        --batch_size 20 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 3201 \
        --save_ckpt_freq 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
