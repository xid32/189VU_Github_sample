# Set the path to save video
OUTPUT_DIR='results/vit_small/BalanceBeam'
# path to video for visualization
VIDEO_PATH='data/UCF-101/BalanceBeam/v_BalanceBeam_g01_c01.avi'
# path to pretrain model
MODEL_PATH='logs_original/checkpoint-499.pth'

python3 VideoMAE/run_videomae_vis.py \
    --mask_ratio 0.75 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_small_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}