#!/bin/bash

#SBATCH --job-name k400_finetune
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=50G
#SBATCH --time 4-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y3
#SBATCH -o /data/jong980812/project/VideoMAE/%A-%x.out
#SBATCH -e /data/jong980812/project/VideoMAE/%A-%x.err
echo $PWD
echo $SLURMD_NODENAME
current_time=$(date "+%Y%m%d-%H:%M:%S")
echo $current_time
# export MASTER_PORT=12345

DATA_PATH=/local_datasets/kinetics400_320p
MODEL_PATH=/data/jong980812/project/VideoMAE/k400_pretrain_base_patch16_224_frame_16x5.pth
OUTPUT_DIR=/data/jong980812/project/VideoMAE/k400_pretrain_base_patch16_224_frame_16x5_tube_mask_ratio_0.9_e800/eval_lr_1e-3_epoch_100 # weight저장
ANNOTATION_PATH=/data/jong980812/project/VideoMAE/video_annotation/kinetics100

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port 12345 --nnodes=1 run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --anno_path ${ANNOTATION_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 12 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --enable_deepspeed

echo "Job finish"
exit 0