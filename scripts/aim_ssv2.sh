#!/bin/bash
#SBATCH -p batch_ce_ugrad 
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=30G
#SBATCH --time=4-00:00:0

socket_ifname=$(cat /etc/hosts | grep $(hostname) | grep -Eo 'en\w+')
export NCCL_SOCKET_IFNAME=$socket_ifname

# DATA_PATH='/local_datasets/kinetics400_320p'
MODEL_PATH='/data/jong980812/project/VideoMAE/k400_pretrain_base_patch16_224_frame_16x5.pth'
# OUTPUT_DIR=/data/jong980812/project/VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x5_tube_mask_ratio_0.9_e800/eval_lr_1e-3_epoch_100 # weight저장


# batch_size can be adjusted according to the graphics card
OUTPUT_DIR=$4 # weight저장.
MASTER_NODE=$1
OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$6 \
    --master_port $3 --nnodes=$5 \
    --node_rank=$2 --master_addr=${MASTER_NODE} \
    /data/jong980812/project/VideoMAE/run_class_finetuning.py \
    --model AIM \
    --data_set SSV2 \
    --nb_classes 174 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --anno_path /data/jong980812/project/VideoMAE/video_annotation/SSV2 \
    --data_path /local_datasets/something-something/something-something-v2-mp4 \
    --unfreeze_layers head Adapter ln_post \
    --batch_size 12 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --num_workers 12 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --warmup_epochs 5 \
    --dist_eval \
    --test_num_segment 1 \
    --test_num_crop 3 \
    --update_freq 2 \
    --enable_deepspeed \
    --mixup_prob 0. \
    --mixup_switch_prob 0. \
    --mixup 0. \
    --cutmix 0. \



    # --eval \
    # --finetune /data/jong980812/project/VideoMAE/result/aim_ssv2/OUT/checkpoint-best/mp_rank_00_model_states.pt
    # --finetune ${MODEL_PATH} \