#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
DATASET=${1:-"scannet"}  # Default to scannet if no argument provided

if [ "$DATASET" == "scannetpp" ]; then
    echo "Training on ScanNet++ dataset"
    DATASET_CONFIG="scannetpp_stage1"
    SAM_FOLDER="gt_mask"
else
    echo "Training on ScanNet dataset"
    DATASET_CONFIG="scannet_stage1"
    SAM_FOLDER="sam"  # or whatever your scannet sam folder is
fi

CUDA_VISIBLE_DEVICES=0,1 python main_instance_segmentation_stage1.py \
    general.experiment_name="train_stage1_${DATASET}_${CURRENT_TIME}" \
    general.project_name="${DATASET}" \
    data/datasets=${DATASET_CONFIG} \
    optimizer.lr=0.0002 \
    data.batch_size=8 \
    data.num_workers=4 \
    data.sam_folder="${SAM_FOLDER}" \
    trainer.max_epochs=20 \
    trainer.log_every_n_steps=50 \
    trainer.val_check_interval=2000 \
    general.save_visualizations=False \
    general.gpus=2 \
    model.num_queries=100