#!/bin/bash
export OMP_NUM_THREADS=3

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
DATASET=${1:-"scannet"}

if [ "$DATASET" == "scannetpp" ]; then
    echo "Training on ScanNet++ dataset"
    DATASET_CONFIG="scannetpp_stage1"
    SAM_FOLDER="gt_mask"
    SCENES_TO_EXCLUDE=\'dfac5b38df,00dd871005,c4c04e6d6c\'
else
    echo "Training on ScanNet dataset"
    DATASET_CONFIG="scannet_stage1"
    SAM_FOLDER="sam"
    SCENES_TO_EXCLUDE=""
fi

CUDA_VISIBLE_DEVICES=0,1 python main_instance_segmentation_stage1.py \
    general.experiment_name="train_stage1_${DATASET}_${CURRENT_TIME}" \
    general.project_name="${DATASET}" \
    data/datasets=${DATASET_CONFIG} \
    data.scenes_to_exclude="${SCENES_TO_EXCLUDE}" \
    data.train_dataset.max_frames=152000 \
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