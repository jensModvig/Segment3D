#!/bin/bash
export OMP_NUM_THREADS=3
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
DATASET=${1:-"scannet"}
shift  # Remove dataset argument, leaving additional config in $@

if [ "$DATASET" == "scannetpp" ]; then
    echo "Training on ScanNet++ dataset"
    DATASET_CONFIG="scannetpp_stage1"
    SAM_FOLDER="gt_mask"
    SCENES_TO_EXCLUDE=\'00dd871005,c4c04e6d6c\'
else
    echo "Training on ScanNet dataset"
    DATASET_CONFIG="scannet_stage1"
    SAM_FOLDER="sam"
    SCENES_TO_EXCLUDE=""
fi

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Found $GPU_COUNT GPUs: $CUDA_VISIBLE_DEVICES"

python main_instance_segmentation_stage1.py \
    general.experiment_name="train_stage1_${DATASET}_${CURRENT_TIME}" \
    general.project_name="${DATASET}" \
    data/datasets=${DATASET_CONFIG} \
    data.scenes_to_exclude="${SCENES_TO_EXCLUDE}" \
    data.max_frames=76320 \
    optimizer.lr=0.0002 \
    data.batch_size=8 \
    data.num_workers=4 \
    data.sam_folder="${SAM_FOLDER}" \
    data.label_min_area=0 \
    trainer.max_epochs=20 \
    trainer.log_every_n_steps=500 \
    trainer.val_check_interval=2000 \
    general.save_visualizations=False \
    general.gpus=${GPU_COUNT} \
    model.num_queries=100 \
    "$@"