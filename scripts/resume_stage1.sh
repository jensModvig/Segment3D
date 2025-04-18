#!/bin/bash
export OMP_NUM_THREADS=3 # speeds up MinkowskiEngine

# Experiment details
EXPERIMENT_NAME="train_stage1_20250409_194708_continue"
EXPERIMENT_DIR="/work3/s173955/Segment3D/saved/${EXPERIMENT_NAME}"
CHECKPOINT_PATH="${EXPERIMENT_DIR}/epoch=15-val_mean_box_ap_50=0.102.ckpt"

# Verify the checkpoint exists in the new location
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Create a symlink with a name that doesn't contain equal signs
SAFE_CHECKPOINT_PATH="${EXPERIMENT_DIR}/resume_checkpoint.ckpt"
ln -sf "${CHECKPOINT_PATH}" "${SAFE_CHECKPOINT_PATH}"

CUDA_VISIBLE_DEVICES=0,1 python main_instance_segmentation_stage1.py \
general.experiment_name="${EXPERIMENT_NAME}" \
general.project_name="scannet" \
logging=minimal \
optimizer.lr=0.0002 \
data.batch_size=8 \
data.num_workers=4 \
trainer.max_epochs=20 \
trainer.log_every_n_steps=5 \
trainer.val_check_interval=2000 \
general.save_visualizations=False \
general.gpus=2 \
model.num_queries=100 \
trainer.resume_from_checkpoint="${SAFE_CHECKPOINT_PATH}"