
export MODEL_PATH=/home/zhq/workdir/GUI/Qwen2.5-VL-3B-Instruct
export PROCESSED_DATA_DIR=/home/zhq/workdir/GUI/screenspot-v2/processed
export IMAGE_DIR=/home/zhq/workdir/GUI/screenspot-v2/screenspotv2_image
export CONFIG_FILE=/home/zhq/workdir/GUI/EasyR1/examples/config.yaml

# --- Training Command ---
python3 -m verl.trainer.main \
    config=${CONFIG_FILE} \
    data.train_files=${PROCESSED_DATA_DIR}/train.jsonl \
    data.val_files=${PROCESSED_DATA_DIR}/val.jsonl \
    data.image_dir=${IMAGE_DIR} \
    algorithm.adv_estimator=mgrpo_v \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_screenspot_mgrpo_v \
    trainer.n_gpus_per_node=1

echo "Training script finished."

