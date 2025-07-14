
export MODEL_PATH=/home/zhq/workdir/GUI/Qwen2.5-VL-3B-Instruct
export PROCESSED_DATA_DIR=/home/zhq/workdir/GUI/screenspot-v2/processed_prompt_old
export IMAGE_DIR=/home/zhq/workdir/GUI/screenspot-v2/screenspotv2_image
export CONFIG_FILE=/home/zhq/workdir/GUI/EasyR1/examples/config.yaml
export CHECKPOINT_TO_LOAD=/home/zhq/workdir/GUI/EasyR1/verl/trainer/checkpoints/easy_r1/qwen2_5_vl_3b_screenspot_mgrpo_v/global_step_330

python3 -m verl.trainer.main \
    config=${CONFIG_FILE} \
    data.train_files=${PROCESSED_DATA_DIR}/train_with_prompt.jsonl \
    data.val_files=${PROCESSED_DATA_DIR}/val_with_prompt.jsonl \
    data.image_dir=${IMAGE_DIR} \
    data.rollout_batch_size=4 \
    data.val_batch_size=8 \
    algorithm.adv_estimator=mgrpo_v \
    worker.actor.global_batch_size=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=3 \
    worker.rollout.gpu_memory_utilization=0.7 \
    trainer.load_checkpoint_path=${CHECKPOINT_TO_LOAD} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_screenspot_mgrpo_v \
    trainer.n_gpus_per_node=1

echo "Training script finished."

