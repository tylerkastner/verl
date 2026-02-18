#!/bin/bash

#SBATCH --job-name=verl
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kn153
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH -c 16
#SBATCH -A aip-sologen


cd /home/tkastner/projects/aip-sologen/tkastner/verl

unset ROCR_VISIBLE_DEVICES

uv run python -m verl.trainer.main_ppo \
 +actor_rollout_ref.model.override_config.attn_implementation=eager \
 algorithm.adv_estimator=grpo   \
 algorithm.norm_adv_by_std_in_grpo=True \
 actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
 actor_rollout_ref.actor.use_kl_loss=False \
 data.train_files=$SCRATCH/data/gsm8k/train.parquet \
 data.val_files=$SCRATCH/data/gsm8k/test.parquet \
 data.dataloader_num_workers=2 \
 actor_rollout_ref.rollout.agent.num_workers=2 \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.rollout.max_model_len=2048 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
actor_rollout_ref.actor.use_remove_padding=False \
actor_rollout_ref.model.use_remove_padding=False \
critic.model.use_remove_padding=False \
trainer.logger='["console","wandb"]' \
trainer.project_name=verl-part1 \
trainer.experiment_name=ppo-initial \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.resume_mode=disable \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log