#!/bin/bash
#SBATCH --job-name=jsq-tinyzero
#SBATCH --output=slurm_output.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=800
#SBATCH --constraint=48GBgpu

nvidia-smi

# Load the appropriate CUDA and GCC modules
module load cuda/11.7  # Choose cuda/11.7 or cuda/11.8 based on availability

# Display CUDA version
echo "CUDA Version:"
nvcc --version

# Verify CUDA availability in PyTorch
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python --version

# List available modules for debugging
echo "Listing Available Modules:"
module avail

# Set environment variables
export N_GPUS=1
export BASE_MODEL=/home1/j/jsq/dev/TinyZero-Efficient/models/qwen2.5_new
export DATA_DIR=/home1/j/jsq/data/countdown
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

export HYDRA_FULL_ERROR=1

# Activate the Conda environment
source /home1/j/jsq/miniconda3/bin/activate zero

# Navigate to the working directory
cd ~/dev/TinyZero-Efficient

# Print job details for debugging
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Time: $(date)"
echo "Current directory: $(pwd)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

ray stop || true

unset RAY_ADDRESS

python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=1312 \
  data.max_prompt_length=256 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  critic.optim.lr=1e-5 \
  critic.model.path=$BASE_MODEL \
  critic.ppo_micro_batch_size=8 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['wandb'] \
  +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.project_name=TinyZero \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.total_epochs=15 \
  +dtype=half \
  2>&1 | tee verl_demo.log
