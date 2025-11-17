#!/bin/bash -l
#SBATCH --job-name=cxrmate2/test_checkpoint
#SBATCH --output=/datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/scratch3/slurm_outputs/cxrmate2/%j.out
#SBATCH --error=/datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/scratch3/slurm_outputs/cxrmate2/%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --nodes=1
#SBATCH --mem=450GB
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aaron.nicolson@csiro.au
#SBATCH --account=OD-232446
#SBATCH --qos=express

echo "Script Arguments:"
echo "$@"

module load python
cd /datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/cxrmate2

export GPUS_PER_NODE=4
HEAD_NODE_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Head node: $HEAD_NODE_IP."

if [[ -z "${VENV_PATH:-}" ]]; then
  echo "Error: VENV_PATH is not set."
  exit 1
fi

source "$VENV_PATH"

srun -u accelerate launch \
  --main_process_ip "$HEAD_NODE_IP" \
  --main_process_port 29500 \
  --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
  --num_machines "$SLURM_NNODES" \
  --rdzv_backend c10d \
  evaluate_checkpoints.py "$@"