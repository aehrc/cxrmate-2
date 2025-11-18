#!/bin/bash -l
#SBATCH --job-name=prepare_dataset/prepare_rexgradient
#SBATCH --output=/scratch3/nic261/slurm_outputs/prepare_rexgradient/%j.out
#SBATCH --error=/scratch3/nic261/slurm_outputs/prepare_rexgradient/%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aaron.nicolson@csiro.au
#SBATCH --account=OD-232446
#SBATCH --partition=h24gpu
#SBATCH --qos=express

module load python
source /scratch3/nic261/environments/cxrmate2_virga_final/bin/activate
cd /datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/cxrmate2
python -c "import datasets; print('datasets version:', datasets.__version__)"
python prepare_rexgradient.py
