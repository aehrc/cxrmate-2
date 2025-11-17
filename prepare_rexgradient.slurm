#!/bin/bash -l
#SBATCH --job-name=prepare_dataset
#SBATCH --output=/scratch3/nic261/slurm_outputs/%j.out
#SBATCH --error=/scratch3/nic261/slurm_outputs/%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aaron.nicolson@csiro.au
#SBATCH --account=OD-232446
#SBATCH --partition=h24gpu

module load python
source /scratch3/nic261/environments/cxrmate2_virga/bin/activate
cd /datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/cxrmate2/base_modules
python prepare_rexgradient_dataset.py
