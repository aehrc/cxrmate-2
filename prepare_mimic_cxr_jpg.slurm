#!/bin/bash -l
#SBATCH --job-name=prepare
#SBATCH --output=/scratch3/nic261/slurm_outputs/%j.out
#SBATCH --error=/scratch3/nic261/slurm_outputs/%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aaron.nicolson@csiro.au
#SBATCH --account=OD-232446
#SBATCH --partition=h24gpu
#SBATCH --qos=express

module load python
source /scratch3/nic261/environments/cxrmate2_virga_rev_a/bin/activate
cd /datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/cxrmate2/prepare_dataset
python prepare_mimic_cxr_jpg.py
