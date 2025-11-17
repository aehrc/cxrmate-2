#!/bin/bash

cd /datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/cxrmate2

# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 5 --test_set mimic_cxr --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 5 --test_set mimic_cxr --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 5 --test_set chexpert_plus --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 5 --test_set chexpert_plus --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 5 --test_set rexgradient --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 5 --test_set rexgradient --evaluate

# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 6 --test_set mimic_cxr --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 6 --test_set mimic_cxr --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 6 --test_set chexpert_plus --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 6 --test_set chexpert_plus --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 6 --test_set rexgradient --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 6 --test_set rexgradient --evaluate

# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 7 --test_set mimic_cxr --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 7 --test_set mimic_cxr --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 7 --test_set chexpert_plus --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 7 --test_set chexpert_plus --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 7 --test_set rexgradient --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 7 --test_set rexgradient --evaluate

# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 8 --test_set mimic_cxr --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 8 --test_set mimic_cxr --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 8 --test_set chexpert_plus --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 8 --test_set chexpert_plus --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 8 --test_set rexgradient --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-2 --trial 8 --test_set rexgradient --evaluate

# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model MAIRA-2 --test_set mimic_cxr --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model MAIRA-2 --test_set mimic_cxr --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model MAIRA-2 --test_set chexpert_plus --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model MAIRA-2 --test_set chexpert_plus --evaluate

SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_medgemma_virga/bin/activate \
    evaluate_checkpoints.slurm --model MedGemma --test_set mimic_cxr --generate | awk '{print $4}')
sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
    evaluate_checkpoints.slurm --model MedGemma --test_set mimic_cxr --evaluate
SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_medgemma_virga/bin/activate \
    evaluate_checkpoints.slurm --model MedGemma --test_set chexpert_plus --generate | awk '{print $4}')
sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
    evaluate_checkpoints.slurm --model MedGemma --test_set chexpert_plus --evaluate
# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_medgemma_virga/bin/activate \
#     evaluate_checkpoints.slurm --model MedGemma --test_set rexgradient --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID  --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model MedGemma --test_set rexgradient --evaluate

# SLURM_JOB_ID=$(sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/virga_cxrmate2/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-RRG24 --test_set mimic_cxr --generate | awk '{print $4}')
# sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-RRG24 --test_set mimic_cxr --evaluate

# sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate-ED --test_set mimic_cxr --evaluate
# sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model CXRMate --test_set mimic_cxr --evaluate
# sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model EMNLI --test_set mimic_cxr --evaluate

# sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model MedVersa --test_set mimic_cxr --evaluate
# sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model MedVersa --test_set chexpert_plus --evaluate
# sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model MedVersa --test_set rexgradient --evaluate

# sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model Libra --test_set mimic_cxr --evaluate
# sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model Libra --test_set chexpert_plus --evaluate
# sbatch --export=ALL,VENV_PATH=/scratch3/nic261/environments/cxrmate2_eval_virga/bin/activate \
#     evaluate_checkpoints.slurm --model Libra --test_set rexgradient --evaluate