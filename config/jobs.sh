cd /datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2

module load python
source /scratch3/nic261/environments/cxrmate2_virga_final/bin/activate

# python main.py -t cxrmate2/final -c cxrmate2/config/000_sft.yaml --submit --train --test --trial 5
# python main.py -t cxrmate2/final -c cxrmate2/config/000_sft.yaml --submit --train --test --trial 6
# python main.py -t cxrmate2/final -c cxrmate2/config/000_sft.yaml --submit --train --test --trial 7
# python main.py -t cxrmate2/final -c cxrmate2/config/000_sft.yaml --submit --train --test --trial 8

# python main.py -t cxrmate2/final -c cxrmate2/config/001_grpo.yaml --submit --train --test --trial 6
# python main.py -t cxrmate2/final -c cxrmate2/config/001_grpo.yaml --submit --train --test --trial 8
# python main.py -t cxrmate2/final -c cxrmate2/config/001_grpo.yaml --submit --train --test --trial 5
# python main.py -t cxrmate2/final -c cxrmate2/config/001_grpo.yaml --submit --train --test --trial 7

# python main.py -t cxrmate2/final -c cxrmate2/config/002_grpo_rev_a.yaml --submit --train --test --trial 5
# python main.py -t cxrmate2/final -c cxrmate2/config/002_grpo_rev_a.yaml --submit --train --test --trial 6

# python main.py -t cxrmate2/final -c cxrmate2/config/003_mimic_cxr_only.yaml --submit --train --test --trial 0
# python main.py -t cxrmate2/final -c cxrmate2/config/004_chexpert_plus_only.yaml --submit --train --test --trial 0
# python main.py -t cxrmate2/final -c cxrmate2/config/005_rexgradient_only.yaml --submit --train --test --trial 0

# python main.py -t cxrmate2/final -c cxrmate2/config/003_mimic_cxr_only.yaml --submit --train --test --trial 1
# python main.py -t cxrmate2/final -c cxrmate2/config/004_chexpert_plus_only.yaml --submit --train --test --trial 1
# python main.py -t cxrmate2/final -c cxrmate2/config/005_rexgradient_only.yaml --submit --train --test --trial 1

# python main.py -t cxrmate2/final -c cxrmate2/config/003_mimic_cxr_only.yaml --submit --train --test --trial 2
# python main.py -t cxrmate2/final -c cxrmate2/config/004_chexpert_plus_only.yaml --submit --train --test --trial 2
# python main.py -t cxrmate2/final -c cxrmate2/config/005_rexgradient_only.yaml --submit --train --test --trial 2

# for config in 003_mimic_cxr_only 004_chexpert_plus_only 005_rexgradient_only; do
# for config in 005_rexgradient_only; do
#     for trial in 0 1 2; do
        # for dataset in mimic_cxr chexpert_plus rexgradient; do
        # for dataset in mimic_cxr chexpert_plus; do
#         for dataset in rexgradient; do
#             python main.py -t cxrmate2/final -c cxrmate2/config/${config}.yaml --submit --test --trial ${trial} --test_datasets ${dataset}
#         done
#     done
# done

# python main.py -t cxrmate2/final -c cxrmate2/config/007_dpo_lora.yaml --submit --train --test --trial 5
# python main.py -t cxrmate2/final -c cxrmate2/config/007_dpo_lora.yaml --submit --train --test --trial 6

# python main.py -t cxrmate2/final -c cxrmate2/config/006_dpo.yaml --submit --train --test --trial 5
# python main.py -t cxrmate2/final -c cxrmate2/config/006_dpo.yaml --submit --train --test --trial 6

# python main.py -t cxrmate2/final -c cxrmate2/config/002_grpo_rev_a.yaml --submit --test --trial 5 --test_datasets mimic_cxr_dpo
# python main.py -t cxrmate2/final -c cxrmate2/config/002_grpo_rev_a.yaml --submit --test --trial 6 --test_datasets mimic_cxr_dpo

python main.py -t cxrmate2/final -c cxrmate2/config/007_dpo_lora.yaml --submit --test --trial 5
python main.py -t cxrmate2/final -c cxrmate2/config/007_dpo_lora.yaml --submit --test --trial 6