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

python main.py -t cxrmate2/final -c cxrmate2/config/002_grpo_rev_a.yaml --submit --train --test --trial 5
python main.py -t cxrmate2/final -c cxrmate2/config/002_grpo_rev_a.yaml --submit --train --test --trial 6