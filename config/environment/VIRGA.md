```shell
module load python
ENV_NAME=/scratch3/nic261/environments/cxrmate2_virga_final
rm -r $ENV_NAME
python -m venv $ENV_NAME
source $ENV_NAME/bin/activate
python -m pip install --upgrade --no-cache-dir pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --no-cache-dir --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-cache-dir -r /datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/cxrmate2/config/requirements/virga_final.txt
```

