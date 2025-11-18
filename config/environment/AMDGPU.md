```shell
# https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html
module load apptainer
cd /scratch3/nic261/images
IMAGE_NAME=rocm_pytorch-training_v25.4.sif
apptainer pull $IMAGE_NAME docker://rocm/pytorch-training:v25.4
APPTAINER_BINDPATH="/datasets,/scratch3" apptainer shell $IMAGE_NAME 
ENV_NAME=/scratch3/nic261/environments/cxrmate2_amdgpu_final
rm -r $ENV_NAME
python -m venv --system-site-packages $ENV_NAME
source $ENV_NAME/bin/activate
python -m pip install --upgrade --no-cache-dir pip
python -m pip install --no-cache-dir -r /datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/cxrmate2/config/requirements/amdgpu_final.txt
```

<!-- ## `amdgpu` `accelerate` `venv` w. ROCm PyTorch optimised Apptainer image
```shell
# https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html
module load apptainer
cd /scratch3/nic261/images
IMAGE_NAME=rocm_pytorch-training_v25.4.sif
apptainer pull $IMAGE_NAME docker://rocm/pytorch-training:v25.4
APPTAINER_BINDPATH="/datasets,/scratch3" apptainer shell $IMAGE_NAME 
ENV_NAME=/scratch3/nic261/environments/amdgpu_cxrmate2
rm -r $ENV_NAME
python -m venv --system-site-packages $ENV_NAME
source $ENV_NAME/bin/activate
python -m pip install --upgrade --no-cache-dir pip
python -m pip install --no-cache-dir -r /datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/cxrmate2/config/requirements/amdgpu.txt
``` -->