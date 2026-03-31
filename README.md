# CXRMate-2

This repository provides the code to train the CXRMate-2 model.

```
Citation
```


## Download model
The model is available on Hugging Face Hub: https://huggingface.co/aehrc/cxrmate-2

```python
alias = 'aehrc/cxrmate-2'

model = transformers.AutoModelForCausalLM.from_pretrained(alias, trust_remote_code=True).to(device='cuda')
model.eval()
generation_config = transformers.GenerationConfig.from_pretrained(alias, trust_remote_code=True)
processor = transformers.AutoProcessor.from_pretrained(alias, trust_remote_code=True)
```

## Generation
```python
url = 'https://prod-images-static.radiopaedia.org/images/220869/76052f7902246ff862f52f5d3cd9cd_big_gallery.jpg'
processed = processor(images=url)
processed = processed.to(device='cuda')
generated_ids = model.generate(**processed, generation_config=generation_config)
findings, impression = processor.split_and_decode_sections(generated_ids) 
```

## Environment

## Training

## Download Datasets

#### MIMIC-CXR:

MIMIC-CXR and MIMIC-CXR-JPG must be in the same Physio Net directory. E.g.:

```shell
user@cluster:~$ ls /home/user/physionet.org/files
mimic-cxr  mimic-cxr-jpg  mimic-iv-ed
```

### Download MIMIC-CXR-JPG:
Download the MIMIC-CXR-JPG dataset from https://physionet.org/content/mimic-cxr-jpg, e.g.,
```shell
wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```
Note that you must be a credentialised user to access this dataset.

### Download the reports from MIMIC-CXR:
MIMIC-CXR-JPG does not include the radiology reports and are instead included with MIMIC-CXR (the DICOM version of the dataset). To download this dataset and avoid downloading the DICOM files (which are very large), use `--reject dcm` with the wget command from https://physionet.org/content/mimic-cxr, e.g, 
```shell
wget -r -N -c -np --reject dcm --user <username> --ask-password https://physionet.org/files/mimic-cxr/2.0.0/
```
Note that you must be a credentialised user to access this dataset.

## Prepare datasets:
