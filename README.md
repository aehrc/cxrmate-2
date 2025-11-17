# CXRMate-2


This is the model and data pipeline for the CXRMate-2 model. 

```
Citation
```

Abstract:

## Hugging Face Hub
The model is available on Hugging Face Hub: https://huggingface.co/aehrc/cxrmate-2

## Generation

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
