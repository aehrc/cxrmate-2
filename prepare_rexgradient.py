
import glob
import io
import multiprocessing
import os
import shutil
import tarfile
from pathlib import Path

import cv2
import datasets
import numpy as np
import pandas as pd
import zstandard as zstd
from huggingface_hub import hf_hub_download
from PIL import Image


def prepare_dataset(database_dir, num_workers=None):

    file_paths = []
    for i in range(10):
        file_paths.append(hf_hub_download(repo_id='rajpurkarlab/ReXGradient-160K', filename=f'deid_png.part0{i}', repo_type='dataset'))

    output_tar_file_path = os.path.join(database_dir, 'rexgradient_images.tar')
    output_images_dir = os.path.join(database_dir, 'rexgradient_images')
    if not os.path.exists(output_tar_file_path):
        with open(output_tar_file_path, 'wb') as tar_f:
            for i in file_paths:
                with open(i, 'rb') as part_f:
                    shutil.copyfileobj(part_f, tar_f)
        with open(output_tar_file_path, 'rb') as comp_fd:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(comp_fd)
            with tarfile.open(mode='r|*', fileobj=stream_reader) as tar:
                tar.extractall(path=output_images_dir)

    dataset = datasets.load_dataset('rajpurkarlab/ReXGradient-160K')
    
    num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()

    def load_image(row):
        images = []
        for subject_id, accession_number, study_id in zip(row['subject_id'], row['AccessionNumber'], row['StudyInstanceUid'], strict=True):
            study_images = []
            image_paths = glob.glob(os.path.join(database_dir, 'rexgradient_images', 'deid_png', subject_id[1:], accession_number, 'studies', study_id, 'series', '*', 'instances', '*'))
            for image_path in image_paths:

                image = Image.open(image_path)
                image_np = np.array(image, dtype=np.float32)
                assert image_np.ndim == 2
                min_val = image_np.min()
                denom = image_np.max() - min_val
                if denom == 0:
                    raise ValueError(f'Cannot normalize image with zero dynamic range (min and max both {min_val}).')
                image_np = (image_np - min_val) / denom
                image_uint8 = (image_np * 255).astype(np.uint8)
                image_eq = cv2.equalizeHist(image_uint8)
                pil_norm = Image.fromarray(image_eq)
                buf = io.BytesIO()
                pil_norm.save(buf, format='PNG')
                study_images.append(buf.getvalue())

            images.append(study_images)
        row['images'] = images
        return row
            
    dataset_dict = {}
    for split in ['train', 'validation', 'test']:
                
        df = dataset[split].to_pandas()
        df['subject_id'] = df['id'].str.split('_').str[0]
        df['StudyDate'] = pd.to_datetime(df['StudyDate'], format='%Y%m%d')
        df = df.sort_values(['subject_id', 'StudyDate'])
        def add_priors(group):
            group = group.sort_values('StudyDate')
            study_ids = group['StudyInstanceUid'].tolist()
            dates = group['StudyDate'].tolist()

            prior_study_ids = []
            prior_study_datetimes = []
            for i, current_date in enumerate(dates):
                ids = [study_ids[j] for j in range(i) if dates[j] < current_date]
                dts = [dates[j] for j in range(i) if dates[j] < current_date]
                prior_study_ids.append(ids)
                prior_study_datetimes.append(dts)

            return group.assign(
                prior_study_ids=prior_study_ids,
                prior_study_datetimes=prior_study_datetimes
            )

        df = df.groupby('subject_id', group_keys=False).apply(add_priors)
        df['demographics'] = df.apply(
            lambda row: ', '.join(
                str(val) for val in [row['PatientSex'], row['PatientAge']] if pd.notnull(val)
            ) or None,
            axis=1
        )
        df = df.reset_index(drop=True)
                            
        dataset_dict[split] = datasets.Dataset.from_pandas(df)
        cache_dir = os.path.join(database_dir, '.cache')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        dataset_dict[split] = dataset_dict[split].map(
            load_image,
            num_proc=num_workers,
            writer_batch_size=8,
            batched=True,
            batch_size=8,
            keep_in_memory=False,
            cache_file_name=os.path.join(cache_dir, f'.{split}'),
            load_from_cache_file=False,
        )
        dataset_dict[split].cleanup_cache_files()
        shutil.rmtree(cache_dir)
        
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(os.path.join(database_dir, 'rexgradient_160k_dataset'))
    
if __name__ == '__main__':
    database_dir = '/scratch3/nic261/database/cxrmate2'  # Where the resultant database will be stored.

    num_workers = None if os.getenv('SLURM_JOB_ID') is not None else 1

    prepare_dataset(
        database_dir=database_dir,
        num_workers=num_workers,
    )
