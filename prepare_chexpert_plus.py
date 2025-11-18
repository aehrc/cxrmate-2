import multiprocessing
import os
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import datasets
import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm


def format_text(text):
    
    # Remove newline, tab, repeated whitespaces, and leading and trailing whitespaces:
    def remove(text):
        text = re.sub(r'\n|\t', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    if isinstance(text, np.ndarray) or isinstance(text, list):
        return [remove(t) if not pd.isna(t) else t for t in text]
    else:
        if pd.isna(text):
            return text
        return remove(text)


def prepare_dataset(chexpert_plus_dir, database_dir, num_workers=None):
    
    num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
    
    zip_paths = glob(os.path.join(chexpert_plus_dir, 'PNG', '*.zip'))
        
    def extract_png_paths(zip_path):
        png_paths = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                png_paths = [{'zip_path': zip_path, 'path_to_image': j} for j in z.namelist() if j.endswith('.png')]
        except Exception as e:
            print(f'Error processing file {zip_path}: {e}')
        return png_paths

    rows = []
    with ThreadPoolExecutor(max_workers=num_workers if num_workers > 0 else 1) as ex:
        futures = {ex.submit(extract_png_paths, zip_path): zip_path for zip_path in zip_paths}
        for f in tqdm(as_completed(futures), total=len(futures)):
            try:
                png_paths = f.result()
                rows.extend(png_paths)
            except Exception as e:
                print(f'Error in future for {futures[f]}: {e}')
                                   
    con = duckdb.connect(database=':memory:')
    con.sql(f"CREATE OR REPLACE TABLE studies AS FROM '{os.path.join(chexpert_plus_dir, 'df_chexpert_plus_240401.csv')}'")
    num_rows = con.sql("SELECT COUNT(*) FROM studies;").fetchone()[0]
    
    # Create study_id column:
    con.sql(
        """
        ALTER TABLE studies
        ADD COLUMN study_id VARCHAR;
        """   
    )
    con.sql(
        """
        UPDATE studies
        SET study_id = CONCAT(deid_patient_id, '_', split_part(path_to_image, '/', 3));
        """
    )
    
    # Fix path_to_image column:
    con.sql(
        """
        UPDATE studies
        SET path_to_image = REPLACE(path_to_image, '.jpg', '.png');
        """
    )

    # Creat zip paths table:
    df_zip_paths = pd.DataFrame(rows)
    df_zip_paths['path_to_image'] = df_zip_paths['path_to_image'].str.replace('PNG/', '', regex=True)
    con.sql(
        """
        CREATE OR REPLACE TABLE zip_paths AS FROM df_zip_paths;
        """
    )
    assert con.sql("SELECT COUNT(*) FROM zip_paths;").fetchone()[0] == num_rows

    # Merge zip paths with studies:
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT 
            s.*, 
            z.zip_path,
        FROM studies s
        LEFT JOIN zip_paths z
        ON s.path_to_image = z.path_to_image;
        """
    )
    assert con.sql("SELECT COUNT(*) FROM studies;").fetchone()[0] == num_rows
    num_studies = con.sql("SELECT COUNT(DISTINCT study_id) FROM studies;").fetchone()[0]

    # Add views column:
    con.sql("ALTER TABLE studies ADD COLUMN views VARCHAR;")
    con.sql("UPDATE studies SET views = COALESCE(ap_pa, frontal_lateral);")

    # Group by study_id:
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT 
            study_id,
            LIST(path_to_image) AS path_to_image,
            LIST(zip_path) AS zip_path,
            LIST(views) AS views,
            FIRST(deid_patient_id) AS deid_patient_id, 
            FIRST(patient_report_date_order) AS patient_report_date_order,
            FIRST(section_narrative) AS section_narrative,
            FIRST(section_clinical_history) AS section_clinical_history,
            FIRST(section_history) AS section_history,
            FIRST(section_comparison) AS section_comparison,
            FIRST(section_technique) AS section_technique,
            FIRST(section_procedure_comments) AS section_procedure_comments,
            FIRST(section_findings) AS section_findings,
            FIRST(section_impression) AS section_impression,
            FIRST(section_end_of_impression) AS section_end_of_impression,
            FIRST(section_summary) AS section_summary,
            FIRST(section_accession_number) AS section_accession_number,
            FIRST(age) AS age,
            FIRST(sex) AS sex,
            FIRST(race) AS race,
            FIRST(ethnicity) AS ethnicity,
            FIRST(interpreter_needed) AS interpreter_needed,
            FIRST(insurance_type) AS insurance_type,
            FIRST(recent_bmi) AS recent_bmi,
            FIRST(split) AS split,
        FROM studies
        GROUP BY study_id;
        """
    )
    assert con.sql("SELECT COUNT(*) FROM studies;").fetchone()[0] == num_studies
    
    # Combine create history and demographics columns:
    con.sql(
        """
        ALTER TABLE studies ADD COLUMN history VARCHAR;
        """
    )
    con.sql(
        """
        ALTER TABLE studies ADD COLUMN demographics VARCHAR;
        """
    )
    con.sql(
        """
        UPDATE studies
        SET history = concat_ws(' ', section_history, section_clinical_history),
            demographics = concat_ws(', ', age, sex, race, ethnicity);
        """
    )
    
    # Remove study with corrupt image:
    con.sql("DELETE FROM studies WHERE study_id = 'patient32368_study1';")
    
    # Prior studies column:
    con.sql(
        """
        ALTER TABLE studies
        ADD COLUMN prior_study_ids VARCHAR[];
        """
    )
    con.sql(
        """
        UPDATE studies AS s1
        SET prior_study_ids = list_sort((
            SELECT list(s2.study_id)
            FROM studies AS s2
            WHERE s2.deid_patient_id = s1.deid_patient_id
            AND s2.patient_report_date_order < s1.patient_report_date_order
        ));
        """
    )

    def load_image(row):
        images = []
        for batch_idx, (zip_paths, image_paths) in enumerate(zip(row['zip_path'], row['path_to_image'])):
            study_images = []
            for image_idx, (zip_path, image_path) in enumerate(zip(zip_paths, image_paths)):
                with zipfile.ZipFile(zip_path, 'r') as z:
                    with z.open(f'PNG/{image_path}') as f: 
                        try:
                            study_images.append(f.read())
                        except Exception as e:
                            
                            row['path_to_image'][batch_idx].pop(image_idx)
                            row['zip_path'][batch_idx].pop(image_idx)
                            row['views'][batch_idx].pop(image_idx)
                            
                            print(f'Error processing file {image_path} from {zip_path}: {e}. Removing from dataset.')
                            
                            assert len(row['path_to_image'][batch_idx]) > 0
            images.append(study_images)
        row['images'] = images
        return row
    
    # Standardise column names across the datasets:
    con.sql("ALTER TABLE studies RENAME COLUMN section_findings TO findings;")
    con.sql("ALTER TABLE studies RENAME COLUMN section_impression TO impression;")
    con.sql("ALTER TABLE studies RENAME COLUMN section_comparison TO comparison;")
    con.sql("ALTER TABLE studies RENAME COLUMN section_technique TO technique;")

    con.sql("ALTER TABLE studies ADD COLUMN study_datetime TIMESTAMP;")
    con.sql("ALTER TABLE studies ADD COLUMN prior_study_datetimes VARCHAR[];")
    con.sql("UPDATE studies SET prior_study_datetimes = prior_study_ids;")  # This is used to sort the prior studies (not used to compute time delta).

    dataset_dict = {}
    for split in ['valid', 'train']:
                
        df = con.sql(
            f"""
            FROM studies
            WHERE split = '{split}'
            ORDER BY study_id;
            """
        ).df()
        
        # Format text columns:
        for i in [j for j in df.columns if ('section_' in j) or (j == 'history')]:
            df[i] = df[i].apply(format_text)

        # Fix capitalised impression sections:
        def to_sentence_case(text: str) -> str:
            if text is not None:
                text = text.lower()
                parts = re.split(r'([.!?]\s*)', text)
                return ''.join(part.capitalize() if i % 2 == 0 else part
                            for i, part in enumerate(parts))
            return text
        df['impression'] = df['impression'].apply(to_sentence_case)
            
        dataset_dict[split] = datasets.Dataset.from_pandas(df)
        dataset_dict[split] = dataset_dict[split].map(
            load_image,
            num_proc=num_workers,
            writer_batch_size=1,
            batched=True,
            batch_size=1,
            keep_in_memory=False,
            load_from_cache_file=False,
            cache_file_name=os.path.join(database_dir, f'cache_{split}.arrow'),
        )
        dataset_dict[split].cleanup_cache_files()
        
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(os.path.join(database_dir, 'chexpert_plus_dataset'))
    
    con.close()

if __name__ == '__main__':
    chexpert_plus_dir = '/datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/work/data/chexpertplus'  # Where the CheXpert Plus DICOM directory, PNG directory, and df_chexpert_plus_240401.csv file are stored.
    database_dir = '/scratch3/nic261/database/cxrmate2'  # Where the resultant database will be stored.

    prepare_dataset(
        chexpert_plus_dir=chexpert_plus_dir, 
        database_dir=database_dir,
        num_workers=4,
    )
