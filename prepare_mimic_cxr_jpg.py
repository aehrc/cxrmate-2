import gc
import multiprocessing
import os
import re
import shutil
from builtins import len
from csv import writer
from glob import glob
from pathlib import Path

import datasets
import duckdb
import numpy as np
import pandas as pd
from create_section_files import create_section_files
from yaspin import yaspin


def mimic_cxr_image_path(dir, subject_id, study_id, dicom_id, ext='dcm'):
    return os.path.join(dir, 'p' + str(subject_id)[:2], 'p' + str(subject_id),
                        's' + str(study_id), str(dicom_id) + '.' + ext)
    
    
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
    

class PrepareDataset:

    def __init__(
            self, 
            physionet_dir, 
            database_dir, 
            mimic_cxr_ver='2.0.0', 
            num_workers=None, 
            splits=['test', 'validate', 'train'],
        ):
        self.physionet_dir = physionet_dir
        self.database_dir = database_dir
        self.mimic_cxr_ver = mimic_cxr_ver
        self.splits = splits

        self.mimic_cxr_sectioned_path = None

        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()

        self.con = duckdb.connect(':memory:')

        Path(self.database_dir).mkdir(parents=True, exist_ok=True)
    
    def __call__(self):
        self.prepare()

    def create_table_from_csv(self, csv_path):
        name = Path(csv_path).stem.replace('.csv', '').replace('.gz', '').replace('-', '_').replace('.', '_')
        with yaspin(text=f'Creating {name} table...') as sp:
            print(f'Creating {name} table...', end='')  
            try:
                self.con.sql(f"CREATE OR REPLACE TABLE {name} AS FROM '{csv_path}';") 
            except Exception:
                self.con.sql(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv_auto('{csv_path}', sample_size=-1);")   

    def prepare_radiology_report_sections(self):

        sectioned_dir = os.path.join(self.database_dir, 'mimic_cxr_sectioned')
        self.mimic_cxr_sectioned_path = os.path.join(sectioned_dir, 'mimic_cxr_sectioned.csv')

        if not os.path.exists(self.mimic_cxr_sectioned_path):
            print(f'{self.mimic_cxr_sectioned_path} does not exist, creating...')
        
            # Check if reports exist. Reports for the first and last patients are checked only for speed, this comprimises comprehensiveness for speed:
            report_paths = [
                os.path.join(self.physionet_dir, 'mimic-cxr/2.0.0/files/p10/p10000032/s50414267.txt'),
                os.path.join(self.physionet_dir, 'mimic-cxr/2.0.0/files/p10/p10000032/s53189527.txt'),
                os.path.join(self.physionet_dir, 'mimic-cxr/2.0.0/files/p10/p10000032/s53911762.txt'),
                os.path.join(self.physionet_dir, 'mimic-cxr/2.0.0/files/p10/p10000032/s56699142.txt'),
                os.path.join(self.physionet_dir, 'mimic-cxr/2.0.0/files/p19/p19999987/s55368167.txt'),
                os.path.join(self.physionet_dir, 'mimic-cxr/2.0.0/files/p19/p19999987/s58621812.txt'),
                os.path.join(self.physionet_dir, 'mimic-cxr/2.0.0/files/p19/p19999987/s58971208.txt'),
            ]
            assert all([os.path.isfile(i) for i in report_paths]), f"""The reports do not exist with the following regex: {os.path.join(self.physionet_dir, 'mimic-cxr/2.0.0/files/p1*/p1*/s*.txt')}.
            "Please download them using wget -r -N -c -np --reject dcm --user <username> --ask-password https://physionet.org/files/mimic-cxr/2.0.0/"""

            print('Extracting sections from reports...')        
            create_section_files(
                reports_path=os.path.join(self.physionet_dir, 'mimic-cxr', '2.0.0', 'files'),
                output_path=sectioned_dir,
                no_split=True,
            )

    def prepare(self):

        self.prepare_radiology_report_sections()
                    
        self.create_table_from_csv(os.path.join(self.physionet_dir, 'mimic-cxr-jpg', self.mimic_cxr_ver, f'mimic-cxr-2.0.0-metadata.csv.gz'))
        self.create_table_from_csv(os.path.join(self.physionet_dir, 'mimic-cxr-jpg', self.mimic_cxr_ver, f'mimic-cxr-2.0.0-split.csv.gz'))
        
        # Studies table:
        self.con.sql(
            """
            CREATE OR REPLACE TABLE studies AS 
            SELECT 
                dicom_id,
                subject_id,
                study_id, 
                PerformedProcedureStepDescription,
                ViewPosition,
                ProcedureCodeSequence_CodeMeaning,
                ViewCodeSequence_CodeMeaning,
                PatientOrientationCodeSequence_CodeMeaning,
                strptime(
                    CAST(StudyDate AS VARCHAR) || ' ' || lpad(split_part(CAST(StudyTime AS VARCHAR), '.', 1), 6, '0'), 
                    '%Y%m%d %H%M%S'
                ) AS study_datetime
            FROM mimic_cxr_2_0_0_metadata;
            """
        )  # Combine StudyDate and StudyTime into a single column.
        print('Studies table no. rows before grouping (i.e., no. DICOMs):', self.con.sql("SELECT COUNT(*) FROM studies").fetchone()[0])
        self.con.sql(
            """
            CREATE OR REPLACE TABLE studies AS
            SELECT 
                LIST(dicom_id) AS dicom_id, 
                FIRST(subject_id) AS subject_id, 
                study_id,    
                LIST(PerformedProcedureStepDescription) AS PerformedProcedureStepDescription, 
                LIST(ViewPosition) AS ViewPosition, 
                LIST(ProcedureCodeSequence_CodeMeaning) AS ProcedureCodeSequence_CodeMeaning,
                LIST(ViewCodeSequence_CodeMeaning) AS ViewCodeSequence_CodeMeaning,
                LIST(PatientOrientationCodeSequence_CodeMeaning) AS PatientOrientationCodeSequence_CodeMeaning,
                -- See info on tag (0008,0030) for why min is used: https://dicom.nema.org/medical/dicom/current/output/html/part03.html#table_C.7-3
                MIN(study_datetime) AS study_datetime
            FROM studies
            GROUP BY study_id;
            """
        )  # Collapse to one row per study, aggregate each studies columns as a list.
        num_studies = self.con.sql("SELECT COUNT(*) FROM studies").fetchone()[0]
        print('Studies table no. rows (i.e., no. studies):', num_studies)

        #
        #
        # Add the reports:
        #
        #
        sections = pd.read_csv(self.mimic_cxr_sectioned_path)  # DuckDB has trouble reading the sectioned .csv file, read with pandas instead.
        self.con.sql(
            """
            CREATE OR REPLACE TABLE mimic_cxr_sectioned AS 
            SELECT *, CAST(SUBSTR(study, 2) AS INT32) AS study_id 
            FROM sections;
            """
        )  # Remove the first character from the study column and rename it to study_id.
        self.con.sql(
            """
            CREATE OR REPLACE TABLE studies AS
            SELECT s.*, r.findings, r.impression, r.indication, r.history, r.comparison, r.last_paragraph, r.technique,
            FROM studies s
            LEFT JOIN mimic_cxr_sectioned r
            ON s.study_id = r.study_id
            """
        )

        #
        #  
        # Add the splits:
        #
        #
        self.con.sql(
            """
            CREATE OR REPLACE TABLE split_structs AS
            SELECT 
                study_id,    
                FIRST(split) AS split,  
            FROM mimic_cxr_2_0_0_split
            GROUP BY study_id;
            """
        )
        self.con.sql(
            """
            CREATE OR REPLACE TABLE studies AS
            SELECT s.*, x.split,
            FROM studies s
            JOIN split_structs x
            ON s.study_id = x.study_id;
            """
        )
        
        #
        #
        # Determine the prior studies:
        #
        #
        self.con.sql(
            """
            CREATE OR REPLACE TABLE prior_studies AS
            WITH sorted AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY study_datetime) AS rn
                FROM studies
            ),
            aggregated AS (
                SELECT subject_id,
                    study_id,
                    study_datetime,
                    ARRAY_AGG(study_id) OVER (PARTITION BY subject_id ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_study_ids,
                    ARRAY_AGG(study_datetime) OVER (PARTITION BY subject_id ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_study_datetimes
                FROM sorted
            )
            SELECT * 
            FROM aggregated;
            """
        )
        self.con.sql(
            """
            CREATE OR REPLACE TABLE studies AS
            SELECT s.*, p.prior_study_ids, p.prior_study_datetimes,
            FROM studies s
            LEFT JOIN prior_studies p
            ON s.study_id = p.study_id
            ORDER BY s.subject_id, s.study_datetime DESC;
            """
        )
        
        # Text columns from the sections .csv file:
        text_columns = ['findings', 'impression', 'indication', 'history', 'comparison', 'last_paragraph', 'technique']
            
        pattern = os.path.join(self.physionet_dir, 'mimic-cxr-jpg', '*', 'files')
        mimic_cxr_jpg_dir = glob(pattern)
        assert len(mimic_cxr_jpg_dir), f'Multiple directories matched the pattern {pattern}: {mimic_cxr_jpg_dir}. Only one is required.'
        mimic_cxr_jpg_dir = mimic_cxr_jpg_dir[0]
        
        def load_image(example):
            example['images'] = []
            for dicom_id in example['dicom_id']: 
                image_path = mimic_cxr_image_path(mimic_cxr_jpg_dir, example['subject_id'], example['study_id'], dicom_id, 'jpg')
                with open(image_path, 'rb') as f:
                    image = f.read()
                example['images'].append(image)
            return example
        
        # Standardise column names across the datasets:
        self.con.sql("ALTER TABLE studies RENAME COLUMN ViewPosition TO views;")

        dataset_dict = {}
        for split in self.splits:
            df = self.con.sql(f"FROM studies WHERE split = '{split}'").df()
            
            self.con.sql(f"DELETE FROM studies WHERE split = '{split}';")

            # Format text columns from the sections .csv file:
            for i in text_columns:
                df[i] = df[i].apply(format_text)

            dataset_dict[split] = datasets.Dataset.from_pandas(df)

            del df
            gc.collect()

            dataset_dict[split] = dataset_dict[split].map(
                load_image,
                num_proc=self.num_workers,
                keep_in_memory=False,
                load_from_cache_file=False,
                cache_file_name=os.path.join(self.database_dir, f'cache_{split}.arrow'),
                writer_batch_size=1,
            )
            dataset_dict[split].cleanup_cache_files()
            gc.collect()
            
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(os.path.join(database_dir, 'mimic_cxr_jpg_dataset'))
        
        self.con.close()


if __name__ == '__main__':

    physionet_dir = '/datasets/work/hb-mlaifsp-mm/work/data/physionet.org/files'  # Where MIMIC-CXR, MIMIC-CXR-JPG, and MIMIC-IV-ED are stored.
    database_dir = f'/scratch3/nic261/database/cxrmate2'  # Where the resultant database will be stored.

    PrepareDataset(
        physionet_dir=physionet_dir, 
        database_dir=database_dir, 
        num_workers=4,
    )()