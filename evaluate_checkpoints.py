import argparse
import inspect
import io
import random
import re
import sys
from datetime import datetime
from glob import glob

import accelerate
import pandas as pd
import torch
import transformers
from hydra import compose, initialize_config_dir
from loggers import ReportLogger, ReportTokenIdentifiersLogger, SizeLogger
from omegaconf import OmegaConf
from PIL import Image
from stages_cxrmate2 import Stages
from torchvision.transforms import v2
from transformers.feature_extraction_utils import BatchFeature
from utils import CSVTracker


class GenerateStages(Stages):

    def __init__(
        self,
        exp_trial_dir,
        test_datasets,
        database_dir='/scratch3/nic261/database/cxrmate2',
        train=False,
        validate=False,
        test=True,
        test_pretrained=True,
        limit_test_samples=None,
        test_mbatch_size=1,
        findings_and_impression_strategy='and',
        history=1,
        monitor=None,
        monitor_mode=None,
        dataloader_num_workers=5,
        prefetch_factor=2,
        is_node_0_rank_0=True,
        debug=False,
    ):
        self.exp_trial_dir = exp_trial_dir
        self.database_dir = database_dir
        self.train = train
        self.validate = validate
        self.test = test
        self.test_pretrained = test_pretrained
        self.limit_test_samples = limit_test_samples
        self.test_mbatch_size = test_mbatch_size
        self.test_datasets = test_datasets
        self.findings_and_impression_strategy = findings_and_impression_strategy
        self.history = history
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.dataloader_num_workers = dataloader_num_workers
        self.prefetch_factor = prefetch_factor
        self.is_node_0_rank_0 = is_node_0_rank_0
        self.debug = debug

        tracker = CSVTracker(self.exp_trial_dir)

        self.accelerator = accelerate.Accelerator(log_with=tracker, project_dir=self.exp_trial_dir)
        self.accelerator.init_trackers('csv')

        assert len (self.test_datasets) == 1

        if 'mimic_cxr' in self.test_datasets:
            report_dir = 'reports'
        elif 'chexpert_plus' in self.test_datasets:
            report_dir = 'reports_chexpert_plus'
        elif 'rexgradient' in self.test_datasets:
            report_dir = 'reports_rexgradient'
        csv_paths = glob(f'{self.exp_trial_dir}/metric_outputs/{report_dir}/*.csv', recursive=True)
        csv_paths = [i for i in csv_paths if 'test_reports' in i]
        
        if csv_paths:

            self.generated_reports_path = max(
                csv_paths,
                key=lambda p: datetime.strptime(
                    re.search(r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})', p).group(1),
                    "%d-%m-%Y_%H-%M-%S"
                )
            )

            if self.generated_reports_path:
                df = pd.read_csv(self.generated_reports_path)
                if len(df) > 32:
                    print('Exiting as reports already generated')
                    sys.exit()

    def __call__(self):
        self.init_dataloaders()
        self.init_model()
        self.accelerate_prepare()
        self.init_metrics()
        self.test_epoch_wrapper()

    def init_metrics(self):
        
        # MIMIC-CXR:
        self.val_report_logger = ReportLogger(exp_dir=self.exp_trial_dir, split='val_reports')
        self.val_report_ids_logger = ReportTokenIdentifiersLogger(exp_dir=self.exp_trial_dir, split='val_report_ids')

        self.test_metrics = {}

        if 'mimic_cxr' in self.test_datasets:
            self.test_report_logger = ReportLogger(exp_dir=self.exp_trial_dir, split='test_reports')
            self.test_report_ids_logger = ReportTokenIdentifiersLogger(exp_dir=self.exp_trial_dir, split='test_report_ids')
            self.test_prompt_len_logger = SizeLogger(exp_dir=self.exp_trial_dir, split='test_prompt_len')   

        # CheXpert Plus:
        if 'chexpert_plus' in self.test_datasets:
            metric_name = inspect.signature(ReportLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_chexpert_plus'
            self.test_report_logger_chexpert_plus = ReportLogger(
                exp_dir=self.exp_trial_dir, split='test_reports_chexpert_plus', metric_name=metric_name,
            )
            metric_name = inspect.signature(ReportTokenIdentifiersLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_chexpert_plus'
            self.test_report_ids_logger_chexpert_plus = ReportTokenIdentifiersLogger(
                exp_dir=self.exp_trial_dir, split='test_report_ids_chexpert_plus', metric_name=metric_name,
            )
            metric_name = inspect.signature(SizeLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_chexpert_plus'
            self.test_prompt_len_logger_chexpert_plus = SizeLogger(
                exp_dir=self.exp_trial_dir, split='test_prompt_len_chexpert_plus', metric_name=metric_name,
            ) 

        # ReXgradient-160K:
        if 'rexgradient' in self.test_datasets:
            metric_name = inspect.signature(ReportLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_rexgradient'
            self.test_report_logger_rexgradient = ReportLogger(
                exp_dir=self.exp_trial_dir, split='test_reports_rexgradient', metric_name=metric_name,
            )
            metric_name = inspect.signature(ReportTokenIdentifiersLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_rexgradient'
            self.test_report_ids_logger_rexgradient = ReportTokenIdentifiersLogger(
                exp_dir=self.exp_trial_dir, split='test_report_ids_rexgradient', metric_name=metric_name,
            )
            metric_name = inspect.signature(SizeLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_rexgradient'
            self.test_prompt_len_logger_rexgradient = SizeLogger(
                exp_dir=self.exp_trial_dir, split='test_prompt_len_rexgradient', metric_name=metric_name,
            ) 

    def dataloader_collate_functions(self):
        raise NotImplementedError
    
    def init_model(self):
        raise NotImplementedError

    def generate_sections(self, batch):
        raise NotImplementedError


class CXRMate2GenerateStages(Stages):

    def __call__(self):

        assert len (self.test_datasets) == 1

        if 'mimic_cxr' in self.test_datasets:
            report_dir = 'reports'
        elif 'chexpert_plus' in self.test_datasets:
            report_dir = 'reports_chexpert_plus'
        elif 'rexgradient' in self.test_datasets:
            report_dir = 'reports_rexgradient'
        csv_paths = glob(f'{self.exp_trial_dir}/metric_outputs/{report_dir}/*.csv', recursive=True)
        csv_paths = [i for i in csv_paths if 'test_reports' in i]

        if csv_paths:
            self.generated_reports_path = max(
                csv_paths,
                key=lambda p: datetime.strptime(
                    re.search(r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})', p).group(1),
                    "%d-%m-%Y_%H-%M-%S"
                )
            )

            if self.generated_reports_path:
                df = pd.read_csv(self.generated_reports_path)
                if len(df) > 32:
                    print('Exiting as reports already generated')
                    sys.exit()

        self.init_dataloaders()
        self.init_processor()
        self.init_model()
        self.warm_start()
        self.accelerate_prepare()
        self.init_metrics()
        self.test_epoch_wrapper()

    def init_metrics(self):
        
        # MIMIC-CXR:
        self.val_report_logger = ReportLogger(exp_dir=self.exp_trial_dir, split='val_reports')
        self.val_report_ids_logger = ReportTokenIdentifiersLogger(exp_dir=self.exp_trial_dir, split='val_report_ids')

        self.test_metrics = {}

        if 'mimic_cxr' in self.test_datasets:
            self.test_report_logger = ReportLogger(exp_dir=self.exp_trial_dir, split='test_reports')
            self.test_report_ids_logger = ReportTokenIdentifiersLogger(exp_dir=self.exp_trial_dir, split='test_report_ids')
            self.test_prompt_len_logger = SizeLogger(exp_dir=self.exp_trial_dir, split='test_prompt_len')   

        # CheXpert Plus:
        if 'chexpert_plus' in self.test_datasets:
            metric_name = inspect.signature(ReportLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_chexpert_plus'
            self.test_report_logger_chexpert_plus = ReportLogger(
                exp_dir=self.exp_trial_dir, split='test_reports_chexpert_plus', metric_name=metric_name,
            )
            metric_name = inspect.signature(ReportTokenIdentifiersLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_chexpert_plus'
            self.test_report_ids_logger_chexpert_plus = ReportTokenIdentifiersLogger(
                exp_dir=self.exp_trial_dir, split='test_report_ids_chexpert_plus', metric_name=metric_name,
            )
            metric_name = inspect.signature(SizeLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_chexpert_plus'
            self.test_prompt_len_logger_chexpert_plus = SizeLogger(
                exp_dir=self.exp_trial_dir, split='test_prompt_len_chexpert_plus', metric_name=metric_name,
            ) 

        # ReXgradient-160K:
        if 'rexgradient' in self.test_datasets:
            metric_name = inspect.signature(ReportLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_rexgradient'
            self.test_report_logger_rexgradient = ReportLogger(
                exp_dir=self.exp_trial_dir, split='test_reports_rexgradient', metric_name=metric_name,
            )
            metric_name = inspect.signature(ReportTokenIdentifiersLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_rexgradient'
            self.test_report_ids_logger_rexgradient = ReportTokenIdentifiersLogger(
                exp_dir=self.exp_trial_dir, split='test_report_ids_rexgradient', metric_name=metric_name,
            )
            metric_name = inspect.signature(SizeLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_rexgradient'
            self.test_prompt_len_logger_rexgradient = SizeLogger(
                exp_dir=self.exp_trial_dir, split='test_prompt_len_rexgradient', metric_name=metric_name,
            ) 


class MAIRA2GenerateStages(GenerateStages):

    frontal_views = ['LPO', 'RAO', 'LAO', 'AP AXIAL', 'AP RLD', 'AP LLD', 'AP', 'PA RLD', 'PA LLD', 'PA']
    lateral_views = ['SWIMMERS', 'XTABLE LATERAL', 'LL', 'LATERAL', 'Lateral']

    def dataloader_collate_functions(self):

        assert 'rexgradient' not in self.test_datasets
        
        def test_collate_fn(batch):

            study_datetime = batch[0]['study_datetime']
            current_cxr_mask = [i == study_datetime for i in batch[0]['image_datetime']]

            current_views = [j for i, j in zip(current_cxr_mask, batch[0]['views']) if i]
            prior_views = [j for i, j in zip(current_cxr_mask, batch[0]['views']) if not i]
            current_images = [j for i, j in zip(current_cxr_mask, batch[0]['images']) if i]
            prior_images = [j for i, j in zip(current_cxr_mask, batch[0]['images']) if not i]

            current_frontal_idx = random.choice([i for i, j in enumerate(current_views) if j in self.frontal_views] or [None]) 
            current_lateral_idx = random.choice([i for i, j in enumerate(current_views) if j in self.lateral_views] or [None])
            if prior_views is not None:
                prior_frontal_idx = random.choice([i for i, j in enumerate(prior_views) if j in self.frontal_views] or [None])
            else:
                prior_frontal_idx = None

            if current_frontal_idx is not None:
                current_frontal = Image.open(io.BytesIO(bytearray(current_images[current_frontal_idx])))
                current_lateral = Image.open(io.BytesIO(bytearray(current_images[current_lateral_idx]))) if current_lateral_idx is not None else None
                prior_frontal = Image.open(io.BytesIO(bytearray(prior_images[prior_frontal_idx]))) if prior_frontal_idx is not None else None

                processed = self.processor.format_and_preprocess_reporting_input(
                    current_frontal=current_frontal,
                    current_lateral=current_lateral,
                    prior_frontal=prior_frontal,
                    indication=batch[0]['indication'],
                    technique=batch[0]['technique'],
                    comparison=batch[0]['comparison'],
                    prior_report=batch[0]['prior_findings'][0] if batch[0]['prior_findings'][0] is not None else None,
                    return_tensors='pt',
                    get_grounding=False,
                )

            else:
                processed = BatchFeature({'input_ids': torch.tensor([[]])})

            processed.data['frontal_exists'] = current_frontal_idx is not None

            processed.data['findings'] = [batch[0]['findings']]
            processed.data['impression'] = [batch[0]['impression']]
            processed.data['study_id'] = [batch[0]['study_id']]

            return processed
        
        return None, test_collate_fn
    
    def init_model(self):

        self.model = transformers.AutoModelForCausalLM.from_pretrained('microsoft/maira-2', trust_remote_code=True)
        self.processor = transformers.AutoProcessor.from_pretrained('microsoft/maira-2', trust_remote_code=True)
        self.generation_config = transformers.GenerationConfig.from_pretrained('microsoft/maira-2', trust_remote_code=True)

    def generate_sections(self, batch):

        if not batch['frontal_exists']:
            findings = ''
            outputs = torch.tensor([[]])
        else:

            _ = batch.pop('frontal_exists')
            with torch.no_grad():         
                gen_model = self.model.module if hasattr(self.model, 'module') else self.model
                outputs = gen_model.generate(**batch, generation_config=self.generation_config)

            # Evaluate:
            prompt_len = batch['input_ids'].shape[-1]
            decoded_text = self.processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
            findings = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
            
        impression = ''

        return outputs, [findings], [impression], [0]


class MedGemmaGenerateStages(GenerateStages):

    def init_model(self):

        model_id = "google/medgemma-4b-it"

        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        self.processor = transformers.AutoProcessor.from_pretrained(model_id)

    def dataloader_collate_functions(self):
        
        def test_collate_fn(batch):

            images = [Image.open(io.BytesIO(bytearray(i))).convert('RGB') for i in batch[0]['images']]

            text = "You are a radiologist. Write the FINDINGS section of a chest X-ray report in standard professional language, without headings or formatting. FINDINGS:"
            if batch[0]['indication'] is not None:
                text = batch[0]['indication'] + ' ' + text
            if self.test_datasets[0] != 'rexgradient' and batch[0]['history'] is not None:
                text = batch[0]['history'] + ' ' + text
            if batch[0]['technique'] is not None:
                text = batch[0]['technique'] + ' ' + text

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist specializing in chest X-ray interpretation."}]
                },
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in images],
                        {"type": "text", "text": text},
                    ]
                }
            ]

            processed = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors='pt'
            ).to(dtype=torch.bfloat16)

            processed.data['findings'] = [batch[0]['findings']]
            processed.data['impression'] = [batch[0]['impression']]
            processed.data['study_id'] = [batch[0]['study_id']]

            return processed
        
        return None, test_collate_fn

    def generate_sections(self, batch):

        input_len = batch['input_ids'].shape[-1]

        with torch.inference_mode():
            gen_model = self.model.module if hasattr(self.model, 'module') else self.model
            generation = gen_model.generate(**batch, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]

        findings = self.processor.decode(generation, skip_special_tokens=True)
            
        impression = ''

        return torch.tensor([[]]), [findings], [impression], [0]


class CXRMateRRG24GenerateStages(GenerateStages):
    
    def init_model(self):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained('aehrc/cxrmate-rrg24')
        self.model = transformers.AutoModel.from_pretrained('aehrc/cxrmate-rrg24', trust_remote_code=True)
        self.transforms = v2.Compose(
            [
                v2.PILToTensor(),
                v2.Grayscale(num_output_channels=3),
                v2.Resize(size=self.model.config.encoder.image_size, antialias=True),
                v2.CenterCrop(size=[self.model.config.encoder.image_size]*2),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.model.config.encoder.image_mean, std=self.model.config.encoder.image_std),
            ]
        )

    def dataloader_collate_functions(self):
        
        def test_collate_fn(batch):

            images = [Image.open(io.BytesIO(bytearray(i))).convert('RGB') for i in batch[0]['images']]
            images = [self.transforms(i) for i in images] 
            images = torch.nn.utils.rnn.pad_sequence(images, batch_first=True, padding_value=0.0).unsqueeze(0)

            processed = BatchFeature({'images': images})

            processed.data['findings'] = [batch[0]['findings']]
            processed.data['impression'] = [batch[0]['impression']]
            processed.data['study_id'] = [batch[0]['study_id']]

            return processed
        
        return None, test_collate_fn

    def generate_sections(self, batch):

        output_ids = self.model.generate(
            pixel_values=batch['images'],
            max_length=512,
            num_beams=1,
            bad_words_ids=[[self.tokenizer.convert_tokens_to_ids('[NF]')], [self.tokenizer.convert_tokens_to_ids('[NI]')]],
        )
        findings, impression = self.model.split_and_decode_sections(output_ids, self.tokenizer)

        return output_ids, [findings], [impression], [0]
    

class EvalGeneratedStages(Stages):

    def __init__(
        self,
        exp_trial_dir,
        test_datasets,
        generated_reports_path=None,
        database_dir='/scratch3/nic261/database/cxrmate2',
        train=False,
        validate=False,
        test=True,
        test_pretrained=True,
        limit_test_samples=None,
        test_mbatch_size=1,
        findings_and_impression_strategy='and',
        history=1,
        monitor=None,
        monitor_mode=None,
        dataloader_num_workers=5,
        prefetch_factor=2,
        is_node_0_rank_0=True,
        debug=False,
    ):
        self.exp_trial_dir = exp_trial_dir
        self.database_dir = database_dir
        self.generated_reports_path = generated_reports_path
        self.train = train
        self.validate = validate
        self.test = test
        self.test_pretrained = test_pretrained
        self.limit_test_samples = limit_test_samples
        self.test_mbatch_size = test_mbatch_size
        self.test_datasets = test_datasets
        self.findings_and_impression_strategy = findings_and_impression_strategy
        self.history = history
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.dataloader_num_workers = dataloader_num_workers
        self.prefetch_factor = prefetch_factor
        self.is_node_0_rank_0 = is_node_0_rank_0
        self.debug = debug

        tracker = CSVTracker(self.exp_trial_dir)

        self.accelerator = accelerate.Accelerator(log_with=tracker, project_dir=self.exp_trial_dir)
        self.accelerator.init_trackers('csv')

        if self.generated_reports_path is None:

            assert len (self.test_datasets) == 1

            if 'mimic_cxr' in self.test_datasets:
                report_dir = 'reports'
            elif 'chexpert_plus' in self.test_datasets:
                report_dir = 'reports_chexpert_plus'
            elif 'rexgradient' in self.test_datasets:
                report_dir = 'reports_rexgradient'
            csv_paths = glob(f'{self.exp_trial_dir}/metric_outputs/{report_dir}/*.csv', recursive=True)
            
            self.generated_reports_path = max(
                csv_paths,
                key=lambda p: datetime.strptime(
                    re.search(r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})', p).group(1),
                    "%d-%m-%Y_%H-%M-%S"
                )
            )


    def __call__(self):
        self.init_dataloaders()
        self.init_model()
        self.accelerate_prepare()
        self.init_metrics()
        self.test_epoch_wrapper()

    def init_model(self):

        self.model = torch.nn.Identity()
        self.reports = pd.read_csv(self.generated_reports_path).fillna('')

    def dataloader_collate_functions(self):
        
        def test_collate_fn(batch):

            processed = BatchFeature(
                {
                    'findings': [batch[0]['findings']],
                    'impression': [batch[0]['impression']],
                    'study_id': [batch[0]['study_id']],                    
                    '_study_id': [batch[0]['study_id']],
                }
            )
            
            return processed
        
        return None, test_collate_fn

    def generate_sections(self, batch):

        if batch['_study_id'][0] not in self.reports['study_id'].values:
            print(f'study_id {batch["_study_id"][0]} not found in generated reports.')
        findings = self.reports[self.reports['study_id'] == batch['_study_id'][0]]['findings'].item()
        impression = self.reports[self.reports['study_id'] == batch['_study_id'][0]]['impression'].item()

        return torch.tensor([[]]), [findings], [impression], [0]


class EMNLIEvalGeneratedStages(EvalGeneratedStages):

    def init_model(self):

        self.model = torch.nn.Identity()
        self.reports = pd.read_csv(self.generated_reports_path, header=None)
        self.reports[['unk', 'findings']] = self.reports[8].str.split(' ', n=1, expand=True)
        self.reports[['study_id', 'unk']] = self.reports[0].str.split('_', n=1, expand=True)
        self.reports = self.reports[['study_id', 'findings']]
        self.reports['impression'] = ''
        self.reports['study_id'] = self.reports['study_id'].astype(int)


class MedVersaEvalGeneratedStages(EvalGeneratedStages):

    def init_model(self):

        self.model = torch.nn.Identity()
        self.reports = pd.read_csv(self.generated_reports_path)
        self.reports = self.reports.drop(columns=['study_id'])
        self.reports = self.reports.rename(columns={'case_id': 'study_id'})

        if self.test_datasets[0] != 'rexgradient':

            pattern = re.compile(r'(?is)\bfindings\s*:\s*(.*?)\s*(?:impression\s*:\s*(.*))$', re.IGNORECASE|re.DOTALL)

            def extract_sections(text):
                m = pattern.search(text or '')
                findings = m.group(1).strip() if (m and m.group(1)) else ''
                impression = m.group(2).strip() if (m and m.lastindex >= 2 and m.group(2)) else ''
                return pd.Series({'findings': findings, 'impression': impression})

            self.reports[['findings', 'impression']] = self.reports['report'].apply(extract_sections)
        else:
            self.reports['impression'] = ''
            self.reports['findings'] = self.reports['report'].str.replace('Findings:', '', regex=False)

        if self.test_datasets[0] == 'rexgradient':
            self.reports['study_id'] = self.reports['study_id'].str[37:]


class LibraEvalGeneratedStages(EvalGeneratedStages):

    def __init__(self, ids_path, **kwargs):
        super().__init__(**kwargs)
        self.ids_path = ids_path

    def init_model(self):
        self.model = torch.nn.Identity()

        if self.test_datasets[0] == 'mimic_cxr':
            with open(self.ids_path, 'r', encoding='utf-8') as f:
                ids = f.readlines()
            study_ids = [p.split(',')[0].split('/')[4][1:] for p in ids]
            with open(self.generated_reports_path, 'r', encoding='utf-8') as f:
                findings = f.readlines()
            self.reports = pd.DataFrame({'study_id': study_ids, 'findings': findings})
            self.reports['impression'] = ''
            self.reports['study_id'] = self.reports['study_id'].astype(int)

        if self.test_datasets[0] == 'chexpert_plus':
            with open(self.ids_path, 'r', encoding='utf-8') as f:
                ids = f.readlines()
            study_ids = [p.split(',')[0].split('/')[3] + '_' + p.split(',')[0].split('/')[4] for p in ids]
            with open(self.generated_reports_path, 'r', encoding='utf-8') as f:
                findings = f.readlines()
            self.reports = pd.DataFrame({'study_id': study_ids, 'findings': findings})
            self.reports['impression'] = ''
            self.reports = self.reports.drop_duplicates(subset=['study_id']).reset_index(drop=True)

        if self.test_datasets[0] == 'rexgradient':
            with open(self.ids_path, 'r', encoding='utf-8') as f:
                ids = f.readlines()
            study_ids = [p.split(',')[0].split('/')[5] for p in ids]
            with open(self.generated_reports_path, 'r', encoding='utf-8') as f:
                findings = f.readlines()
            self.reports = pd.DataFrame({'study_id': study_ids, 'findings': findings})
            self.reports['impression'] = ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model to evaluate')
    parser.add_argument('--trial', type=str, help='Trial to evaluate')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--limit_test_samples', type=int, default=None, help='Limit test samples')
    parser.add_argument('--test_set', type=str, default=None, help='Test set')
    parser.add_argument('--generate', action='store_true', help='Generate reports')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate reports')
    args = parser.parse_args()
    test_datasets = [args.test_set.strip()]
    model_key = args.model.strip()

    if model_key == 'CXRMate-2':
        config_dir = '/datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/cxrmate2/config'
        config_name = '000_sft.yaml'
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = OmegaConf.to_container(
                compose(config_name=config_name), resolve=True
            )
        config.pop('test_datasets', None)

        # if args.generate:
        #     CXRMate2GenerateStages(
        #         exp_trial_dir=f'/datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/scratch3/experiments/cxrmate2/final/001_grpo/trial_{args.trial}',
        #         trial=int(args.trial),
        #         test=True,
        #         submit=False,
        #         debug=args.debug,
        #         limit_test_samples=args.limit_test_samples,
        #         test_datasets=test_datasets,
        #         **config,
        #     )()
        # elif args.evaluate:
        #     EvalGeneratedStages(
        #         exp_trial_dir=f'/datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/scratch3/experiments/cxrmate2/final/001_grpo/trial_{args.trial}',
        #         debug=args.debug,
        #         limit_test_samples=args.limit_test_samples,
        #         test_datasets=test_datasets,
        #     )()

        if args.generate:
            CXRMate2GenerateStages(
                exp_trial_dir=f'/datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/scratch3/experiments/cxrmate2/final/002_grpo_rev_a/trial_{args.trial}',
                trial=int(args.trial),
                test=True,
                submit=False,
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
                **config,
            )()
        elif args.evaluate:
            EvalGeneratedStages(
                exp_trial_dir=f'/datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/scratch3/experiments/cxrmate2/final/002_grpo_rev_a/trial_{args.trial}',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()

    elif model_key == 'CXRMate-RRG24':
        exp_trial_dir='/datasets/work/hb-mlaifsp-mm/work/repositories/25_cxrmate2/scratch3/experiments/cxrmate2/final/cxrmate_rrg24/trial_0'
        if args.generate:
            CXRMateRRG24GenerateStages(
                exp_trial_dir=exp_trial_dir,
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()
        elif args.evaluate:
            EvalGeneratedStages(
                exp_trial_dir=exp_trial_dir,
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()

    elif model_key == 'MAIRA-2':
        if args.generate:
            MAIRA2GenerateStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/maira2/trial_0',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()
        elif args.evaluate:
            EvalGeneratedStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/maira2/trial_0',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()

    elif model_key == 'MoERad':
        pass

    elif model_key == 'MedVersa':
        if args.test_set == 'mimic_cxr':
            MedVersaEvalGeneratedStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/medversa/trial_0',
                generated_reports_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/medversa/mimic-cxr.csv',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()
        elif args.test_set == 'chexpert_plus':
            MedVersaEvalGeneratedStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/medversa/trial_0',
                generated_reports_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/medversa/chexpert_plus.csv',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()
        elif args.test_set == 'rexgradient':
            MedVersaEvalGeneratedStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/medversa/trial_0',
                generated_reports_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/medversa/ReXGradient_Publictest_Findings.csv',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()

    elif model_key == 'Libra':
        if args.test_set == 'mimic_cxr':
            LibraEvalGeneratedStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/libra/trial_0',
                generated_reports_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/libra/RadEval_mimic_chexpert_rexgradient/libra.v1.7b.mimic-cxr.test.findings.tok',
                ids_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/libra/mimic.test.findings.image.tok',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()
        elif args.test_set == 'chexpert_plus':
            LibraEvalGeneratedStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/libra/trial_0',
                generated_reports_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/libra/RadEval_mimic_chexpert_rexgradient/libra.v1.7b.chexpert-plus.valid.findings.tok',
                ids_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/libra/chexpert.valid.findings.image.tok',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()
        elif args.test_set == 'rexgradient':
            LibraEvalGeneratedStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/libra/trial_0',
                generated_reports_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/libra/RadEval_mimic_chexpert_rexgradient/libra.v1.7b.rexgradient.test.findings.tok',
                ids_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/libra/rexgradient.test.image.tok',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()

    elif model_key == 'MedGemma':
        if args.generate:
            MedGemmaGenerateStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/medgemma/trial_0',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()
        elif args.evaluate:
            EvalGeneratedStages(
                exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/medgemma/trial_0',
                debug=args.debug,
                limit_test_samples=args.limit_test_samples,
                test_datasets=test_datasets,
            )()

    elif model_key == 'CXRMate-ED':
        EvalGeneratedStages(
            exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/cxrmate_ed/trial_0',
            generated_reports_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/cxrmate_ed/mimic_cxr_test_set_generated_reports.csv',
            debug=args.debug,
            limit_test_samples=args.limit_test_samples,
            test_datasets=test_datasets,
        )()
    
    elif model_key == 'CXRMate':
        EvalGeneratedStages(
            exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/cxrmate/trial_0',
            generated_reports_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/cxrmate/test_reports_epoch-0_24-07-2023_13-09-03.csv',
            debug=args.debug,
            limit_test_samples=args.limit_test_samples,
            test_datasets=test_datasets,
        )()

    elif model_key == 'EMNLI':
        EMNLIEvalGeneratedStages(
            exp_trial_dir='/scratch3/nic261/experiments/cxrmate2/final/emnli/trial_0',
            generated_reports_path='/datasets/work/hb-mlaifsp-mm/work/data/generated_radiology_reports/emnli/test_0-0_samples.txt',
            debug=args.debug,
            limit_test_samples=args.limit_test_samples,
            test_datasets=test_datasets,
        )()
    
    else:
        raise SystemExit(f"Unknown model '{args.model}'. Available: MAIRA-2")
        raise SystemExit(f"Unknown model '{args.model}'. Available: MAIRA-2")
