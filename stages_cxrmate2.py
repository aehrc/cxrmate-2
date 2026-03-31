import inspect
import math
import os
import random
import re
from typing import Optional, Union

import accelerate
import datasets
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from base_stages import BaseStages
from configuration_cxrmate2 import CXRMate2Config
from dataset import CXRMate2Dataset
from huggingface_hub import upload_file
from loggers import ReportLogger, ReportTokenIdentifiersLogger, SizeLogger
from modelling_cxrmate2 import CXRMate2ForConditionalGeneration
from processing_cxrmate2 import CXRMate2Processor
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from utils import rename_added_tokens


class Stages(BaseStages):
    
    def __init__(
        self, 
        database_dir: str,
        history: int,
        train_mbatch_size: int,
        val_mbatch_size: int,
        test_mbatch_size: int,
        optimiser_kwargs: dict,
        num_q_adapter_queries: int, 
        num_q_adapter_layers: int, 
        language_model_ckpt_alias: str,
        max_generated_tokens: int,
        max_train_images_per_study: int,
        num_warmup_steps: int,
        num_cycles: int,
        set_special_tokens: dict,
        train_datasets: list,
        test_datasets: Union[list, str],
        findings_and_impression_strategy: str,
        preferences: Optional[str] = None,
        warm_start_ckpt_dir: Optional[str] = None,
        enable_gradient_checkpointing: bool = False,
        other_exp_dir: Optional[str] = None,
        rename_state_dict_keys: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.database_dir = database_dir
        self.history = history
        self.train_mbatch_size = train_mbatch_size
        self.val_mbatch_size = val_mbatch_size
        self.test_mbatch_size = test_mbatch_size
        self.optimiser_kwargs = optimiser_kwargs
        self.num_q_adapter_queries = num_q_adapter_queries
        self.num_q_adapter_layers = num_q_adapter_layers
        self.language_model_ckpt_alias = language_model_ckpt_alias
        self.max_generated_tokens = max_generated_tokens
        self.max_train_images_per_study = max_train_images_per_study
        self.num_warmup_steps = num_warmup_steps
        self.num_cycles = num_cycles
        self.set_special_tokens = set_special_tokens
        self.train_datasets = train_datasets
        self.test_datasets = [test_datasets] if isinstance(test_datasets, str) else test_datasets
        self.findings_and_impression_strategy = findings_and_impression_strategy
        self.preferences = preferences
        self.warm_start_ckpt_dir = warm_start_ckpt_dir
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.other_exp_dir = other_exp_dir
        self.rename_state_dict_keys = rename_state_dict_keys

        assert all(i in ['mimic_cxr', 'mimic_cxr_dpo', 'chexpert_plus', 'rexgradient'] for i in self.test_datasets), f'test_datasets must be a list containing any of "mimic_cxr", "mimic_cxr_dpo", "chexpert_plus", "rexgradient", not {self.test_datasets}.'

    def init_processor(self):

        # Model:
        token_type_to_token = {
            'image': '<|reserved_special_token_16|>',
            'indication': '<|reserved_special_token_17|>',
            'history': '<|reserved_special_token_18|>',
            'comparison': '<|reserved_special_token_19|>',
            'technique': '<|reserved_special_token_20|>',
            'findings': '<|reserved_special_token_21|>',
            'impression': '<|reserved_special_token_22|>',
            'prior_image': '<|reserved_special_token_23|>',
            'prior_findings': '<|reserved_special_token_24|>',
            'prior_impression': '<|reserved_special_token_25|>',
        }
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.language_model_ckpt_alias)          
        for k, v in self.set_special_tokens.items():
            assert int(re.findall(r'\d+', v)[-1]) >= 64, f'The reserved special token number must be >= 64, not {v}.'
            setattr(tokenizer, k, v)
        self.print('Description, Special token, Index')
        for k, v in tokenizer.special_tokens_map.items():
            if k != 'additional_special_tokens':
                self.print(f'{k}, {v}, {getattr(tokenizer, k + "_id")}')
            else:
                for i, j in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids):
                    self.print(f'additional_special_token, {i}, {j}')

        self.processor = CXRMate2Processor(
            image_processor=transformers.AutoImageProcessor.from_pretrained('microsoft/rad-dino-maira-2', trust_remote_code=True),
            tokenizer=tokenizer,
            token_type_to_token=token_type_to_token,
            max_generated_tokens=self.max_generated_tokens,
            embeddings_per_image=self.num_q_adapter_queries,
            image_token='<|reserved_special_token_0|>',
            max_train_images_per_study=self.max_train_images_per_study,
            generate_findings_token='<|reserved_special_token_1|>',
            generate_impression_token='<|reserved_special_token_2|>',
            mimic_cxr_normalisation=False,  # The images are normalised in their prepare python scripts if needed.
        )

    def init_model(self):

        text_config = transformers.AutoConfig.from_pretrained(
            self.language_model_ckpt_alias, 
            trust_remote_code=True,
            torch_dtype='float32',
        )

        self.generation_config = transformers.GenerationConfig(
            max_new_tokens=self.max_generated_tokens,
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )  # do_sample=False by default.

        """
        Use _attn_implementation='eager': https://github.com/huggingface/transformers/blob/b673c16cad81c71f70903a9a63f5b5f06014aa9e/src/transformers/models/llama/modeling_llama.py#L675
        
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype) sets all fully masked rows to unmasked, causing NaN loss when 'sdpa' is used. It is not compatible with CXRMate2's 4D attention mask.
        """

        config = CXRMate2Config(
            vision_config=transformers.AutoConfig.from_pretrained('microsoft/rad-dino-maira-2', trust_remote_code=True),
            text_config=text_config,
            permute_encoder_last_hidden_state=True,
            time_delta_encoder_intermediate_size=2048,
            num_q_adapter_queries=self.num_q_adapter_queries,
            num_q_adapter_layers=self.num_q_adapter_layers,
            num_q_adapter_positions=self.num_q_adapter_queries + 1369,
            sep_token_id=self.processor.tokenizer.sep_token_id, 
            bos_token_id=self.processor.tokenizer.bos_token_id,
            findings_token_type_id=self.processor.tokenizer.convert_tokens_to_ids(self.processor.token_type_to_token['findings']),
            impression_token_type_id=self.processor.tokenizer.convert_tokens_to_ids(self.processor.token_type_to_token['impression']),
            missing_time_delta_token_id=self.processor.tokenizer.convert_tokens_to_ids('<|reserved_special_token_4|>'),
            image_token_index=self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token),
            _attn_implementation = 'eager',
            generate_findings_token_id=self.processor.tokenizer.convert_tokens_to_ids(self.processor.generate_findings_token),
            generate_impression_token_id=self.processor.tokenizer.convert_tokens_to_ids(self.processor.generate_impression_token),
        )
        self.model = CXRMate2ForConditionalGeneration(config)
                      
        for p in self.model.vision_tower.parameters():
            p.requires_grad = False

        if self.enable_gradient_checkpointing:
            self.model.language_model.gradient_checkpointing_enable()
                  
    def warm_start(self):
        if self.warm_start_ckpt_dir is None and self.other_exp_dir is None:
            self.print('No warm start checkpoint directory provided, warm starting from pre-trained checkpoints.')
            self.model.vision_tower.load_state_dict(transformers.AutoBackbone.from_pretrained('microsoft/rad-dino-maira-2', config=self.model.config.vision_config).state_dict())
            state_dict = transformers.AutoModelForCausalLM.from_pretrained(
                self.language_model_ckpt_alias,
            ).state_dict()
            self.model.language_model.load_state_dict(state_dict)

        elif self.other_exp_dir:
            assert self.warm_start_ckpt_dir is None, 'Cannot provide both warm_start_ckpt_dir and other_exp_dir.'
            other_exp_trial_dir = os.path.join(self.other_exp_dir, f'trial_{self.trial}')
            self.warm_start_ckpt_dir = os.path.join(self.get_best_ckpt(other_exp_trial_dir), 'evaluate')

        if self.warm_start_ckpt_dir:
            self.load_safetensors_state_dict(self.model, self.warm_start_ckpt_dir, self.rename_state_dict_keys)
                
    def dataloader_collate_functions(self):
        def train_collate_fn(batch):
            if isinstance(batch, list):
                keys = set().union(*(d.keys() for d in batch))
                batch = {j: [i.setdefault(j, None) for i in batch] for j in keys}
                batch = {k: torch.stack(v) if isinstance(v[0], torch.Tensor) else v for k, v in batch.items()}
                    
            processed = self.processor(train=True, **batch)
                        
            return processed      
            
        def test_collate_fn(batch):
            if isinstance(batch, list):
                keys = set().union(*(d.keys() for d in batch))
                batch = {j: [i.setdefault(j, None) for i in batch] for j in keys}
                batch = {k: torch.stack(v) if isinstance(v[0], torch.Tensor) else v for k, v in batch.items()}
                        
            processed = self.processor(train=False, **batch)
            
            processed.data['findings'] = batch['findings']
            processed.data['impression'] = batch['impression']     
            processed.data['study_id'] = batch['study_id']
            
            return processed
        
        return train_collate_fn, test_collate_fn

    def init_dataloaders(self):
        
        # Dataset:
        dataset = datasets.load_from_disk(os.path.join(self.database_dir, 'mimic_cxr_jpg_dataset'))
        if (self.train and 'chexpert_plus' in self.train_datasets) or (self.test and 'chexpert_plus' in self.test_datasets):
            dataset_chexpert_plus = datasets.load_from_disk(os.path.join(self.database_dir, 'chexpert_plus_dataset'))
        if (self.train and 'rexgradient' in self.train_datasets) or (self.test and 'rexgradient' in self.test_datasets):
            dataset_rexgradient = datasets.load_from_disk(os.path.join(self.database_dir, 'rexgradient_160k_dataset'))

        train_collate_fn, test_collate_fn = self.dataloader_collate_functions()

        if self.train:
            
            train_datasets = []

            # MIMIC-CXR:
            if 'mimic_cxr' in self.train_datasets:
                train_set = dataset['train']

                df = pd.DataFrame({'study_id': train_set['study_id'], 'findings': train_set['findings'], 'impression': train_set['impression']})

                if self.findings_and_impression_strategy == 'or':
                    indices = df[df[['findings', 'impression']].isnull().all(axis=1)].index.tolist()  # Consider studies with findings OR impression section.
                elif self.findings_and_impression_strategy == 'and':
                    indices = df[df[['findings', 'impression']].isnull().any(axis=1)].index.tolist()  # Consider studies with findings AND impression section.

                indices = list(set(range(len(train_set))) - set(indices))

                train_set = CXRMate2Dataset(train_set, self.history)
                train_set = Subset(train_set, indices)
                train_datasets.append(train_set)
                self.print(f'No. of MIMIC-CXR training examples: {train_set.__len__()}.')
                self.accelerator.log(
                    {
                        'stage': 'MIMIC-CXR train', 
                        'epoch': 0, 
                        'step': 0,
                        'len': train_set.__len__(),
                    },
                    step=0,
                )

            # CheXpert Plus (history section exists as 'history' alread in the dataset):
            if 'chexpert_plus' in self.train_datasets:
                train_set_chexpert_plus = dataset_chexpert_plus['train']     
                df = pd.DataFrame({'findings': train_set_chexpert_plus['findings'], 'impression': train_set_chexpert_plus['impression']})
                if self.findings_and_impression_strategy == 'or':
                    indices = df[df[['findings', 'impression']].isnull().all(axis=1)].index.tolist()  # Consider studies with findings OR impression section.
                elif self.findings_and_impression_strategy == 'and':
                    indices = df[df[['findings', 'impression']].isnull().any(axis=1)].index.tolist()  # Consider studies with findings AND impression section.
                indices = list(set(range(len(train_set_chexpert_plus))) - set(indices))
                train_set_chexpert_plus = CXRMate2Dataset(train_set_chexpert_plus, self.history)
                train_set_chexpert_plus = Subset(train_set_chexpert_plus, indices)
                train_datasets.append(train_set_chexpert_plus)
                self.print(f'No. of CheXpert Plus training examples: {train_set_chexpert_plus.__len__()}.')
                self.accelerator.log(
                    {
                        'stage': 'CheXpert Plus train', 
                        'epoch': 0, 
                        'step': 0,
                        'len': train_set_chexpert_plus.__len__(),
                    },
                    step=0,
                )

            # ReXgradient-160K:
            if 'rexgradient' in self.train_datasets:
                train_set_rexgradient = dataset_rexgradient['train']
                df = pd.DataFrame({'findings': train_set_rexgradient['findings'], 'impression': train_set_rexgradient['impression']})
                if self.findings_and_impression_strategy == 'or':
                    indices = df[df[['findings', 'impression']].isnull().all(axis=1)].index.tolist()  # Consider studies with findings OR impression section.
                elif self.findings_and_impression_strategy == 'and':
                    indices = df[df[['findings', 'impression']].isnull().any(axis=1)].index.tolist()  # Consider studies with findings AND impression section.
                indices = list(set(range(len(train_set_rexgradient))) - set(indices))
                train_set_rexgradient = CXRMate2Dataset(train_set_rexgradient, self.history)
                train_set_rexgradient = Subset(train_set_rexgradient, indices)
                train_datasets.append(train_set_rexgradient)
                self.print(f'No. of ReXgradient-160K training examples: {train_set_rexgradient.__len__()}.')
                self.accelerator.log(
                    {
                        'stage': 'ReXgradient-160K train', 
                        'epoch': 0, 
                        'step': 0,
                        'len': train_set_rexgradient.__len__(),
                    },
                    step=0,
                )

            # Concatenate:
            train_set = ConcatDataset(train_datasets)  # ConcatDataset forces a Hugging Face dataset to process one example at a time (therefore, CXRMate2Dataset is changed to reflect this instead of handling a batch).
        
            if self.limit_train_samples:  # For debugging.
                train_set = Subset(train_set, random.sample(range(len(train_set)), self.limit_train_samples))
            self.train_dataloader = DataLoader(
                    train_set,
                    batch_size=self.train_mbatch_size,
                    num_workers=self.dataloader_num_workers,
                    shuffle=True,
                    prefetch_factor=self.prefetch_factor,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                )           
            self.print(f'No. of training examples: {train_set.__len__()}.')
            self.accelerator.log(
                {
                    'stage': 'Combined train', 
                    'epoch': 0, 
                    'step': 0,
                    'len': train_set.__len__(),
                },
                step=0,
            )
            
        if self.validate:

            val_set = dataset['validate']
            df = pd.DataFrame({'study_id': val_set['study_id'], 'findings': val_set['findings'], 'impression': val_set['impression']})
            if self.findings_and_impression_strategy == 'or':
                indices = df[df[['findings', 'impression']].isnull().all(axis=1)].index.tolist()  # Consider studies with findings OR impression section.
            elif self.findings_and_impression_strategy == 'and':
                indices = df[df[['findings', 'impression']].isnull().any(axis=1)].index.tolist()  # Consider studies with findings AND impression section.

            val_set = CXRMate2Dataset(val_set, self.history)    
            indices = list(set(range(len(val_set))) - set(indices))
            val_set = ConcatDataset([Subset(val_set, indices)])

            if self.limit_val_samples:  # For debugging.
                val_set = Subset(val_set, random.sample(range(len(val_set)), self.limit_val_samples))
            self.val_dataloader = DataLoader(
                    val_set,
                    batch_size=self.val_mbatch_size,
                    num_workers=self.dataloader_num_workers,
                    shuffle=False,
                    prefetch_factor=self.prefetch_factor,
                    collate_fn=test_collate_fn,
                    pin_memory=True,
                )
            self.print(f'No. of validation examples: {val_set.__len__()}.')
            self.accelerator.log(
                {
                    'stage': 'MIMIC-CXR val', 
                    'epoch': 0, 
                    'step': 0,
                    'len': val_set.__len__(),
                },
                step=0,
            )

        if self.test:

            self.test_dataloaders = []

            # MIMIC-CXR:
            if 'mimic_cxr' in self.test_datasets:
                test_set = dataset['test']
                df = pd.DataFrame({'study_id': test_set['study_id'], 'findings': test_set['findings'], 'impression': test_set['impression']})
                if self.findings_and_impression_strategy == 'or':
                    indices = df[df[['findings', 'impression']].isnull().all(axis=1)].index.tolist()  # Consider studies with findings OR impression section.
                elif self.findings_and_impression_strategy == 'and':
                    indices = df[df[['findings', 'impression']].isnull().any(axis=1)].index.tolist()  # Consider studies with findings AND impression section.

                test_set = CXRMate2Dataset(test_set, self.history)  
                indices = list(set(range(len(test_set))) - set(indices))  
                test_set = ConcatDataset([Subset(test_set, indices)])
                
                if self.limit_test_samples:  # For debugging.
                    test_set = Subset(test_set, random.sample(range(len(test_set)), self.limit_test_samples))
                test_dataloader = DataLoader(
                        test_set,
                        batch_size=self.test_mbatch_size,
                        num_workers=self.dataloader_num_workers, 
                        shuffle=False,
                        prefetch_factor=self.prefetch_factor,
                        collate_fn=test_collate_fn,
                        pin_memory=True,
                    )            
                self.print(f'No. of MIMIC-CXR test examples: {test_set.__len__()}.')
                self.accelerator.log(
                    {
                        'stage': 'MIMIC-CXR test', 
                        'epoch': 0, 
                        'step': 0,
                        'len': test_set.__len__(),
                    },
                    step=0,
                )
                self.test_dataloaders.append(test_dataloader)

            # MIMIC-CXR DPO:
            if 'mimic_cxr_dpo' in self.test_datasets:

                # Load radiologist preferences:
                preferences = pd.read_json(self.preferences)

                test_set_mimic_cxr_dpo = dataset['test']
                df = pd.DataFrame({'study_id': test_set_mimic_cxr_dpo['study_id'], 'findings': test_set_mimic_cxr_dpo['findings'], 'impression': test_set_mimic_cxr_dpo['impression']})
                df = df[~df['study_id'].isin(preferences['study_id'])]

                if self.findings_and_impression_strategy == 'or':
                    indices = df[df[['findings', 'impression']].isnull().all(axis=1)].index.tolist()  # Consider studies with findings OR impression section.
                elif self.findings_and_impression_strategy == 'and':
                    indices = df[df[['findings', 'impression']].isnull().any(axis=1)].index.tolist()  # Consider studies with findings AND impression section.

                test_set_mimic_cxr_dpo = CXRMate2Dataset(test_set_mimic_cxr_dpo, self.history)  
                indices = list(set(range(len(test_set_mimic_cxr_dpo))) - set(indices))  
                test_set_mimic_cxr_dpo = ConcatDataset([Subset(test_set_mimic_cxr_dpo, indices)])
                
                if self.limit_test_samples:  # For debugging.
                    test_set_mimic_cxr_dpo = Subset(test_set_mimic_cxr_dpo, random.sample(range(len(test_set_mimic_cxr_dpo)), self.limit_test_samples))
                test_dataloader_mimic_cxr_dpo = DataLoader(
                        test_set_mimic_cxr_dpo,
                        batch_size=self.test_mbatch_size,
                        num_workers=self.dataloader_num_workers, 
                        shuffle=False,
                        prefetch_factor=self.prefetch_factor,
                        collate_fn=test_collate_fn,
                        pin_memory=True,
                    )            
                self.print(f'No. of MIMIC-CXR DPO test examples: {test_set_mimic_cxr_dpo.__len__()}.')
                self.accelerator.log(
                    {
                        'stage': 'MIMIC-CXR DPO test', 
                        'epoch': 0, 
                        'step': 0,
                        'len': test_set_mimic_cxr_dpo.__len__(),
                    },
                    step=0,
                )
                self.test_dataloaders.append(test_dataloader_mimic_cxr_dpo)

            # CheXpert Plus:
            if 'chexpert_plus' in self.test_datasets:
                test_set_chexpert_plus = dataset_chexpert_plus['valid']     
                df = pd.DataFrame({'findings': test_set_chexpert_plus['findings'], 'impression': test_set_chexpert_plus['impression']})
                if self.findings_and_impression_strategy == 'or':
                    indices = df[df[['findings', 'impression']].isnull().all(axis=1)].index.tolist()  # Consider studies with findings OR impression section.
                elif self.findings_and_impression_strategy == 'and':
                    indices = df[df[['findings', 'impression']].isnull().any(axis=1)].index.tolist()  # Consider studies with findings AND impression section.
                indices = list(set(range(len(test_set_chexpert_plus))) - set(indices))
                test_set_chexpert_plus = CXRMate2Dataset(test_set_chexpert_plus, self.history)
                test_set_chexpert_plus = ConcatDataset([Subset(test_set_chexpert_plus, indices)])
                
                if self.limit_test_samples:  # For debugging.
                    test_set_chexpert_plus = Subset(test_set_chexpert_plus, random.sample(range(len(test_set_chexpert_plus)), self.limit_test_samples))
                test_dataloader_chexpert_plus = DataLoader(
                        test_set_chexpert_plus,
                        batch_size=self.test_mbatch_size,
                        num_workers=self.dataloader_num_workers,  # num_workers > 0 not working, at least with FSDP.
                        shuffle=False,
                        prefetch_factor=self.prefetch_factor,  # Has to be None when num_workers = 0.
                        collate_fn=test_collate_fn,
                        pin_memory=True,
                    )
                self.print(f'No. of CheXpert Plus test examples: {test_dataloader_chexpert_plus.__len__()}.')
                self.accelerator.log(
                    {
                        'stage': 'CheXpert Plus test', 
                        'epoch': 0, 
                        'step': 0,
                        'len': test_set_chexpert_plus.__len__(),
                    },
                    step=0,
                )
                self.test_dataloaders.append(test_dataloader_chexpert_plus)

            # ReXgradient-160K:
            if 'rexgradient' in self.test_datasets:
                test_set_rexgradient = dataset_rexgradient['test']

                df = pd.DataFrame({'findings': test_set_rexgradient['findings'], 'impression': test_set_rexgradient['impression']})
                if self.findings_and_impression_strategy == 'or':
                    indices = df[df[['findings', 'impression']].isnull().all(axis=1)].index.tolist()  # Consider studies with findings OR impression section.
                elif self.findings_and_impression_strategy == 'and':
                    indices = df[df[['findings', 'impression']].isnull().any(axis=1)].index.tolist()  # Consider studies with findings AND impression section.
                indices = list(set(range(len(test_set_rexgradient))) - set(indices))
                test_set_rexgradient = CXRMate2Dataset(test_set_rexgradient, self.history)
                test_set_rexgradient = ConcatDataset([Subset(test_set_rexgradient, indices)])

                if self.limit_test_samples:  # For debugging.
                    test_set_rexgradient = Subset(test_set_rexgradient, random.sample(range(len(test_set_rexgradient)), self.limit_test_samples))
                test_dataloader_rexgradient = DataLoader(
                        test_set_rexgradient,
                        batch_size=self.test_mbatch_size,
                        num_workers=self.dataloader_num_workers,
                        shuffle=False,
                        prefetch_factor=self.prefetch_factor,
                        collate_fn=test_collate_fn,
                        pin_memory=True,
                    )
                self.print(f'No. of ReXgradient-160K test examples: {test_dataloader_rexgradient.__len__()}.')
                self.accelerator.log(
                    {
                        'stage': 'ReXgradient-160K Plus test', 
                        'epoch': 0, 
                        'step': 0,
                        'len': test_set_rexgradient.__len__(),
                    },
                    step=0,
                )
                self.test_dataloaders.append(test_dataloader_rexgradient)

    def accelerate_prepare(self):

        assert self.model is not None, "The model attribute is not defined. Please ensure `self.model` is set correctly."

        if not self.train and self.accelerator.state.distributed_type.name == 'FSDP':

            raise NotImplementedError(
                "FSDP is currently not supported for inference-only runs. Please set `train=True` in the configuration."
            )

            # self.model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.model)
            # self.model = self.accelerator.prepare_model(self.model)
            if (self.validate and self.train) or (self.validate and self.validate_ckpt_dir):
                self.val_dataloader = self.accelerator.prepare_data_loader(self.val_dataloader)
                
            if self.test:
                self.test_dataloader = self.accelerator.prepare_data_loader(self.test_dataloader)
            
            return

        to_prepare = [self.model]

        if self.train:
            to_prepare.extend([self.optimiser, self.train_dataloader])
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                to_prepare.append(self.scheduler)
                
        if self.validate:
            to_prepare.append(self.val_dataloader)
            
        if self.test:
            for test_dataloader in self.test_dataloaders:
                to_prepare.append(test_dataloader)

        prepared = self.accelerator.prepare(*to_prepare)

        self.model = prepared if len(to_prepare) == 1 else prepared[0]
        index = 1

        if self.train:
            self.optimiser = prepared[index]
            self.train_dataloader = prepared[index + 1]
            index += 2

            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler = prepared[index]
                index += 1

        if self.validate:
            self.val_dataloader = prepared[index]
            index += 1

        if self.test:
            for i in range(len(self.test_dataloaders)):
                self.test_dataloaders[i] = prepared[index]
                index += 1

    def init_metrics(self):
        
        # MIMIC-CXR:
        self.val_metrics = self.metric_suite('findings', 'val')
        self.val_metrics = {**self.val_metrics, **self.metric_suite('impression', 'val')}
        self.val_report_logger = ReportLogger(exp_dir=self.exp_trial_dir, split='val_reports')
        self.val_report_ids_logger = ReportTokenIdentifiersLogger(exp_dir=self.exp_trial_dir, split='val_report_ids')

        self.test_metrics = {}

        # MIMIC-CXR:
        if 'mimic_cxr' in self.test_datasets:
            self.test_metrics = {**self.test_metrics, **self.metric_suite('findings', 'test')}
            self.test_metrics = {**self.test_metrics, **self.metric_suite('impression', 'test')}
            self.test_report_logger = ReportLogger(exp_dir=self.exp_trial_dir, split='test_reports')
            self.test_report_ids_logger = ReportTokenIdentifiersLogger(exp_dir=self.exp_trial_dir, split='test_report_ids')
            self.test_prompt_len_logger = SizeLogger(exp_dir=self.exp_trial_dir, split='test_prompt_len')   

        # MIMIC-CXR DPO:
        if 'mimic_cxr_dpo' in self.test_datasets:
            self.test_metrics = {**self.test_metrics, **self.metric_suite('findings', 'test', 'mimic_cxr_dpo')}
            self.test_metrics = {**self.test_metrics, **self.metric_suite('impression', 'test', 'mimic_cxr_dpo')}

            metric_name = inspect.signature(ReportLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_mimic_cxr_dpo'
            self.test_report_logger_mimic_cxr_dpo = ReportLogger(
                exp_dir=self.exp_trial_dir, split='test_reports_mimic_cxr_dpo', metric_name=metric_name,
            )
            metric_name = inspect.signature(ReportTokenIdentifiersLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_mimic_cxr_dpo'
            self.test_report_ids_logger_mimic_cxr_dpo = ReportTokenIdentifiersLogger(
                exp_dir=self.exp_trial_dir, split='test_report_ids_mimic_cxr_dpo', metric_name=metric_name,
            )
            metric_name = inspect.signature(SizeLogger).parameters.get('metric_name').default
            metric_name = f'{metric_name}_mimic_cxr_dpo'
            self.test_prompt_len_logger_mimic_cxr_dpo = SizeLogger(
                exp_dir=self.exp_trial_dir, split='test_prompt_len_mimic_cxr_dpo', metric_name=metric_name,
            ) 

        # CheXpert Plus:
        if 'chexpert_plus' in self.test_datasets:
            self.test_metrics = {**self.test_metrics, **self.metric_suite('findings', 'test', 'chexpert_plus')}
            self.test_metrics = {**self.test_metrics, **self.metric_suite('impression', 'test', 'chexpert_plus')}

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
            self.test_metrics = {**self.test_metrics, **self.metric_suite('findings', 'test', 'rexgradient')}
            # self.test_metrics = {**self.test_metrics, **self.metric_suite('impression', 'test', 'rexgradient')}  # This test set is massive; thus avoiding impression evaluation.
        
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
        
    def metric_suite(self, section, stage, test_set=None):

        from metrics.arn import AbsenceOfRepeatedNGramesMetric
        from metrics.bertscore import BERTScoreRoBERTaLargeMetric
        from metrics.bleu import BLEUMetric
        from metrics.chexbert import CheXbertMetric
        from metrics.cxrbert import CXRBERTMetric
        from metrics.green import GREENMetric
        from metrics.radeval_bertscore import RadEvalBERTScoreMetric
        from metrics.radgraph_xl import RadGraphXLMetric
        from metrics.ratescore import RaTEScoreMetric
        from metrics.rouge_l import ROUGELMetric
        from metrics.srr import SRRMetric
        
        metrics = {}
        
        section = f'{test_set}_{section}' if test_set else section

        # BLEU metric:
        metric_name = inspect.signature(BLEUMetric).parameters.get('metric_name').default
        metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
        metrics[f'{stage}_{section}_bleu'] = BLEUMetric(
            split=f'{stage}_{section}',
            exp_dir=self.exp_trial_dir,
            metric_name=metric_name,
        )

        # RaTEScore metric:
        metric_name = inspect.signature(RaTEScoreMetric).parameters.get('metric_name').default
        metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
        metrics[f'{stage}_{section}_ratescore'] = RaTEScoreMetric(
            split=f'{stage}_{section}',
            exp_dir=self.exp_trial_dir,
            metric_name=metric_name,
            accelerator=self.accelerator,
        )

        if stage == 'test' and self.accelerator.state.distributed_type.name != 'FSDP':
            # Green metric:
            metric_name = inspect.signature(GREENMetric).parameters.get('metric_name').default
            metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
            metrics[f'{stage}_{section}_green'] = GREENMetric(
                split=f'{stage}_{section}',
                exp_dir=self.exp_trial_dir,
                metric_name=metric_name,
                accelerator=self.accelerator,
            )

        # ARN metric:
        metric_name = inspect.signature(AbsenceOfRepeatedNGramesMetric).parameters.get('metric_name').default
        metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
        metrics[f'{stage}_{section}_arn'] = AbsenceOfRepeatedNGramesMetric(
            split=f'{stage}_{section}',
            exp_dir=self.exp_trial_dir,
            metric_name=metric_name,
            accelerator=self.accelerator,
        )

        # ROUGE-L metric:
        metric_name = inspect.signature(ROUGELMetric).parameters.get('metric_name').default
        metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
        metrics[f'{stage}_{section}_rouge_l'] = ROUGELMetric(
            split=f'{stage}_{section}',
            exp_dir=self.exp_trial_dir,
            metric_name=metric_name,
        )

        # CheXbert metric:
        metric_name = inspect.signature(CheXbertMetric).parameters.get('metric_name').default
        metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
        metrics[f'{stage}_{section}_chexbert'] = CheXbertMetric(
            exp_dir=self.exp_trial_dir,
            split=f'{stage}_{section}',
            metric_name=metric_name,
            accelerator=self.accelerator,
        )

        # SRR metric:
        metric_name = inspect.signature(SRRMetric).parameters.get('metric_name').default
        metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
        metrics[f'{stage}_{section}_srr'] = SRRMetric(
            exp_dir=self.exp_trial_dir,
            split=f'{stage}_{section}',
            metric_name=metric_name,
            accelerator=self.accelerator,
        )
        
        # RadGraph-XL metric:
        metric_name = inspect.signature(RadGraphXLMetric).parameters.get('metric_name').default
        metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
        metrics[f'{stage}_{section}_radgraph_xl'] = RadGraphXLMetric(
            exp_dir=self.exp_trial_dir,
            split=f'{stage}_{section}',
            metric_name=metric_name,
            accelerator=self.accelerator,
        )

        # RadEval metric:
        if hasattr(transformers, 'ModernBertModel'):
            metric_name = inspect.signature(RadEvalBERTScoreMetric).parameters.get('metric_name').default
            metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
            metrics[f'{stage}_{section}_radeval_bertscore'] = RadEvalBERTScoreMetric(
                exp_dir=self.exp_trial_dir,
                split=f'{stage}_{section}',
                metric_name=metric_name,
                accelerator=self.accelerator,
                num_workers=self.dataloader_num_workers,
            )

        # CXR-BERT metric:
        metric_name = inspect.signature(CXRBERTMetric).parameters.get('metric_name').default
        metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
        metrics[f'{stage}_{section}_cxr-bert'] = CXRBERTMetric(
            exp_dir=self.exp_trial_dir,
            split=f'{stage}_{section}',
            metric_name=metric_name,
            accelerator=self.accelerator,
        )

        # BERTScore metric:
        metric_name = inspect.signature(BERTScoreRoBERTaLargeMetric).parameters.get('metric_name').default
        metric_name = f'{metric_name}_{test_set}' if test_set else metric_name
        metrics[f'{stage}_{section}_bertscore'] = BERTScoreRoBERTaLargeMetric(
            exp_dir=self.exp_trial_dir,
            split=f'{stage}_{section}',
            num_workers=self.dataloader_num_workers,
            metric_name=metric_name,
            accelerator=self.accelerator,
        )
        return metrics 

    def init_optimisers(self):
        
        # Creates dummy optimizer if `optimizer` was specified in the ds_config:
        optimiser_cls = (
            torch.optim.AdamW if self.accelerator.state.deepspeed_plugin is None
            or 'optimizer' not in self.accelerator.state.deepspeed_plugin.deepspeed_config
            else accelerate.utils.DummyOptim
        )
          
        self.optimiser = optimiser_cls(self.model.parameters(), **self.optimiser_kwargs)
        
        # Don't use global mini-batch size due to: https://huggingface.co/docs/accelerate/concept_guides/performance#learning-rates.
        steps_per_epoch = math.ceil(self.train_dataloader.dataset.__len__() / self.train_mbatch_size)

        self.print(f'Total steps per epoch: {steps_per_epoch}.')
        self.print(f'Total warmup steps: {self.num_warmup_steps * self.accelerator.num_processes}.')
        
        global_batch_size = self.train_mbatch_size * self.accelerator.num_processes
        global_steps_per_epoch = math.ceil(self.train_dataloader.dataset.__len__() / global_batch_size)

        self.print(f'Global steps per epoch: {global_steps_per_epoch}.')
        self.print(f'Global warmup steps: {self.num_warmup_steps}.')
        
        # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
        if (
            self.accelerator.state.deepspeed_plugin is None
            or 'scheduler' not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            if self.num_cycles:
                self.scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self.optimiser, 
                    num_warmup_steps=self.num_warmup_steps * self.accelerator.num_processes,
                    num_training_steps=steps_per_epoch * self.num_epochs,
                    num_cycles=self.num_cycles,
                )
            else:
                self.scheduler = transformers.get_constant_schedule_with_warmup(
                    self.optimiser, num_warmup_steps=self.num_warmup_steps * self.accelerator.num_processes,
                )
        else:
            self.scheduler = accelerate.utils.DummyScheduler(
                self.optimiser, warmup_num_steps=self.num_warmup_steps * self.accelerator.num_processes,
            )

        self.print(f'No. of training examples: {self.train_dataloader.dataset.__len__()}.')

    def training_step(self, batch):

        batch = batch.to(self.accelerator.device)
        
        if self.debug:
            self.model.label_ids = batch['label_ids']

        logits = self.model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            position_ids=batch['position_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
            time_deltas=batch['time_deltas'],
            time_deltas_mask=batch['time_deltas_mask'],
            use_cache=False,
        ).logits
                
        loss = F.cross_entropy(logits.permute([0, 2, 1]), batch['label_ids'], ignore_index=self.processor.tokenizer.pad_token_id)
        
        self.accelerator.backward(loss)        
        self.optimiser.step()
        self.scheduler.step()
        self.optimiser.zero_grad()

        loss_detached = loss.item()
        
        accumulate_scores = {
            'train_loss': loss_detached, 
            'prompt_len': batch['time_deltas_mask'].sum(dim=1).mean().item(),
        }

        step_scores = {
            'scheduler_lr': self.scheduler.get_last_lr()[-1],
            'scheduler_step': self.scheduler.scheduler._step_count,
        }
                
        return loss_detached, step_scores, accumulate_scores

    def generate_sections(self, batch):
        with torch.no_grad():         
            gen_model = self.model.module if hasattr(self.model, 'module') else self.model
            generated_ids = gen_model.generate(**batch, generation_config=self.generation_config)
            
        # Evaluate:
        findings, impression = self.processor.split_and_decode_sections(generated_ids) 

        prompt_len = (batch['input_ids'] != self.processor.tokenizer.pad_token_id).sum(dim=1).tolist()

        return generated_ids, findings, impression, prompt_len

    def validation_epoch(self, epoch):

        if not self.accelerator.state.distributed_type.name == 'NO':
            assert isinstance(self.val_dataloader, accelerate.data_loader.DataLoaderShard), 'You must prepare the dataloader with accelerate.'
        
        pbar = tqdm(range(len(self.val_dataloader)))
        pbar.set_description('Validation')
        for step, batch in enumerate(self.val_dataloader):
            
            batch = batch.to(self.accelerator.device)
            
            findings_gt = batch.data.pop('findings', None)
            impression_gt = batch.data.pop('impression', None)
            study_id = batch.data.pop('study_id', None)

            generated_ids, findings, impression, _ = self.generate_sections(batch)

            # Evaluate:
            self.val_report_ids_logger.update(generated_ids, study_ids=study_id)
            self.val_report_logger.update(findings, impression, study_ids=study_id)            
            
            # Handle missing radiolgist sections:
            findings = [i for i, j in zip(findings, findings_gt, strict=True) if j is not None]
            findings_gt = [i for i in findings_gt if i is not None]
            
            impression = [i for i, j in zip(impression, impression_gt, strict=True) if j is not None]
            impression_gt = [i for i in impression_gt if i is not None]
            
            for metric_name, metric in self.val_metrics.items():
                if 'findings' in metric_name and findings_gt:
                    metric.update(findings, findings_gt, study_ids=study_id)
                elif 'impression' in metric_name and impression_gt:
                    metric.update(impression, impression_gt, study_ids=study_id)     

            if self.accelerator.is_main_process:
                pbar.update(1)
        
        pbar.close()
        
        self.val_report_logger.compute(epoch)
        self.val_report_logger.reset()
        self.val_report_ids_logger.compute(epoch)
        self.val_report_ids_logger.reset()
        
        scores = {}
        
        for i in self.val_metrics.keys():
            output = self.val_metrics[i].compute(epoch)
            if isinstance(output, dict):
                for k, v in output.items():
                    scores.update({k: v})
            else:
                scores.update({i: output})
            self.val_metrics[i].reset()
        
        return scores

    def test_epoch(self, epoch):
        
        self.model.eval()
        
        for test_dataloader, test_set in zip(self.test_dataloaders, self.test_datasets, strict=True):
            
            if not self.accelerator.state.distributed_type.name == 'NO':
                assert isinstance(test_dataloader, accelerate.data_loader.DataLoaderShard), 'You must prepare the dataloader with accelerate.'

            pbar = tqdm(range(len(test_dataloader)))
            pbar.set_description(f'Test ({test_set})')
            
            for step, batch in enumerate(test_dataloader):
                
                batch = batch.to(self.accelerator.device)
                
                findings_gt = batch.data.pop('findings', None)
                impression_gt = batch.data.pop('impression', None)
                study_id = batch.data.pop('study_id', None)

                generated_ids, findings, impression, prompt_len = self.generate_sections(batch)

                if test_set == 'rexgradient':
                    self.test_report_logger_rexgradient.update(findings, impression, study_ids=study_id)
                    self.test_report_ids_logger_rexgradient.update(generated_ids, study_ids=study_id)
                    self.test_prompt_len_logger_rexgradient.update(prompt_len, study_ids=study_id)
                elif test_set == 'chexpert_plus':
                    self.test_report_logger_chexpert_plus.update(findings, impression, study_ids=study_id)
                    self.test_report_ids_logger_chexpert_plus.update(generated_ids, study_ids=study_id)
                    self.test_prompt_len_logger_chexpert_plus.update(prompt_len, study_ids=study_id)
                elif test_set == 'mimic_cxr_dpo':
                    self.test_report_logger_mimic_cxr_dpo.update(findings, impression, study_ids=study_id)
                    self.test_report_ids_logger_mimic_cxr_dpo.update(generated_ids, study_ids=study_id)
                    self.test_prompt_len_logger_mimic_cxr_dpo.update(prompt_len, study_ids=study_id)
                else:
                    self.test_report_ids_logger.update(generated_ids, study_ids=study_id)
                    self.test_report_logger.update(findings, impression, study_ids=study_id)  
                    self.test_prompt_len_logger.update(prompt_len, study_ids=study_id)
                
                # Handle missing radiologist sections:
                findings = [i for i, j in zip(findings, findings_gt, strict=True) if j is not None]
                findings_gt = [i for i in findings_gt if i is not None]   
            
                impression = [i for i, j in zip(impression, impression_gt, strict=True) if j is not None]
                impression_gt = [i for i in impression_gt if i is not None]
                        
                for metric_name, metric in self.test_metrics.items():

                    if ('chexpert_plus' in metric_name) != (test_set == 'chexpert_plus'):
                        continue

                    if ('rexgradient' in metric_name) != (test_set == 'rexgradient'):
                        continue

                    if ('mimic_cxr_dpo' in metric_name) != (test_set == 'mimic_cxr_dpo'):
                        continue

                    if 'findings' in metric_name and findings_gt:
                        metric.update(findings, findings_gt, study_ids=study_id)
                    elif 'impression' in metric_name and impression_gt:
                        metric.update(impression, impression_gt, study_ids=study_id)
                        
                if self.accelerator.is_main_process:
                    pbar.update(1)
                        
        pbar.close()

        scores = {}

        if 'mimic_cxr' in self.test_datasets:
            self.test_report_logger.compute(epoch)
            self.test_report_logger.reset()
            self.test_report_ids_logger.compute(epoch)
            self.test_report_ids_logger.reset()
            output = self.test_prompt_len_logger.compute(epoch)
            self.test_prompt_len_logger.reset()

            scores = {**scores, **output}
        
        if 'mimic_cxr_dpo' in self.test_datasets:
            self.test_report_logger_mimic_cxr_dpo.compute(epoch)
            self.test_report_logger_mimic_cxr_dpo.reset()
            self.test_report_ids_logger_mimic_cxr_dpo.compute(epoch)
            self.test_report_ids_logger_mimic_cxr_dpo.reset()
            output = self.test_prompt_len_logger_mimic_cxr_dpo.compute(epoch)
            self.test_prompt_len_logger_mimic_cxr_dpo.reset()
                        
            scores = {**scores, **output}


        if 'chexpert_plus' in self.test_datasets:
            self.test_report_logger_chexpert_plus.compute(epoch)
            self.test_report_logger_chexpert_plus.reset()
            self.test_report_ids_logger_chexpert_plus.compute(epoch)
            self.test_report_ids_logger_chexpert_plus.reset()
            output = self.test_prompt_len_logger_chexpert_plus.compute(epoch)
            self.test_prompt_len_logger_chexpert_plus.reset()
                        
            scores = {**scores, **output}

        if 'rexgradient' in self.test_datasets:
            self.test_report_logger_rexgradient.compute(epoch)
            self.test_report_logger_rexgradient.reset()
            self.test_report_ids_logger_rexgradient.compute(epoch)
            self.test_report_ids_logger_rexgradient.reset()
            output = self.test_prompt_len_logger_rexgradient.compute(epoch)
            self.test_prompt_len_logger_rexgradient.reset()
                        
            scores = {**scores, **output}
            
        for i in self.test_metrics.keys():
            output = self.test_metrics[i].compute(epoch)
            if isinstance(output, dict):
                for k, v in output.items():
                    scores.update({k: v})
            else:
                scores.update({i: output})    
            self.test_metrics[i].reset()
            
        scores = {**scores, **output}
        
        return scores

    def upload_to_hf_hub(self):
        transformers.AutoConfig.register('cxrmate-2', CXRMate2Config)
        transformers.AutoModelForCausalLM.register(CXRMate2Config, CXRMate2ForConditionalGeneration)

        self.processor.mimic_cxr_normalisation = True   # Set for inference.

        token_map = {
            '<|reserved_special_token_0|>': '<|image|>',
            '<|reserved_special_token_1|>': '<|generate_findings|>',
            '<|reserved_special_token_2|>': '<|generate_impression|>',

            '<|reserved_special_token_4|>': '<|time_delta_token_type|>',

            '<|reserved_special_token_16|>': '<|image_token_type|>',
            '<|reserved_special_token_17|>': '<|indication_token_type|>',
            '<|reserved_special_token_18|>': '<|history_token_type|>',
            '<|reserved_special_token_19|>': '<|comparison_token_type|>',
            '<|reserved_special_token_20|>': '<|technique_token_type|>',
            '<|reserved_special_token_21|>': '<|findings_token_type|>',
            '<|reserved_special_token_22|>': '<|impression_token_type|>',
            '<|reserved_special_token_23|>': '<|prior_image_token_type|>',
            '<|reserved_special_token_24|>': '<|prior_findings_token_type|>',
            '<|reserved_special_token_25|>': '<|prior_impression_token_type|>',

            '<|reserved_special_token_64|>': '<|sep|>',
            '<|reserved_special_token_65|>': '<|pad|>',
        }

        self.processor.image_token = '<|image|>'
        self.processor.generate_findings_token = '<|generate_findings|>'
        self.processor.generate_impression_token = '<|generate_impression|>'
        self.processor.token_type_to_token = {k:token_map[v] for k, v in self.processor.token_type_to_token.items()}

        used_ids = set()
        for k in token_map.keys():
            idx = int(k.split('_')[-1].rstrip('|>'))
            used_ids.add(idx)

        def next_free_id(start=0):
            for i in range(start, 248):
                if i not in used_ids:
                    return i
            raise ValueError("No free reserved_special_token slots left.")

        # Add X tokens:
        for i in range(100):
            free_id = next_free_id()
            used_ids.add(free_id)
            token_map[f'<|reserved_special_token_{free_id}|>'] = f'<|x{i}|>'

        # Add Y tokens:
        for i in range(100):
            free_id = next_free_id()
            used_ids.add(free_id)
            token_map[f'<|reserved_special_token_{free_id}|>'] = f'<|y{i}|>'

        CXRMate2Processor.register_for_auto_class()
        self.processor.push_to_hub(self.hf_hub_alias)

        rename_added_tokens(self.hf_hub_alias, token_map=token_map)

        CXRMate2Config.register_for_auto_class()
        CXRMate2ForConditionalGeneration.register_for_auto_class('AutoModelForCausalLM')

        self.load_safetensors_state_dict(self.model, self.upload_ckpt_dir)
        self.model.push_to_hub(self.hf_hub_alias)

        self.generation_config.push_to_hub(self.hf_hub_alias)
