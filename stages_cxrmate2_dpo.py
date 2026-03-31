import copy
import glob
import os
import random
import re

import accelerate
import datasets
import pandas as pd
import torch
import torch.nn.functional as F
from dataset import CXRMate2Dataset
from stages_cxrmate2 import Stages as BaseStages
from torch.utils.data import ConcatDataset, DataLoader, Subset
from trl.trainer.utils import selective_log_softmax


class Stages(BaseStages):

    def __init__(
        self, 
        generated_reports,
        beta, 
        max_train_prompt_len,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generated_reports = generated_reports
        self.beta = beta
        self.max_train_prompt_len = max_train_prompt_len

        assert self.train_mbatch_size == 1

    def dataloader_collate_functions(self):
        def train_collate_fn(batch):
            if isinstance(batch, list):
                keys = set().union(*(d.keys() for d in batch))
                batch = {j: [i.setdefault(j, None) for i in batch] for j in keys}
                batch = {k: torch.stack(v) if isinstance(v[0], torch.Tensor) else v for k, v in batch.items()}
                    
            processed = self.processor(train=False, **batch)

            processed.data['findings'] = batch['findings']
            processed.data['impression'] = batch['impression']  
            processed.data['study_id'] = batch['study_id']  

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

        # Load radiologist preferences:
        self.preferences = pd.read_json(self.preferences)

        # Load generated reports:
        self.generated = pd.read_csv(self.generated_reports)
        
        # Dataset:
        dataset = datasets.load_from_disk(os.path.join(self.database_dir, 'mimic_cxr_jpg_dataset'))
        if (self.train and 'chexpert_plus' in self.train_datasets) or (self.test and 'chexpert_plus' in self.test_datasets):
            dataset_chexpert_plus = datasets.load_from_disk(os.path.join(self.database_dir, 'chexpert_plus_dataset'))
        if (self.train and 'rexgradient' in self.train_datasets) or (self.test and 'rexgradient' in self.test_datasets):
            dataset_rexgradient = datasets.load_from_disk(os.path.join(self.database_dir, 'rexgradient_160k_dataset'))

        train_collate_fn, test_collate_fn = self.dataloader_collate_functions()
            
        train_datasets = []

        # MIMIC-CXR:
        if 'mimic_cxr' in self.train_datasets:
            train_set = dataset['test']

            df = pd.DataFrame({'study_id': train_set['study_id']})

            train_indices = df[~df['study_id'].isin(self.preferences['study_id'])].index.tolist()

            train_indices = list(set(range(len(train_set))) - set(train_indices))

            train_set = CXRMate2Dataset(train_set, self.history)
            train_set = Subset(train_set, train_indices)
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

            # MIMIC-CXR DPO:
            if 'mimic_cxr_dpo' in self.test_datasets:
                test_set = dataset['test']
                df = pd.DataFrame({'study_id': test_set['study_id'], 'findings': test_set['findings'], 'impression': test_set['impression']})
                if self.findings_and_impression_strategy == 'or':
                    indices = df[df[['findings', 'impression']].isnull().all(axis=1)].index.tolist()  # Consider studies with findings OR impression section.
                elif self.findings_and_impression_strategy == 'and':
                    indices = df[df[['findings', 'impression']].isnull().any(axis=1)].index.tolist()  # Consider studies with findings AND impression section.

                test_set = CXRMate2Dataset(test_set, self.history)  
                indices = list(set(range(len(test_set))) - set(indices))  
                indices = [x for x in indices if x not in train_indices]

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
                self.print(f'No. of MIMIC-CXR DPO test examples: {test_set.__len__()}.')
                self.accelerator.log(
                    {
                        'stage': 'MIMIC-CXR DPO test', 
                        'epoch': 0, 
                        'step': 0,
                        'len': test_set.__len__(),
                    },
                    step=0,
                )
                self.test_dataloaders.append(test_dataloader)

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

    def init_model(self):
        super().init_model()

        # Reference model:
        if self.beta > 0:
            base = self.model.module if hasattr(self.model, "module") else self.model
            self.ref_model = copy.deepcopy(base)
            self.ref_model.eval()

    def accelerate_prepare(self):

        super().accelerate_prepare()

        if self.accelerator.state.distributed_type.name == 'FSDP' and self.train:

            assert self.accelerator.state.fsdp_plugin.fsdp_version == 2

            self.ref_model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.ref_model)

    def post_prepare(self):
        
        if self.accelerator.state.distributed_type.name != 'FSDP' and self.train:

            # Reference model:
            if self.beta > 0:
                self.ref_model.to(device=self.accelerator.device)

    def training_epoch(self, epoch):
        if self.beta > 0:

            ckpt_dirs = glob.glob(os.path.join(self.exp_trial_dir, 'completed_epoch_*_step_*/evaluate'))
            ckpt_dir = None
            if ckpt_dirs:
                def extract_epoch_step(path):
                    match = re.search(r'completed_epoch_(\d+)_step_(\d+)\.ckpt', os.path.basename(path))
                    if match:
                        return int(match.group(1)), int(match.group(2))
                    return -1, -1
                ckpt_dirs.sort(key=lambda x: extract_epoch_step(x), reverse=True)
                ckpt_dir = ckpt_dirs[0]
            if ckpt_dir is None:
                model = self.model.module if hasattr(self.model, "module") else self.model
                self.ref_model.load_state_dict(model.state_dict(), strict=True)
                self.ref_model.to(device=self.accelerator.device)
            else:
                self.load_safetensors_state_dict(self.ref_model, ckpt_dir)

            self.ref_model.eval()

        self.accelerator.wait_for_everyone()
        
        return super().training_epoch(epoch)

    def training_step(self, batch):
        """
        DPO with radiologist preferences.

        For each study, we have a radiologist preference ('Generated', 'Radiologist', or 'No preference').
        - 'Generated': chosen = generated report, rejected = radiologist report.
        - 'Radiologist': chosen = radiologist report, rejected = generated report.
        - 'No preference': skip (label_weight = 0).

        The generated and radiologist reports form the preference pair.

        DPO loss: -log sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))
        """

        batch = batch.to(self.accelerator.device)

        findings_gt = batch.data.pop('findings', None)
        impression_gt = batch.data.pop('impression', None)
        study_ids = batch.data.pop('study_id', None)

        # Look up radiologist preferences for this study:
        study_id = study_ids[0]  # batch_size == 1
        pref_row = self.preferences[self.preferences['study_id'] == study_id]
        preferences = pref_row['preference'].to_list()

        # Convert votes to soft preference probability p = P(gen > rad):
        # Each 'Generated' vote contributes 1.0, each 'No preference' contributes 0.5, each 'Radiologist' contributes 0.0
        # Works with any number of raters (1, 3, 5, etc.)
        n_gen = preferences.count('Generated')
        n_rad = preferences.count('Radiologist')
        n_tie = preferences.count('No preference')
        n_total = len(preferences)

        # Calculate probability that generated is preferred over radiologist:
        if n_total > 0:
            p = (n_gen + 0.5 * n_tie) / n_total
        else:
            p = 0.5  # Default to neutral if no preferences

        # Look up pre-generated report for this study:
        generated = self.generated[self.generated['study_id'] == study_id]
        findings_gen = generated['findings'].values[0] 
        impression_gen = generated['impression'].values[0]

        # Prompt size:
        prompt_len = batch['input_ids'].shape[1]
        prompt_ids = batch['input_ids'][0, :prompt_len]

        # Construct generated report token ids:
        generated_report = f'{self.processor.tokenizer.bos_token}{findings_gen}{self.processor.tokenizer.sep_token}{impression_gen}{self.processor.tokenizer.eos_token}'
        generated_report_ids = self.processor.tokenizer.encode(
            generated_report,
            add_special_tokens=False,
            return_tensors='pt',
        )[0].to(self.accelerator.device)
        generated_ids = torch.cat([prompt_ids, generated_report_ids], dim=0).unsqueeze(0)

        # Construct radiologist report token ids:
        findings_gt = findings_gt[0] if findings_gt[0] is not None else ''
        impression_gt = impression_gt[0] if impression_gt[0] is not None else ''
        radiologist_report = f'{self.processor.tokenizer.bos_token}{findings_gt}{self.processor.tokenizer.sep_token}{impression_gt}{self.processor.tokenizer.eos_token}'
        radiologist_report_ids = self.processor.tokenizer.encode(
            radiologist_report,
            add_special_tokens=False,
            return_tensors='pt',
        )[0].to(self.accelerator.device)
        radiologist_ids = torch.cat([prompt_ids, radiologist_report_ids], dim=0).unsqueeze(0)

        # Determine chosen and rejected based on soft preference probability:
        if p > 0.5: # Generated is preferred
            chosen_ids = generated_ids
            rejected_ids = radiologist_ids
            label_weight = 2 * p - 1  # Maps [0.5, 1.0] to [0.0, 1.0]
        elif p < 0.5: # Radiologist is preferred
            chosen_ids = radiologist_ids
            rejected_ids = generated_ids
            label_weight = 1 - 2 * p  # Maps [0.0, 0.5] to [1.0, 0.0]
        else:  # p == 0.5, no preference (neutral):
            chosen_ids = generated_ids
            rejected_ids = radiologist_ids
            label_weight = 0.0

        # Prepare chosen batch (deep copy batch data so update_batch_for_rl can mutate it):
        batch_chosen = copy.deepcopy(batch)
        batch_chosen = self.processor.update_batch_for_rl(batch_chosen, chosen_ids)
        num_chosen_completion_tokens = chosen_ids.shape[1] - prompt_len
        chosen_label_ids = batch_chosen.data.pop('label_ids', None)
        chosen_completion_mask = batch_chosen.data.pop('completion_mask', None)
        chosen_label_ids = chosen_label_ids[:, -num_chosen_completion_tokens:]

        # Prepare rejected batch (deep copy batch data so update_batch_for_rl can mutate it):
        batch_rejected = copy.deepcopy(batch)
        batch_rejected = self.processor.update_batch_for_rl(batch_rejected, rejected_ids)
        num_rejected_completion_tokens = rejected_ids.shape[1] - prompt_len
        rejected_label_ids = batch_rejected.data.pop('label_ids', None)
        rejected_completion_mask = batch_rejected.data.pop('completion_mask', None)
        rejected_label_ids = rejected_label_ids[:, -num_rejected_completion_tokens:]

        # Reference policy log probabilities:
        with torch.no_grad():
            ref_chosen_logits = self.ref_model(**batch_chosen).logits
            ref_chosen_logits = ref_chosen_logits[:, -num_chosen_completion_tokens:]
            ref_chosen_log_p = selective_log_softmax(ref_chosen_logits, chosen_label_ids)
            ref_chosen_log_p = (ref_chosen_log_p * chosen_completion_mask).sum(dim=-1)

            ref_rejected_logits = self.ref_model(**batch_rejected).logits
            ref_rejected_logits = ref_rejected_logits[:, -num_rejected_completion_tokens:]
            ref_rejected_log_p = selective_log_softmax(ref_rejected_logits, rejected_label_ids)
            ref_rejected_log_p = (ref_rejected_log_p * rejected_completion_mask).sum(dim=-1)

        # Current policy log probabilities:
        chosen_logits = self.model(use_cache=False, **batch_chosen).logits
        chosen_logits = chosen_logits[:, -num_chosen_completion_tokens:]
        policy_chosen_log_p = selective_log_softmax(chosen_logits, chosen_label_ids)
        policy_chosen_log_p = (policy_chosen_log_p * chosen_completion_mask).sum(dim=-1)

        rejected_logits = self.model(use_cache=False, **batch_rejected).logits
        rejected_logits = rejected_logits[:, -num_rejected_completion_tokens:]
        policy_rejected_log_p = selective_log_softmax(rejected_logits, rejected_label_ids)
        policy_rejected_log_p = (policy_rejected_log_p * rejected_completion_mask).sum(dim=-1)

        # DPO loss: -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
        log_ratio_chosen = policy_chosen_log_p - ref_chosen_log_p
        log_ratio_rejected = policy_rejected_log_p - ref_rejected_log_p
        logits_diff = self.beta * (log_ratio_chosen - log_ratio_rejected)
        loss = -F.logsigmoid(logits_diff).mean() * label_weight

        self.accelerator.backward(loss)
        self.optimiser.step()
        self.scheduler.step()
        self.optimiser.zero_grad()

        loss_detached = loss.item()

        # Reward accuracy: fraction where chosen is preferred by the model:
        with torch.no_grad():
            reward_accuracy = (logits_diff > 0).float().mean().item()

        # Sequence lengths:
        chosen_seq_len = chosen_completion_mask.sum(dim=-1).float().mean().item()
        rejected_seq_len = rejected_completion_mask.sum(dim=-1).float().mean().item()

        scheduler_step = self.scheduler.scheduler.last_batch_iteration if hasattr(self.scheduler.scheduler, 'last_batch_iteration') else self.scheduler.scheduler._step_count

        accumulate_scores = {
            'train_loss': loss_detached,
            'gpu_allocated_memory_gb': torch.cuda.memory_allocated(self.accelerator.device) / 1e9,
            'gpu_reserved_memory_gb': torch.cuda.memory_reserved(self.accelerator.device) / 1e9,
            'chosen_seq_len': chosen_seq_len,
            'rejected_seq_len': rejected_seq_len,
            'prompt_len': prompt_len,
            'reward_accuracy': reward_accuracy,
            'label_weight': label_weight,
            'preference_prob': p,
            'log_ratio_chosen': log_ratio_chosen.mean().item(),
            'log_ratio_rejected': log_ratio_rejected.mean().item(),
        }

        step_scores = {
            'scheduler_lr': self.scheduler.get_last_lr()[-1],
            'scheduler_step': scheduler_step,
        }

        return loss_detached, step_scores, accumulate_scores

