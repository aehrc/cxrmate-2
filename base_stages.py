import csv
import datetime
import glob
import json
import os
import pathlib
import pprint
import random
import re
import shutil
import signal
import sys
import time
import warnings
from pathlib import Path
from subprocess import call
from typing import Optional

import accelerate
import pandas as pd
import pkg_resources
import psutil
import torch
import transformers
import yaml
from command_line_arguments import read_command_line_arguments
from munch import DefaultMunch
from peft import PeftModel
from safetensors.torch import load_file
from slurm import SlurmSubmit
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from utils import (
    CSVTracker,
    TrainStepScoresMetric,
    copy_dir_ignoring_extensions,
    get_best_ckpt,
    load_config_and_update_args,
    submit_stages,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class BaseStages:
    
    def __init__(
        self,
        trial: int,
        exp_dir: str,
        exp_trial_dir: str,
        num_epochs: int,
        submit: bool,
        one_epoch_only: bool,
        monitor: str,
        monitor_mode: str,
        num_workers: int,
        train: bool = False,
        validate: Optional[bool] = None,
        test: bool = False,
        test_pretrained: bool = False,
        debug: bool = False,
        limit_train_samples: Optional[int] = None,
        limit_val_samples: Optional[int] = None,
        limit_test_samples: Optional[int] = None,
        log_loss_every_n_steps: int = 50,
        prefetch_factor: Optional[int] = 2,
        submit_evaluation: bool = False,
        submit_evaluation_kwargs: Optional[dict] = None,
        validate_ckpt_dir: Optional[str] = None,
        validate_check_interval: float = 1.0,
        test_job: bool = False,
        long_term_exp_dir: Optional[str] = None,
        ignore_extensions: list = [],
        time_limit: Optional[int] = None,
        dataloader_num_workers: Optional[int] = None,
        fake_submit: bool = False,
        partial_checkpointing: bool = True,
        debug_partial_checkpointing: bool = False,
        example_id_key: Optional[str] = None,       
        validate_remaining_ckpts: bool = False, 
        transformers_model: bool = True,
        upload: bool = False,
        upload_ckpt_dir: Optional[str] = None,
        hf_hub_alias: Optional[str] = None,
        **kwargs,
    ):
        self.trial = trial
        self.exp_dir = exp_dir
        self.exp_trial_dir = exp_trial_dir
        self.num_epochs = num_epochs
        self.submit = submit
        self.one_epoch_only = one_epoch_only
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.num_workers = num_workers
        self.train = train
        self.validate = validate
        self.test = test
        self.test_pretrained = test_pretrained
        self.debug = debug
        self.limit_train_samples = limit_train_samples
        self.limit_val_samples = limit_val_samples
        self.limit_test_samples = limit_test_samples
        self.log_loss_every_n_steps = log_loss_every_n_steps
        self.prefetch_factor = prefetch_factor
        self.submit_evaluation = submit_evaluation
        self.submit_evaluation_kwargs = submit_evaluation_kwargs
        self.validate_ckpt_dir = validate_ckpt_dir
        self.validate_check_interval = validate_check_interval
        self.test_job = test_job
        self.long_term_exp_dir = long_term_exp_dir
        self.ignore_extensions = ignore_extensions
        self.dataloader_num_workers = dataloader_num_workers
        self.fake_submit = fake_submit
        self.partial_checkpointing = partial_checkpointing
        self.debug_partial_checkpointing = debug_partial_checkpointing
        self.example_id_key = example_id_key
        self.validate_remaining_ckpts = validate_remaining_ckpts
        self.transformers_model = transformers_model
        self.upload = upload
        self.upload_ckpt_dir = upload_ckpt_dir
        self.hf_hub_alias = hf_hub_alias

        if self.validate is None:
            self.validate = self.train or self.validate_remaining_ckpts or self.validate_ckpt_dir
        
        assert self.validate_check_interval <= 1.0 and self.validate_check_interval > 0.0, '[__init__]: "validate_check_interval" must be in the range (0, 1].'
        
        self.dataloader_num_workers = self.num_workers if self.dataloader_num_workers is None else self.dataloader_num_workers
        if self.dataloader_num_workers == 0:
            self.prefetch_factor = None

        self.modules_initialised = False

        self.save_partial_ckpt = False
        self.starting_step = None
        
        self.train_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None
        
        self.model = None
        
        self.peft_config = None
                    
        tracker = CSVTracker(exp_trial_dir)
        self.accelerator = accelerate.Accelerator(log_with=tracker, project_dir=exp_trial_dir)
        self.accelerator.init_trackers('csv')
        # else:
        #     self.accelerator = accelerate.Accelerator(project_dir=exp_trial_dir)
        #     self.logger = CSVLogger(exp_trial_dir)

        if self.accelerator.state.distributed_type.name == 'FSDP':
            assert self.accelerator.state.fsdp_plugin.fsdp_version == 2

        self.rank = os.environ.get('RANK', '0')
        self.node = os.environ.get('SLURM_NODEID')
        self.is_node_0_rank_0 = self.rank == '0' and (self.node == '0' or self.node == None)

        self.print('[__init__]: accelerator state:')
        self.print(self.accelerator.state)
        
        if (self.submit or self.fake_submit) and self.train and self.partial_checkpointing:
                       
            if time_limit.count(':') == 1:
                format = '%M:%S'
            elif time_limit.count(':') == 2:
                format = '%H:%M:%S'               
            if '-' in time_limit:
                format = '%d-' + format
            t = datetime.datetime.strptime(time_limit, format)
            
            if '-' in time_limit:
                delta = datetime.timedelta(days=t.day, hours=t.hour, minutes=t.minute, seconds=t.second)
            else:
                delta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
            
            timer_duration_seconds = int(delta.total_seconds() - 15 * 60)  # Subtract 10 minutes to allow for saving the partial checkpoint and startup.

            self.print(f'[__init__]: setting signal. SIGALRM will be triggered in {timer_duration_seconds} seconds.')
            self.accelerator.wait_for_everyone()
            signal.alarm(timer_duration_seconds)

            self.print('[__init__]: setting signal handler that flags: 1) saving an partial checkpoint during an epoch, and 2) requeuing the job.')
            signal.signal(signal.SIGALRM, self.sig_handler)
            
        transformers.set_seed(self.trial)
        random.seed(self.trial)
        
        self.limit_train_samples = 1 if self.debug and self.limit_train_samples is None else self.limit_train_samples
        self.limit_val_samples = 1 if self.debug and self.limit_val_samples is None else self.limit_val_samples
        self.limit_test_samples = 1 if self.debug and self.limit_test_samples is None else self.limit_test_samples
            
        Path(f'{self.exp_trial_dir}/metric_outputs').mkdir(parents=True, exist_ok=True)
        
        # Check if training can be resumed:
        dirs = glob.glob(f"{self.exp_trial_dir}/partial_epoch_*_step_*") + glob.glob(f"{self.exp_trial_dir}/completed_epoch_*_step_*")
        self.print(f'[__init__]: available checkpoints to resume from: {dirs}.')
        self.latest_ckpt_dir = max(dirs, key=lambda d: int(re.search(r'step_(\d+)', d).group(1)), default=None)
        self.print(f'[__init__]: selected checkpoint to resume from: {self.latest_ckpt_dir}. (This is just the selected checkpoint, it has not been used to resume the state yet.)')

        # Extract the epoch and step count from the selected directory:
        if self.latest_ckpt_dir:
            epoch_match = re.search(r'(?:partial|completed)_epoch_(\d+)', self.latest_ckpt_dir)
            step_match = re.search(r'step_(\d+)', self.latest_ckpt_dir)
            self.last_epoch = int(epoch_match.group(1)) if epoch_match else None
            self.last_step = int(step_match.group(1)) if step_match else None
            self.print(f'[__init__]: last epoch: {self.last_epoch}, last step: {self.last_step}.')

        else:
            self.last_epoch = None
            self.last_step = None
 
        self.train_step_scores = TrainStepScoresMetric()
        
        self.step_log_path = os.path.join(self.exp_trial_dir, 'step_log.csv')

    def init_modules(self):
        self.init_dataloaders()
        self.init_processor()
        self.init_model()
        if isinstance(self.model, PeftModel) or hasattr(self.model, 'peft_config'):
            assert hasattr(self, 'peft_config'), '[__init_modules__]: a "peft_config" attribute is needed to load a PeftModel model checkpoint.'
        if (self.last_epoch is None and self.train) or self.test_pretrained:
            self.warm_start()
        if self.train:
            self.init_optimisers()
        if self.train or self.validate or self.test:
            self.accelerate_prepare()
        self.post_prepare()
        self.init_metrics()  

        if self.debug:
            try:
                self.model.processor = self.processor
            except Exception:
                pass

        self.log_details()

        self.modules_initialised = True
        
    def init_dataloaders(self):
        warnings.warn('init_dataloaders has not been implemented in the subclass.')
         
    def init_processor(self):
        warnings.warn('init_processor has not been implemented in the subclass.')

    def init_model(self):
        warnings.warn('init_model has not been implemented in the subclass.')

    def warm_start(self):
        warnings.warn('warm_start has not been implemented in the subclass.')

    def init_optimisers(self):
        warnings.warn('init_optimisers has not been implemented in the subclass.')

    def upload_to_hf_hub(self):
        warnings.warn('upload_to_hf_hub has not been implemented in the subclass.')

    def accelerate_prepare(self):

        assert self.model is not None, "[accelerate_prepare]: the model attribute is not defined. Please ensure `self.model` is set correctly."

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
            to_prepare.append(self.test_dataloader)

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
            self.test_dataloader = prepared[index]

    def post_prepare(self):
        warnings.warn('post_prepare has not been implemented in the subclass.')

    def init_metrics(self):
        warnings.warn('init_metrics has not been implemented in the subclass.')

    def log_details(self):

        # Save environment packages:
        packages = sorted([f"{d.project_name}=={d.version}" for d in pkg_resources.working_set])
        packages_path = os.path.join(self.exp_trial_dir, 'packages.txt')
        try:
            with open(packages_path, 'w') as f:
                for pkg in packages:
                    f.write(pkg + '\n')
            self.print(f"[log_details]: Saved environment packages to {packages_path}.")
        except Exception as e:
            self.print(f"[log_details]: Failed to save packages list: {e}")

        # Save model layer summary:
        model = getattr(self, 'model', None)
        if model is not None:
            model_summary_path = os.path.join(self.exp_trial_dir, 'model_layers.txt')
            try:
                with open(model_summary_path, 'w') as f:
                    f.write(f"{'Layer':100} {'#Params':>14} {'Dtype':>16} {'Size_MB':>14}\n")
                    f.write('\n')
                    total_params = 0
                    total_size = 0.0
                    for name, param in model.named_parameters():
                        num_params = param.numel()
                        dtype = str(param.dtype)
                        size_mb = param.element_size() * num_params / (1024 ** 2)
                        total_params += num_params
                        total_size += size_mb
                        f.write(f"{name:100} {num_params:14} {dtype:>16} {size_mb:14.2f}\n")
                    f.write('\n')
                    f.write(f"{'Total':100} {total_params:14} {'':>16} {total_size:14.2f}\n")
                self.print(f'[log_details]: Saved model layer summary to {model_summary_path}.')
            except Exception as e:
                self.print(f'[log_details]: Failed to save model layer summary: {e}.')

    def stages(self):
        self()

    def __call__(self):

        if not self.modules_initialised:
            self.init_modules()

        # Resume training from last checkpoint:
        if self.last_epoch is not None and self.train and self.last_epoch < self.num_epochs:
            self.load_state(self.latest_ckpt_dir)
            self.starting_step = self.last_step + 1
            if 'partial_epoch_' in self.latest_ckpt_dir:
                starting_epoch = self.last_epoch 
                self.print(f'[__call__]: loaded checkpoint from epoch {self.last_epoch}, step {self.starting_step}. resuming training from epoch {self.last_epoch}, step {self.starting_step + 1}.')
            else:
                self.print(f'[__call__]: loaded checkpoint from the end of epoch {self.last_epoch}, resuming training from the start of epoch {self.last_epoch + 1}.')
                starting_epoch = self.last_epoch + 1
        elif self.train:
            self.print('[__call__]: no last checkpoint found to resume training from.')
            starting_epoch, self.starting_step = 0, 0
                        
        starting_epoch = self.num_epochs if not self.train else starting_epoch
        epoch = starting_epoch
        for epoch in range(starting_epoch, self.num_epochs):
                        
            if self.train:

                assert self.train_dataloader is not None, '[__call__]: train_dataloader is None and has not been set.'

                if self.last_epoch is not None and epoch == self.last_epoch and 'partial_epoch_' in self.latest_ckpt_dir:
                    resume_step = self.starting_step - (len(self.train_dataloader) * starting_epoch)
                    assert resume_step < len(self.train_dataloader)
                    self.active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step)
                else:
                    self.active_dataloader = self.train_dataloader
                    
                self.training_epoch(epoch)
                
            if self.validate and not self.submit_evaluation:
                self.validation_epoch_wrapper(epoch, self.last_step)
                
            elif self.validate and self.submit_evaluation:
                self.submit_validate_job(ckpt_dir=os.path.join(self.exp_trial_dir, f'completed_epoch_{epoch}_step_{self.last_step}'))
                
            if self.one_epoch_only and self.submit and epoch != self.num_epochs - 1:
                rank_0 = True
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() != 0:
                        rank_0 = False
                if rank_0:
                    SlurmSubmit.sig_handler('one_epoch_only', None)
            elif self.one_epoch_only and self.submit and epoch == self.num_epochs - 1:
                self.test = False if self.submit_evaluation else self.test

            self.last_epoch = epoch

        if self.validate:
            if self.validate_remaining_ckpts:

                assert not self.train
                assert not self.test
                assert self.validate_ckpt_dir is None
                ckpt_dirs = [d for d in os.listdir(self.exp_trial_dir) if d.startswith('completed_epoch_')]

                # Remove epochs that have already been validated based on metrics.csv:
                ckpt_dirs = [
                    d for d in ckpt_dirs
                    if int(re.search(r'completed_epoch_(\d+)', d).group(1)) not in self.get_validated_epochs()
                ]

                for ckpt in ckpt_dirs:
                    if self.submit_evaluation:
                        self.submit_validate_job(ckpt_dir=os.path.join(self.exp_trial_dir, ckpt))
                    else:
                        self.validation_epoch_wrapper(None, None, os.path.join(self.exp_trial_dir, ckpt))

            if self.validate_ckpt_dir:
                epoch = self.validation_epoch_wrapper(None, None, self.validate_ckpt_dir)
                self.last_epoch = epoch
            
        if self.test and (self.validate_ckpt_dir is None or 'partial_validate_' not in self.validate_ckpt_dir):

            if not self.debug:
                self.print('[__call__]: preliminary check for best checkpoint directory...')
                best_ckpt_dir = self.get_best_ckpt(self.exp_trial_dir)
                self.print(f'[__call__]: best checkpoint directory: {best_ckpt_dir}.')

            if self.submit_evaluation and self.last_epoch == self.num_epochs - 1 and self.last_epoch in self.get_validated_epochs() and self.validate:
                self.submit_test_job()
            
            if not self.submit_evaluation or (self.submit_evaluation and self.test_job):
                self.test_epoch_wrapper()
                if self.long_term_exp_dir is not None and not self.debug:
                    self.print('[__call__]: copying experiment directory to long-term storage...')
                    if self.is_node_0_rank_0:
                        copy_dir_ignoring_extensions(
                            self.exp_trial_dir, 
                            os.path.join(self.long_term_exp_dir, self.exp_trial_dir.replace(self.exp_dir, '').lstrip('/')), 
                            self.ignore_extensions,
                        )
                    self.print('[__call__]: copy completed.')

        self.accelerator.end_training()
        self.print('[__call__]: stages completed.')

        if self.upload:
            self.upload_to_hf_hub()

    def training_epoch(self, epoch):

        total_loss = 0
        start_time = time.time()
        if self.is_node_0_rank_0:         
            pbar = tqdm(
                range(len(self.train_dataloader) * (epoch + 1)),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]',
            )
            pbar.set_description(f'Training epoch {epoch}')
            if self.starting_step > 0:
                pbar.update(self.starting_step)
        
        validation_check_frequency = int(self.validate_check_interval * len(self.train_dataloader))
        
        if not self.accelerator.state.distributed_type.name == 'NO':
            assert isinstance(self.active_dataloader, accelerate.data_loader.DataLoaderShard), '[training_epoch]: you must prepare the training dataloader with accelerate.'

        self.set_train()
        for step, batch in enumerate(self.active_dataloader):
            
            if self.debug_partial_checkpointing:
                write_header = not os.path.exists(self.step_log_path) or os.stat(self.step_log_path).st_size == 0
                with open(self.step_log_path, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['epoch', 'step', 'step_plus_starting_step', 'rank', 'node'])
                    if write_header:
                        writer.writeheader()
                    writer.writerow(
                        {
                            'epoch': epoch, 
                            'step': step, 
                            'step_plus_starting_step': step + self.starting_step, 
                            'rank': self.rank,
                            'node': self.node,
                            'example_id': batch[self.example_id_key] if self.example_id_key is not None else None,
                        }
                    )
            
            loss, step_scores, accumulate_scores = self.training_step(batch)     
            
            self.train_step_scores.update(accumulate_scores)
            
            if (step + self.starting_step) % self.log_loss_every_n_steps == 0:
                self.accelerator.log(
                    {
                        'stage': 'train_step', 
                        'epoch': epoch, 
                        'step': step + self.starting_step,
                        'step_active_dataloader': step,
                        **step_scores,
                        **self.train_step_scores.compute(),
                    },
                    step=step + self.starting_step,
                )
                self.train_step_scores.reset()      
                
                if self.is_node_0_rank_0:         
                    pbar.set_description(f'Training epoch {epoch} loss: {loss:.3f}')
                    pbar.update(self.log_loss_every_n_steps)
            
            total_loss += loss  # Not persistent for partial checkpointing. 
                                  
            if self.save_partial_ckpt:
                self.save_partial_ckpt_and_requeue(epoch, step + self.starting_step)
                
            if self.validate_check_interval != 1.0 and (step + self.starting_step + 1) % validation_check_frequency == 0:

                save_dir = os.path.join(self.exp_trial_dir, f'partial_epoch_{epoch}_step_{step + self.starting_step}')
                self.save_state(save_dir, resumable=True)

                #### DELETE PRIOR INTERIM EPOCH?????

                assert self.train and not self.validate_ckpt_dir
                if self.validate and not self.submit_evaluation:
                    self.validation_epoch_wrapper(epoch, step + self.starting_step)
                    self.set_train()
                elif self.validate and self.submit_evaluation:
                    save_dir = os.path.join(self.exp_trial_dir, f'partial_validate_epoch_{epoch}_step_{step + self.starting_step}')
                    self.save_state(save_dir, evaluate=True)
                    self.submit_validate_job(ckpt_dir=save_dir)
        
        if self.is_node_0_rank_0:         
            pbar.close()                

        train_epoch_duration_hrs = (time.time() - start_time) / 3600
        
        scores = {
            'stage': 'train_epoch',
            'epoch': epoch,
            'step': step + self.starting_step,
            'train_loss': total_loss / len(self.train_dataloader),
            'train_epoch_duration_hrs': train_epoch_duration_hrs,
            'gpu_allocated_memory_gb': torch.cuda.memory_allocated(self.accelerator.device) / 1e9,
            'gpu_reserved_memory_gb': torch.cuda.memory_reserved(self.accelerator.device) / 1e9,
            'gpu_max_allocated_memory_gb': torch.cuda.max_memory_allocated(self.accelerator.device) / 1e9,
            'gpu_max_reserved_memory_gb': torch.cuda.max_memory_reserved(self.accelerator.device) / 1e9,
            'cpu_memory_gb': psutil.Process().memory_info().rss / 1e9,
        }
        
        self.print(f'[training_epoch]: {epoch} train scores:')
        if self.is_node_0_rank_0:
            pprint.pprint(scores)
        self.accelerator.log(scores, step=epoch)

        if not self.debug:

            last_ckpt_dir = os.path.join(self.exp_trial_dir, f'completed_epoch_{epoch}_step_{step + self.starting_step}')

            self.save_state(last_ckpt_dir, resumable=True, evaluate=True)
            
            # Remove the prior last epoch checkpoints (if not using submit_evaluation):
            if self.is_node_0_rank_0 and not self.submit_evaluation:
                for i in os.listdir(self.exp_trial_dir):
                    if i.startswith('completed_epoch_') and i != os.path.basename(last_ckpt_dir):
                        shutil.rmtree(os.path.join(self.exp_trial_dir, i))
            
            if self.is_node_0_rank_0 and self.latest_ckpt_dir is not None and 'partial_epoch_' in self.latest_ckpt_dir:
                shutil.rmtree(self.latest_ckpt_dir)
                self.latest_ckpt_dir = None
            self.accelerator.wait_for_everyone()
                
        self.print(f'[training_epoch]: end of epoch: {epoch}; Step: {step + self.starting_step}; train dataloader length: {len(self.train_dataloader)}.')

        self.last_step = step + self.starting_step
        self.starting_step += step + 1
    
    def training_step(self, batch):
        raise NotImplementedError
                  
    def validation_epoch_wrapper(self, epoch=None, step=None, ckpt_dir=None):
                
        if not self.train and ckpt_dir:
                    
            if self.debug: 
                epoch = -1
            else:
                self.print(f'[validation_epoch_wrapper]: loading checkpoint for validation: {ckpt_dir}.')
                self.load_state(ckpt_dir, evaluate=True)
                
                epoch_match = re.search(r'_epoch_(\d+)', ckpt_dir)
                step_match = re.search(r'step_(\d+)', ckpt_dir)
                epoch = int(epoch_match.group(1))
                step = int(step_match.group(1))
                    
        start_time = time.time()
        
        self.set_eval()
        
        scores = self.validation_epoch(epoch)
        
        assert isinstance(scores, dict), f'[validation_epoch_wrapper]: validation_epoch must return a dictionary of the scores; got {type(scores)}.'

        scores.update(
            {
                'stage': 'val',
                'val_epoch_duration_hrs': (time.time() - start_time) / 3600,
                'epoch': epoch,
                'step': step,
            },
        )
        
        self.print(f'[validation_epoch_wrapper]: epoch {epoch}, step {step} validation scores:')
        if self.is_node_0_rank_0:
            pprint.pprint(scores)
        self.accelerator.log(scores, step=step)
        
        if not self.debug:
            
            # Save the best checkpoint:
            best_ckpt_dir, best_score = get_best_ckpt(self.exp_trial_dir, self.monitor, self.monitor_mode)
            assert self.monitor in scores, f'[validation_epoch_wrapper]: {self.monitor} not in scores; available keys: {scores.keys()}.'
            new_ckpt_dir = os.path.join(self.exp_trial_dir, f'epoch={epoch}_{self.monitor}={scores[self.monitor]:.4f}')
            improvement = scores[self.monitor] > best_score if self.monitor_mode == 'max' else scores[self.monitor] < best_score
            if best_ckpt_dir is None or improvement:
                ckpt_dir_list = glob.glob(f'{self.exp_trial_dir}/completed_epoch_{epoch}_step_{step}') + glob.glob(f'{self.exp_trial_dir}/partial_epoch_{epoch}_step_{step}')
                assert len(ckpt_dir_list) == 1, f'[validation_epoch_wrapper]: expected one completed epoch directory, got: {ckpt_dir_list}.'
                
                if self.is_node_0_rank_0:
                    self.copy_dir(ckpt_dir_list[0], new_ckpt_dir)
                self.accelerator.wait_for_everyone()
                if best_ckpt_dir is not None:
                    if self.is_node_0_rank_0:
                        shutil.rmtree(best_ckpt_dir)

            # Remove the partial epoch validation checkpoint:
            if ckpt_dir is not None and 'partial_validate_' in ckpt_dir and not self.train:
                if self.is_node_0_rank_0:
                    shutil.rmtree(ckpt_dir)
            """
            not sure if this is working correctly, in particular, finding the very last epoch dir.


            !!!! ALSO, DO NOT DELETE IF IT IS THE BEST CHECKPOINT
            """
            # Remove the validated checkpoint if its not the last epoch:
            # elif ckpt_dir is not None and self.is_node_0_rank_0 and not self.train:
            #     completed_epoch_dirs = [f for f in os.listdir(self.exp_trial_dir) if f.startswith('completed_epoch_')]
            #     highest_epoch_dir = max(completed_epoch_dirs, key=lambda x: int(x.split('_')[-1]))
            #     if ckpt_dir != highest_epoch_dir:
            #         shutil.rmtree(ckpt_dir)
                    
        return epoch
                                                    
    def validation_epoch(self, epoch):
        raise NotImplementedError
    
    def test_epoch_wrapper(self):
        
        best_ckpt_dir = self.get_best_ckpt(self.exp_trial_dir)

        # Validate the best checkpoint directory:
        if (self.debug and best_ckpt_dir is None) or self.test_pretrained: 
            epoch = -1
        else:
            self.print(f'[test_epoch_wrapper]: loading checkpoint for testing: {best_ckpt_dir}.')   

            self.load_state(best_ckpt_dir, evaluate=True)
            epoch = int(re.search(r'epoch=(\d+)', best_ckpt_dir).group(1))
            
        self.set_eval()
        
        scores = self.test_epoch(epoch)

        assert isinstance(scores, dict), f'[test_epoch_wrapper]: test_epoch must return a dictionary of the scores; got {type(scores)}.'

        scores.update({'stage': 'test', 'epoch': epoch, 'step': float('nan')})

        self.print(f'[test_epoch_wrapper]: epoch {epoch} test scores:')
        if self.is_node_0_rank_0:
            pprint.pprint(scores)
        self.accelerator.log(scores, step=epoch)
                    
    def test_epoch(self):
        raise NotImplementedError
        
    def set_train(self):
        try:
            self.model.train()
        except AttributeError as e:
            raise AttributeError(
                "The model attribute is not defined or not a torch.nn.Module. "
                "Please ensure `self.model` is set correctly or overwrite `set_train` in your class."
            ) from e

    def set_eval(self):
        try:
            self.model.eval()
        except AttributeError as e:
            raise AttributeError(
                "The model attribute is not defined or not a torch.nn.Module. "
                "Please ensure `self.model` is set correctly or overwrite `set_eval` in your class."
            ) from e

    def print(self, *args):
        if self.is_node_0_rank_0:
            print(*args)

    def sig_handler(self, signum, frame):

        self.print(f'[sig_handler]: caught signal: {signum}. Scheduling partial checkpoint save for training loop.')
        self.save_partial_ckpt = True

    def submit_validate_job(self, ckpt_dir):

        self.print('[submit_validate_job]: submitting validation job...')
        
        cmd_line_args = read_command_line_arguments()
        
        assert cmd_line_args.manager_script_path is not None
        
        # Load arguments for session:
        session = re.search(r'\/session_(\d+)', cmd_line_args.manager_script_path).group(1)
        args_path = os.path.join(self.exp_trial_dir, 'arguments', f'session_{session}.yaml')
        with open(args_path, 'r') as f:
            args = yaml.safe_load(f)
            args = DefaultMunch.fromDict(args)
        
        args.train = None
        args.validate = True
        args.validate_ckpt_dir = ckpt_dir
        args.validate_remaining_ckpts = False
        args.accelerate_config = args.evaluate_accelerate_config if args.evaluate_accelerate_config is not None else args.accelerate_config
        args.manager_script_path = None
                        
        if self.submit_evaluation_kwargs is not None:
            for k, v in self.submit_evaluation_kwargs.items():
                setattr(args, k, v)      
                        
        args = submit_stages(
            args, self.is_node_0_rank_0, exit_after_submit=False, email_on_complete=False, run_cmd='sbatch --export=PATH,LD_LIBRARY_PATH,PYTHONPATH,HF_HOME', job_type='validation'
        )
        
    def submit_test_job(self):
        cmd_line_args = read_command_line_arguments()
        
        # Load arguments for session:
        session = re.search(r'\/session_(\d+)', cmd_line_args.manager_script_path).group(1)
        args_path = os.path.join(self.exp_trial_dir, 'arguments', f'session_{session}.yaml')
        with open(args_path, 'r') as f:
            args = yaml.safe_load(f)
            args = DefaultMunch.fromDict(args)
        
        args.train = None
        args.validate = None
        args.validate_ckpt_dir = None
        args.test = True
        args.test_job = True
        args.accelerate_config = args.evaluate_accelerate_config if args.evaluate_accelerate_config is not None else args.accelerate_config
        args.manager_script_path = None

        if self.submit_evaluation_kwargs is not None:
            for k, v in self.submit_evaluation_kwargs.items():
                setattr(args, k, v)      
                        
        args = submit_stages(
            args, self.is_node_0_rank_0, exit_after_submit=False, email_on_complete=False, run_cmd='sbatch --export=PATH,LD_LIBRARY_PATH,PYTHONPATH,HF_HOME', job_type='test'
        )

    @staticmethod
    def get_args_and_submit_job(args: object = None):
        
        if args is None:
            cmd_line_args = read_command_line_arguments()
            args = load_config_and_update_args(cmd_line_args=cmd_line_args)
                                                  
        is_main_process = os.environ['RANK'] == '0' if 'RANK' in os.environ else True

        if args.submit:
            args = submit_stages(args, is_main_process, email_on_complete=False)
            
        if is_main_process:
            pprint.pprint(args.__dict__)
            
        return args

    def save_state(self, save_dir, resumable=False, evaluate=False):

        assert resumable or evaluate, '[save_state]: Either resumable or evaluate must be True.'
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        resumable_dir = os.path.join(save_dir, 'resumable')
        Path(resumable_dir).mkdir(parents=True, exist_ok=True)

        evaluate_dir = os.path.join(save_dir, 'evaluate')
        Path(evaluate_dir).mkdir(parents=True, exist_ok=True)

        if evaluate and self.transformers_model:
            self.print(f'[save_state]: Saving a safetensors checkpoint at {evaluate_dir}...')
            self.accelerator.wait_for_everyone()
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.save_pretrained(
                evaluate_dir,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=self.accelerator.get_state_dict(self.model),
            )
            self.accelerator.wait_for_everyone()
            evaluate = False

            assert os.path.isdir(evaluate_dir), f'[save_state]: Directory {evaluate_dir} was not created.'
            assert os.listdir(evaluate_dir), f'[save_state]: Directory {evaluate_dir} is empty.'

        if resumable or evaluate:
            self.print(f'[save_state]: Saving the state at {resumable_dir}...')
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(resumable_dir)
            self.accelerator.wait_for_everyone()

            assert os.path.isdir(resumable_dir), f'[save_state]: Directory {resumable_dir} was not created.'
            assert os.listdir(resumable_dir), f'[save_state]: Directory {resumable_dir} is empty.'

    def load_state(self, load_dir, evaluate=False):
        evaluate = False if not self.transformers_model else evaluate

        resumable_dir = os.path.join(load_dir, 'resumable')
        evaluate_dir = os.path.join(load_dir, 'evaluate')

        model = self.model.module if hasattr(self.model, 'module')  else self.model
        is_peft_model = isinstance(model, PeftModel) or hasattr(self.model, 'peft_config')

        if evaluate:
            self.print(f'[load_state]: Loading checkpoint for evaluation: {evaluate_dir}...')
            # if self.train is not None:
            #     del self.model
            #     self.init_model()
            if hasattr(self.model, 'module'):
                if is_peft_model:
                    base_model = self.model.module.unload()
                    self.model = PeftModel.from_pretrained(base_model, evaluate_dir, config=self.peft_config)
                else:
                    try:
                        self.model = self.model.module.from_pretrained(evaluate_dir, config=self.model.module.config)
                    except Exception as e_1:
                        try:
                            self.print(f'[load_state]: Failed to load from evaluate_dir ({e_1}), attempting to load from resumable_dir...')
                            self.accelerator.load_state(resumable_dir)
                        except Exception as e_2:
                            raise RuntimeError(f'[load_state]: Both evaluate ({e_1}) and resumable ({e_2}) checkpoint directories failed to load.')
            else:
                if is_peft_model:
                    base_model = self.model.unload()
                    self.model = PeftModel.from_pretrained(base_model, evaluate_dir, config=self.peft_config)
                else:
                    try:
                        self.model = self.model.from_pretrained(evaluate_dir, config=self.model.config)
                    except Exception as e_1:
                        try:
                            self.print(f'[load_state]: Failed to load from evaluate_dir ({e_1}), attempting to load from resumable_dir...')
                            self.accelerator.load_state(resumable_dir)
                        except Exception as e_2:
                            raise RuntimeError(f'[load_state]: Both evaluate ({e_1}) and resumable ({e_2}) checkpoint directories failed to load.')
            
            # This check does not account for non-DDP distributed methods:
            if not isinstance(self.model, DistributedDataParallel):
                self.model = self.accelerator.prepare_model(self.model)

            if self.debug:
                try:
                    self.model.processor = self.processor
                except Exception:
                    pass
        else:
            self.print(f'[load_state]: Resuming from state: {resumable_dir}...')
            self.accelerator.load_state(resumable_dir)

    def save_partial_ckpt_and_requeue(self, epoch, step):

        self.print(f'[save_partial_ckpt_and_requeue]: saving partial checkpoint for epoch {epoch}, step {step}.')
        self.save_state(os.path.join(self.exp_trial_dir, f'partial_epoch_{epoch}_step_{step}'), resumable=True)

        if self.is_node_0_rank_0 and self.latest_ckpt_dir is not None and 'partial_epoch_' in self.latest_ckpt_dir:
            shutil.rmtree(self.latest_ckpt_dir)
        self.accelerator.wait_for_everyone()

        result = 0
        if self.submit:            
            if self.is_node_0_rank_0:
                job_id = os.environ['SLURM_JOB_ID']

                self.print(f'\n[save_partial_ckpt_and_requeue]: requeing job {job_id}...')

                cmd = f'scontrol requeue {job_id}'
                result = call(cmd, shell=True)
            
                if result == 0:
                    self.print(f'[save_partial_ckpt_and_requeue]: requeued job {job_id}.')
                else:
                    self.print('[save_partial_ckpt_and_requeue]: requeue failed...')

        self.save_partial_ckpt = False

        self.accelerator.end_training()
        
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        sys.exit(result)
      
    def get_validated_epochs(self):
        metrics_file = os.path.join(self.exp_trial_dir, 'metrics.csv')
        validated_epochs = set()
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            if 'stage' in df.columns and 'epoch' in df.columns:
                epochs = pd.to_numeric(df.loc[df['stage'] == 'val', 'epoch'], errors='coerce')
                validated_epochs = set(epochs.dropna().astype(int).tolist())
        return validated_epochs

    def get_best_ckpt(self, exp_trial_dir):

        best_ckpt_dir, _ = get_best_ckpt(exp_trial_dir, self.monitor, self.monitor_mode)
        if not (self.debug or self.test_pretrained) and best_ckpt_dir is None:
            self.print(f'[get_best_ckpt]: no checkpoint found for testing. Attempting to save and load a checkpoint for the best performing completed epoch.')

            metrics_file = os.path.join(exp_trial_dir, 'metrics.csv')
            df = pd.read_csv(metrics_file)
            val = df[df['stage'] == 'val']
            if val.empty:
                raise RuntimeError(f'No validation entries in {metrics_file}; cannot determine the best epoch for testing.')
            best_idx = val[self.monitor].idxmax() if self.monitor_mode == 'max' else val[self.monitor].idxmin()
            best_epoch = int(val.at[best_idx, 'epoch'])
            best_score = val.at[best_idx, self.monitor]
            completed_ckpt_dir = glob.glob(f'{exp_trial_dir}/completed_epoch_{best_epoch}_step_*')[0]
            assert os.path.isdir(completed_ckpt_dir), f'[get_best_ckpt]: completed checkpoint directory not found: {completed_ckpt_dir}.'
            assert os.listdir(completed_ckpt_dir), f'[get_best_ckpt]: completed checkpoint directory is empty: {completed_ckpt_dir}.'
            best_ckpt_dir = os.path.join(exp_trial_dir, f'epoch={best_epoch}_{self.monitor}={best_score:.4f}')
            if self.is_node_0_rank_0:
                self.copy_dir(completed_ckpt_dir, best_ckpt_dir)
            self.accelerator.wait_for_everyone()

        return best_ckpt_dir

    def copy_dir(self, src_dir, dst_dir, ignore_errors=False):
        self.print(f'[copy_dir]: copying directory from {src_dir} to {dst_dir}...')
        os.makedirs(dst_dir, exist_ok=True)
        for root, _, files in os.walk(src_dir):
            rel_path = os.path.relpath(root, src_dir)
            target_root = os.path.join(dst_dir, rel_path)
            os.makedirs(target_root, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_root, file)
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    if not ignore_errors:
                        raise
                    self.print(f'[copy_dir]: warning: failed to copy {src_file} → {dst_file}: {e}')

    @staticmethod
    def get_safetensors_state_dict(ckpt_dir):
        root = pathlib.Path(ckpt_dir)
        index = json.load(open(root / 'model.safetensors.index.json'))
        state_dict = {}
        for v in set(index['weight_map'].values()):
            state_dict.update(load_file(root / v))
        missing = set(index['weight_map'].keys()) - state_dict.keys()
        assert not missing, f'[get_safetensors_state_dict]: missing keys in state_dict: {missing}'
        return state_dict
        
    @staticmethod
    def find_tied_parameters(model: torch.nn.Module):
        param_by_ptr = {}
        tied = []
        for name, p in model.named_parameters(recurse=True, remove_duplicate=False):
            try:
                ptr = p.untyped_storage().data_ptr()
            except AttributeError:
                ptr = p.data_ptr()

            if ptr in param_by_ptr:
                first_name, nelem = param_by_ptr[ptr]
                tied.append((name, first_name, nelem))
            else:
                param_by_ptr[ptr] = (name, p.numel())

        return tied
    
    def load_safetensors_state_dict(self, model: torch.nn.Module, ckpt_dir, rename_state_dict_keys: Optional[dict] = None):
        self.print(f'[load_safetensors_state_dict]: loading safetensors state_dict from {ckpt_dir}...')
        state_dict = self.get_safetensors_state_dict(ckpt_dir)
        
        tied_params = self.find_tied_parameters(model)
        if tied_params:
            self.print(f'[load_safetensors_state_dict]: found tied parameters: {tied_params}.')

        if rename_state_dict_keys:
            rename_map = {}
            for k in list(state_dict.keys()):
                new_key = k
                for old, new in rename_state_dict_keys.items():
                    if new_key.startswith(old):
                        new_key = new_key.replace(old, new, 1)
                    elif old in new_key:
                        new_key = new_key.replace(old, new)
                if new_key != k:
                    rename_map[k] = new_key

            for old_k, new_k in rename_map.items():
                state_dict[new_k] = state_dict.pop(old_k)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        first_tied = {name for name, _, _ in tied_params}
        missing_not_tied = set(missing_keys) - first_tied
        assert not missing_not_tied, f'[load_safetensors_state_dict]: missing keys not in tied parameters: {missing_not_tied}.'
        assert not unexpected_keys, f'[load_safetensors_state_dict]: unexpected keys present in state_dict: {unexpected_keys}.'

        self.print('[load_safetensors_state_dict]: state_dict loaded successfully.')
