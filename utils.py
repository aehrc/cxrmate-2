import copy
import glob
import importlib
import json
import os
import re
import shutil
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate.tracking import GeneralTracker, on_main_process
from huggingface_hub import HfApi, hf_hub_download
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from torchmetrics import Metric
from transformers import PreTrainedTokenizerFast

try:
    from slurm import SlurmSubmit
except ImportError:
    from .slurm import SlurmSubmit


class TrainStepScoresMetric(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('rows', default=[], dist_reduce_fx=self.dist_reduce_fx)

    @staticmethod
    def dist_reduce_fx(rows):
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, rows)
        gathered = [j for i in gathered for j in i]
        return gathered
    
    def update(self, scores: dict):
        self.rows.append(scores)

    def compute(self):
        return pd.DataFrame(self.rows).mean().to_dict()


def get_score_from_ckpt_path(ckpt_path: str) -> float:
    return float(re.findall(r'[-+]?\d*\.\d+|\d+', ckpt_path.rsplit('=', 1)[1])[0])


def get_best_ckpt(
    exp_dir: str,
    monitor: str,
    monitor_mode: str,
) -> str:
    """
    Get the best performing checkpoint from the experiment directory.

    Argument/s:
        exp_dir - Experiment directory where the checkpoints are saved.
        monitor - Monitored metric.
        monitor_mode - Metric monitoring mode, either "min" or "max".

    Returns:
        Path to the epoch's checkpoint.
    """

    ckpt_list = glob.glob(os.path.join(exp_dir, f'epoch=*_{monitor}=*'))

    if not ckpt_list:
        warnings.warn(
            f'No checkpoints exist for the regex: epoch=*_{monitor}=* in the checkpoint directory: {exp_dir}.'
        )
        return None, -float('inf') if monitor_mode == 'max' else float('inf')

    scores = [get_score_from_ckpt_path(i) for i in ckpt_list]

    if monitor_mode == 'max':
        ckpt_path = ckpt_list[np.argmax(scores)]
        best_score = np.max(scores)
    elif monitor_mode == 'min':
        ckpt_path = ckpt_list[np.argmin(scores)]
        best_score = np.min(scores)
    else:
        raise ValueError(
            '"monitor_mode" must be max or min, not {}.'.format(monitor_mode)
        )
    return ckpt_path, best_score


class CSVTracker(GeneralTracker):
    name = 'csv'
    requires_logging_directory = True

    @on_main_process
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'metrics.csv')

    @property
    def tracker(self):
        return self.log_dir

    @on_main_process
    def store_init_configuration(self, values: Dict[str, Any]):
        config_file = os.path.join(self.log_dir, 'hparams.yaml')
        with open(config_file, mode='w') as f:
            yaml.dump(values, f)

    @on_main_process
    def log(self, values: Dict[str, Any], step: Optional[int] = None):
        if step is not None:
            values['step'] = step
        if 'epoch' not in values or 'stage' not in values:
            raise ValueError("Both 'epoch' and 'stage' keys must be included in the values for proper tracking.")

        # Update table:
        key = (values['stage'], values['epoch'], values['step'])
        self.composite_key_dict = self._load_existing_file()
        if key in self.composite_key_dict:
            self.composite_key_dict[key].update(values)
        else:
            self.composite_key_dict[key] = values

        # Save to file:
        df = pd.DataFrame(self.composite_key_dict.values())
        known = ['test', 'val', 'train_epoch', 'train_step']
        present = pd.Index(df['stage'].unique())
        categories = list(dict.fromkeys(known + [s for s in present if s not in known]))
        df['stage'] = pd.Categorical(df['stage'], categories=categories, ordered=True)
        df = df.sort_values(by=['stage', 'epoch', 'step']).reset_index(drop=True)

        df.to_csv(self.log_file, index=False)
        
    def _load_existing_file(self) -> Dict[tuple, Dict[str, Any]]:
        if os.path.exists(self.log_file):
            try:
                df = pd.read_csv(self.log_file)
                composite_key_dict = {
                    (row['stage'], row['epoch'], row['step']): row.to_dict() for _, row in df.iterrows()
                }
            except Exception as e:
                shutil.move(self.log_file, os.path.join(self.log_dir, 'metrics_corrupt.csv'))
                composite_key_dict = {}
        else:
            composite_key_dict = {}
        return composite_key_dict


def compute_time_delta(event_time, reference_time, to_tensor=True):
    time_delta = reference_time - event_time
    time_delta = time_delta.total_seconds()
    assert isinstance(time_delta, float), f'time_delta should be float, not {type(time_delta)}.'
    if time_delta < 0:
        raise ValueError(f'time_delta should be greater than or equal to zero, not {time_delta}.')
    if to_tensor: 
        time_delta = torch.tensor(time_delta)
    return time_delta


def importer(
    definition: str,
    module: str,
    work_dir: Optional[str] = None,
):
    """
    Loads an instance of the definition from the given module. Can also load from an absolute or relative file path.
    Arguments:
        definition - the definition in the module. Typically the name of a function or class.
        module - the module name (e.g., 'tasks.mnist.datamodule') or a relative/absolute path to a Python file.
        work_dir - working directory; loads the module relative to this directory (ignored if `module` is an absolute path).
    Returns:
        An instance of 'definition'.
    """
    module_path = Path(module)
    # If module is an absolute or relative path to a Python file:
    if module_path.suffix == ".py":
        # Convert to absolute path if it's relative:
        if not module_path.is_absolute():
            if work_dir:
                module_path = Path(work_dir) / module_path
            else:
                module_path = module_path.resolve()
        if not module_path.is_file():
            raise FileNotFoundError(f"Python file '{module_path}' does not exist.")
        module_name = module_path.stem  # Extracts filename without .py extension.
        loader = importlib.machinery.SourceFileLoader(module_name, str(module_path))
        spec = importlib.util.spec_from_loader(module_name, loader)
    # Handle case where `module` is a relative module path with `work_dir`:
    elif work_dir:
        absolute_path = Path(work_dir) / Path(*module.split(".")) .with_suffix(".py")
        if not absolute_path.is_file():
            raise FileNotFoundError(
                f"{absolute_path} does not exist. The target definition and module: {definition} & {module}. "
                f"Working directory: {work_dir}."
            )
        module_name = module.split(".")[-1]  # Use the last part of the module name.
        loader = importlib.machinery.SourceFileLoader(module_name, str(absolute_path))
        spec = importlib.util.spec_from_loader(module_name, loader)
    # Standard Python module import:
    else:
        try:
            return getattr(importlib.import_module(module), definition)
        except ModuleNotFoundError as e:
            raise ImportError(f"Failed to import module '{module}'. Ensure it is installed and accessible.") from e
        except AttributeError as e:
            raise ImportError(f"Definition '{definition}' not found in module '{module}'.") from e
    # Ensure module specification is valid:
    if spec is None:
        raise ImportError(f"Could not create a module specification for '{module}'.")
    module_obj = importlib.util.module_from_spec(spec)
    loader.exec_module(module_obj)
    if not hasattr(module_obj, definition):
        raise ImportError(f"Definition '{definition}' not found in module '{module}'.")
    return getattr(module_obj, definition)


def load_config_and_update_args(
    cmd_line_args: Namespace, print_args: bool = False
) -> None:
    """
    Loads the configuration YAML file and updates the args object.

    Argument/s:
        cmd_line_args - command line arguments object.
        print_args - print the arguments for the job.
    """
    # Make a deepcopy of the command line arguments:
    args = copy.deepcopy(cmd_line_args)

    # Add the working directory to paths:
    args.work_dir = args.work_dir if "work_dir" in args else None
    if not args.work_dir:
        args.work_dir = os.getcwd()

    # Configuration:
    if args.config.endswith((".yaml", ".yml")):
        args.config_file_name = args.config
        args.config = args.config.rsplit(".", 1)[0]
    else:
        if os.path.exists(os.path.join(args.work_dir, args.config + ".yaml")):
            args.config_file_name = args.config + ".yaml"
        elif os.path.exists(os.path.join(args.work_dir, args.config + ".yml")):
            args.config_file_name = args.config + ".yml"
        else:
            raise FileNotFoundError(f"Configuration file not found (with either .yml or .yaml as the extension): {args.config}.")

    args.config_name = Path(args.config).parts[-1]

    # Load configuration using Hydra's Compose API:
    args.config_dir = str(Path(args.config).parent)
    if not os.path.isabs(args.config_dir):
        args.config_dir = os.path.join(args.work_dir, args.config_dir)
    assert isinstance(args.config_dir, str)
            
    with initialize_config_dir(version_base=None, config_dir=args.config_dir):
        config = OmegaConf.to_container(
            compose(config_name=args.config_name), resolve=True
        )

    # Command line arguments overwrite configuration attributes if the command line arguments are not None:
    args.config_full_path = os.path.join(args.work_dir, args.config_file_name)
    for k, v in config.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
        else:
            if getattr(args, k) != v:
                warnings.warn(
                    f'Command line argument "--{k} {getattr(args, k)}" '
                    f'({type(getattr(args, k))}) will overwrite configuration argument "{k}: {v}" from '
                    f"{args.config_full_path} ({type(v)})."
                )

    # The model name must be defined in the configuration or in the command line arguments:
    # assert (args.module), f'"module" must be specified as a command line argument or in {args.config_full_path}.'
    # assert (args.definition), f'"definition" must be specified as a command line argument or in {args.config_full_path}.'
    assert (args.exp_dir), f'"exp_dir" must be specified as a command line argument or in {args.config_full_path}.'

    # Defaults: There is probably a better place to do this:
    args.num_workers = args.num_workers if args.num_workers is not None else 1
    args.num_nodes = args.num_nodes if args.num_nodes is not None else 1
    args.devices = args.devices if args.devices is not None else 1

    # Add the task, configuration name, and the trial number to the experiment directory:
    args.trial = args.trial if args.trial is not None else 0
    if args.exp_trial_dir is None:
        args.exp_trial_dir = os.path.join(
            args.exp_dir, args.task, args.config_name, "trial_" + f"{args.trial}"
        )
    Path(args.exp_trial_dir).mkdir(parents=True, exist_ok=True)

    # Prevent auto_resubmit and resume_last if not resume_last:
    args.resume_last, args.auto_resubmit = True, True
    if args.resume_ckpt_path or args.resume_epoch:
        args.resume_last, args.auto_resubmit = False, False

    if print_args:
        print(f"args: {args.__dict__}")

    return args

def submit_stages(args, is_main_process, exit_after_submit=True, email_on_complete=True, run_cmd='sbatch', job_type=''):

    # Slurm object:
    cluster = SlurmSubmit(
        args=args,
        task=args.task,
        config=args.config,
        trial=args.trial,
        is_main_process=is_main_process,
        save_dir=args.exp_trial_dir,
        manager_script_path=args.manager_script_path,
        time_limit=args.time_limit,
        num_gpus=args.devices,
        num_nodes=args.num_nodes,
        num_workers=args.num_workers,
        memory=args.memory,
        python_cmd='python3' if not hasattr(args, 'python_cmd') else args.python_cmd,
        no_cpus_per_task=args.no_cpus_per_task,
        no_gpus_per_node=args.no_gpus_per_node,
        no_ntasks_per_node=1,
        srun_options=args.srun_options,
        email=args.email, 
        email_on_complete=email_on_complete, 
        email_on_fail=True,
        auto_resubmit=False,
        exit_after_submit=exit_after_submit,
        run_cmd=run_cmd,
        job_type=job_type,
    )

    # Source virtual environment:
    if args.venv_path:
        cluster.add_command('source ' + args.venv_path)

    # Add options to the cluster manager submission script:
    if not 'cluster_manager_script_commands' in args:
        args.cluster_manager_script_commands = []        
    args.cluster_manager_script_commands.append(f'export GPUS_PER_NODE={args.devices}')
    args.cluster_manager_script_commands.append('HEAD_NODE_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)')
    args.cluster_manager_script_commands.append('echo "Head node: $HEAD_NODE_IP."')
    for i in args.cluster_manager_script_commands:
        cluster.add_command(i)
       
    accelerate_config = args.accelerate_config if (args.train or (args.evaluate_accelerate_config is None)) else args.evaluate_accelerate_config

    accelerate_exec_path = 'accelerate' if args.accelerate_exec_path is None else args.accelerate_exec_path
    cluster.entrypoint = f'{accelerate_exec_path} launch --config-file {accelerate_config} --main_process_ip $HEAD_NODE_IP --main_process_port 29500 --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) --num_machines $SLURM_NNODES --rdzv_backend c10d main.py'

    # Add options to the cluster manager submission script:
    if 'cluster_manager_script_clean_up_commands' in args:
        for i in args.cluster_manager_script_clean_up_commands:
            cluster.add_clean_up_command(i)

    # Add cluster manager options:
    if 'cluster_manager_options' in args:
        for i in args.cluster_manager_options:
            cluster.add_manager_option(option=i[0], value=i[1])

    # Request the quality of service for the job:
    if args.qos:
        cluster.add_manager_option(option='qos', value=args.qos)

    # Submit job to workload manager:
    job_display_name = args.task + '_' + args.config_name

    if args.trial is not None:
        job_display_name = job_display_name + f'_trial_{args.trial}'

    args = cluster.submit(job_display_name=job_display_name)
    
    return args
                

def copy_dir_ignoring_extensions(source_dir, destination_dir, ignore_extensions):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for root, _, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        dest_path = os.path.join(destination_dir, rel_path)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for file in files:
            if not file.endswith(tuple(ignore_extensions)):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, file)
                shutil.copy2(src_file, dest_file)


def rename_added_tokens(repo_id, token_map, repo_type='model'):

    path = hf_hub_download(repo_id=repo_id, filename='tokenizer.json', repo_type=repo_type)
    with open(path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)

    path = hf_hub_download(repo_id=repo_id, filename='tokenizer_config.json', repo_type=repo_type)
    with open(path, 'r', encoding='utf-8') as f:
        tokenizer_config_json = json.load(f)

    path = hf_hub_download(repo_id=repo_id, filename='special_tokens_map.json', repo_type=repo_type)
    with open(path, 'r', encoding='utf-8') as f:
        special_tokens_map_json = json.load(f)

    added_tokens = tokenizer_json['added_tokens']
    added_tokens_decoder = tokenizer_config_json['added_tokens_decoder']

    for old, new in token_map.items():

        for idx, token_dict in enumerate(added_tokens):
            if token_dict['content'] == old:
                added_tokens[idx]['content'] = new
                break

        for k, v in tokenizer_config_json.items():
            if v == old:
                tokenizer_config_json[k] = new
                break

        for k, v in added_tokens_decoder.items():
            if v['content'] == old:
                added_tokens_decoder[k]['content'] = new
                break

        for k, v in special_tokens_map_json.items():
            if v == old:
                special_tokens_map_json[k] = new
                break

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

        with open(os.path.join(tmp_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config_json, f, ensure_ascii=False, indent=2)

        with open(os.path.join(tmp_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map_json, f, ensure_ascii=False, indent=2)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp_dir)

    tokenizer.push_to_hub(repo_id=repo_id)



if __name__ == '__main__':
    import sys
    from pathlib import Path

    try:
        from command_line_arguments import read_command_line_arguments
    except ImportError:
        from .command_line_arguments import read_command_line_arguments

    cmd_line_args = read_command_line_arguments()
    args = load_config_and_update_args(cmd_line_args=cmd_line_args)

    if not getattr(args, 'stages_module', None): raise ValueError('stages_module must be provided.')
    if not Path(args.stages_module).exists(): raise FileNotFoundError(f'{args.stages_module} does not exist.')

    stages_module_path = Path(args.stages_module).resolve()
    if not stages_module_path.suffix == ".py":
        raise ValueError(f"{stages_module_path} is not a Python file.")
    sys.path.insert(0, str(stages_module_path.parent))

    if not getattr(args, 'stages_definition', None): 
        stages_definition = 'Stages'
    else:
        stages_definition = args.stages_definition

    try:
        Stages = importer(definition=stages_definition, module=args.stages_module)
    except ImportError as e:
        raise ImportError(f"Failed to import stages definition '{stages_definition}' from module '{args.stages_module}': {e}.")

    args = Stages.get_args_and_submit_job(args)
    stages = Stages(**args.__dict__)
    stages()
