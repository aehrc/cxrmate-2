import copy
import glob
import os
import re

import accelerate
import torch
from loguru import logger
from RaTEScore import RaTEScore
from rewards import ARNReward, BERTScoreReward, CXRBERTReward
from stages_cxrmate2 import Stages as BaseStages
from trl.trainer.utils import selective_log_softmax

logger.disable('PyRuSH')


class Stages(BaseStages):

    def __init__(
        self, 
        reward_weights: list,
        completions_top_p: float,
        completions_top_k: float,
        completions_temperature: float,
        beta, 
        epsilon, 
        iterations,
        group_size,
        max_train_prompt_len,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_weights = reward_weights
        self.completions_top_p = completions_top_p
        self.completions_top_k = completions_top_k
        self.completions_temperature = completions_temperature
        self.beta = beta
        self.epsilon = epsilon
        self.iterations = iterations
        self.group_size = group_size
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

    def init_model(self):
        super().init_model()

        # Reference model:
        if self.beta > 0:
            base = self.model.module if hasattr(self.model, "module") else self.model
            self.ref_model = copy.deepcopy(base)
            self.ref_model.eval()

        if self.accelerator.state.distributed_type.name == 'FSDP' and self.train:
            # Rewards:
            self.reward_ratescore = RaTEScore()
            self.reward_cxrbert = CXRBERTReward(device=self.accelerator.device)
            self.reward_bertscore = BERTScoreReward(device=self.accelerator.device, num_workers=self.num_workers)
            self.reward_arn = ARNReward(n=3, device=self.accelerator.device, tokenizer=self.processor.tokenizer)

    def accelerate_prepare(self):

        super().accelerate_prepare()

        if self.accelerator.state.distributed_type.name == 'FSDP' and self.train:

            assert self.accelerator.state.fsdp_plugin.fsdp_version == 2

            self.reward_ratescore.model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.reward_ratescore.model)
            self.reward_ratescore.eval_model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.reward_ratescore.eval_model)
            self.reward_cxrbert.model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.reward_cxrbert.model)
            self.reward_bertscore.bert_scorer._model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.reward_bertscore.bert_scorer._model)
            self.ref_model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.ref_model)

    def post_prepare(self):
        
        if self.accelerator.state.distributed_type.name != 'FSDP' and self.train:
            # Rewards:
            self.reward_ratescore = RaTEScore()
            self.reward_ratescore.model.to(device=self.accelerator.device)
            self.reward_ratescore.eval_model.to(device=self.accelerator.device)
            self.reward_cxrbert = CXRBERTReward(device=self.accelerator.device)
            self.reward_bertscore = BERTScoreReward(device=self.accelerator.device, num_workers=self.num_workers)
            self.reward_arn = ARNReward(n=3, device=self.accelerator.device, tokenizer=self.processor.tokenizer)

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
        https://huggingface.co/docs/trl/main/en/grpo_trainer
        """

        batch = batch.to(self.accelerator.device)
                
        findings_gt = batch.data.pop('findings', None)
        impression_gt = batch.data.pop('impression', None)
        
        # Disable gradient checkpointing for generation:
        if self.accelerator.state.distributed_type.name == 'MULTI_GPU':
            self.model.module.language_model.gradient_checkpointing_disable() 
        else:
            self.model.language_model.gradient_checkpointing_disable() 
        
        # Prompt size:
        prompt_len = batch['input_ids'].shape[1]

        # Group completions:
        gen_model = self.model.module if hasattr(self.model, 'module') else self.model
        completion_ids = gen_model.generate(
            do_sample=True,
            top_p=self.completions_top_p,
            top_k=self.completions_top_k,
            temperature=self.completions_temperature,
            bad_words_ids=[[self.processor.tokenizer.bos_token_id], [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]],
            generation_config=self.generation_config,
            num_return_sequences=self.group_size,
            **batch,
        )

        valid_mask = []
        for seq in completion_ids:
            bos = (seq == self.processor.tokenizer.bos_token_id).sum().item() == 1
            sep = (seq == self.processor.tokenizer.sep_token_id).sum().item() == 1
            eos = (seq == self.processor.tokenizer.eos_token_id).sum().item() == 1
            valid_mask.append(bos and sep and eos)
        any_valid = any(valid_mask)

        # Radiologist and group sample reports:
        prompt_len = batch['input_ids'].shape[-1]
        reports_gt = [f'{i} {j}'.strip() for i, j in zip(findings_gt, impression_gt, strict=True)]
        findings, impression = self.processor.split_and_decode_sections(completion_ids[:, prompt_len:])
        reports = [f'{i} {j}'.strip() for i, j in zip(findings, impression, strict=True)]
        reports_gt = [i for i in reports_gt for _ in range(self.group_size)]

        # Calculate advantages unless all invalid (in which case set zeros to keep sync):
        if any_valid:
            advantages, advantages_stats = self.advantages(reports, reports_gt)
        else:
            advantages = torch.zeros(len(reports), device=self.accelerator.device)
            advantages_stats = {
                'reward_ratescore_mean': 0.0,
                'reward_ratescore_std': 0.0,
                'reward_cxrbert_mean': 0.0,
                'reward_cxrbert_std': 0.0,
                'reward_bertscore_mean': 0.0,
                'reward_bertscore_std': 0.0,
                'reward_arn_mean': 0.0,
                'reward_arn_std': 0.0,
            }
        
        # Prepare batch for RL; expand to group size:
        for k in batch.keys():
            batch[k] = batch[k].repeat_interleave(repeats=self.group_size, dim=0)
        batch = self.processor.update_batch_for_rl(batch, completion_ids)
    
        # Number of completion tokens:
        num_completion_tokens = completion_ids.shape[1] - prompt_len
        label_ids = batch.data.pop('label_ids', None)
        completion_mask = batch.data.pop('completion_mask', None)

        # Zero out any invalid sampled sequences so they contribute no loss:
        completion_mask *= torch.tensor(valid_mask, device=completion_ids.device).unsqueeze(1)
        label_ids = label_ids[:, -num_completion_tokens:]
        
        # Get reference policy logits:
        if self.beta > 0:
            with torch.no_grad():
                ref_logits = self.ref_model(**batch).logits
                ref_logits = ref_logits[:, -num_completion_tokens:]
                ref_policy_completions_log_p = selective_log_softmax(ref_logits, label_ids)

        # Enable gradient checkpointing:
        if self.accelerator.state.distributed_type.name == 'MULTI_GPU':
            self.model.module.language_model.gradient_checkpointing_enable() 
        else:
            self.model.language_model.gradient_checkpointing_enable() 

        for i in range(self.iterations):
            
            # Logits from the current policy:
            logits = self.model(use_cache=False, **batch).logits

            # Get log probabilities of the policies:
            logits = logits[:, -num_completion_tokens:]
            cur_policy_completions_log_p = selective_log_softmax(logits, label_ids)
            
            if i == 0:
                old_policy_completions_log_p = cur_policy_completions_log_p.detach()
            
            # KL divergence between the model and reference model for the label tokens:
            if self.beta > 0:
                kl = torch.exp(ref_policy_completions_log_p - cur_policy_completions_log_p) - (ref_policy_completions_log_p - cur_policy_completions_log_p) - 1
            
            # Compute loss:
            coef_1 = torch.exp(cur_policy_completions_log_p - old_policy_completions_log_p)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if self.beta > 0:
                per_token_loss = per_token_loss + self.beta * kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp_min(1)
            
            if i > 0:
                old_policy_completions_log_p = cur_policy_completions_log_p.detach()

            self.accelerator.backward(loss)        
            self.optimiser.step()
            self.scheduler.step()
            self.optimiser.zero_grad()

            loss_detached = loss.item()
        
        # Sequence length:
        seq_len = torch.sum(completion_mask, dim=-1).float()
        
        scheduler_step = self.scheduler.scheduler.last_batch_iteration if hasattr(self.scheduler.scheduler, 'last_batch_iteration') else self.scheduler.scheduler._step_count
        
        accumulate_scores = {
            'train_loss': loss_detached, 
            'gpu_allocated_memory_gb': torch.cuda.memory_allocated(self.accelerator.device) / 1e9,
            'gpu_reserved_memory_gb': torch.cuda.memory_reserved(self.accelerator.device) / 1e9,
            'seq_len': torch.mean(seq_len).item(),
            'advantages_mean': advantages.mean().item(),
            'prompt_len': prompt_len,
            'valid_completions_fraction': sum(valid_mask) / len(valid_mask),
            **advantages_stats,
        }
        if self.beta > 0:
            accumulate_scores['kl'] = ((kl * completion_mask).sum() / completion_mask.sum()).item()

        step_scores = {
            'scheduler_lr': self.scheduler.get_last_lr()[-1],
            'scheduler_step': scheduler_step,
        }
                
        return loss_detached, step_scores, accumulate_scores

    def advantages(self, predictions, labels):
        
        advantages_stats = {}
        
        # Calculate individual advantages from reward groups
        with torch.no_grad():
           
            reward_ratescore = []
            for i, j in zip(predictions, labels, strict=True):
                try:
                    score = self.reward_ratescore.compute_score([i], [j])
                    reward_ratescore.append(score[0] if score is not None else 0.0)
                except Exception as _:
                    reward_ratescore.append(0.0)
                
            reward_ratescore = torch.tensor(reward_ratescore).to(device=self.accelerator.device)         
            advantages_ratescore, mean, std = self.calc_advantages_from_reward_group(reward_ratescore)
            advantages_stats['reward_ratescore_mean'] = mean.mean().item()
            advantages_stats['reward_ratescore_std'] = std.mean().item()

            reward_cxrbert = self.reward_cxrbert(predictions, labels)
            advantages_cxrbert, mean, std = self.calc_advantages_from_reward_group(reward_cxrbert)
            advantages_stats['reward_cxrbert_mean'] = mean.mean().item()
            advantages_stats['reward_cxrbert_std'] = std.mean().item()

            reward_bertscore = self.reward_bertscore(predictions, labels)
            advantages_bertscore, mean, std = self.calc_advantages_from_reward_group(reward_bertscore)
            advantages_stats['reward_bertscore_mean'] = mean.mean().item()
            advantages_stats['reward_bertscore_std'] = std.mean().item()
            
            reward_arn = self.reward_arn(predictions)
            advantages_arn, mean, std = self.calc_advantages_from_reward_group(reward_arn)
            advantages_stats['reward_arn_mean'] = mean.mean().item()
            advantages_stats['reward_arn_std'] = std.mean().item()
        
            # Weight individual advantages:
            advantages_ratescore = self.reward_weights[0] * advantages_ratescore
            advantages_cxrbert = self.reward_weights[1] * advantages_cxrbert
            advantages_bertscore = self.reward_weights[2] * advantages_bertscore
            advantages_arn = self.reward_weights[3] * advantages_arn

            # Composite advantages:
            advantages = advantages_cxrbert + advantages_bertscore + advantages_arn
            advantages = advantages_ratescore + advantages_cxrbert + advantages_bertscore + advantages_arn

        
        return advantages, advantages_stats
    
    def calc_advantages_from_reward_group(self, rewards):
        with torch.no_grad():

            # Calculate advantages from the reward groups:
            rewards_grouped = rewards.view(-1, self.group_size)
            mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
            std_rewards = rewards_grouped.std(dim=1, keepdim=True)
            advantages = (rewards - mean_rewards.repeat_interleave(self.group_size)) 
            advantages /= (std_rewards.repeat_interleave(self.group_size) + 1e-4)  # https://github.com/huggingface/trl/blob/e3244d2d096ff1e2e248c931d06d39e165e20623/trl/trainer/grpo_trainer.py#L863
            
        return advantages, mean_rewards, std_rewards
