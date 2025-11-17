import os
import time
from pathlib import Path
from typing import Optional
import torch
import accelerate
import pandas as pd


class NLGMetric():
    """
    Natural Language Generation (NLG) metrics for distributed computing.
    """
    def __init__(
        self, 
        mbatch_size: int = 1, 
        accelerator: accelerate.Accelerator = None,
    ):
        super().__init__()
        self.mbatch_size = mbatch_size
        self.epoch = None
        self.accelerator = accelerator
        self.save_dir = None
        self.save_time = None

    @staticmethod
    def mini_batch(iterable, mbatch_size=1):
        length = len(iterable)
        for i in range(0, length, mbatch_size):
            yield iterable[i:min(i + mbatch_size, length)]

    def update(self, **kwargs):
        raise NotImplementedError
    
    def init_metric(self):
        pass

    def cleanup_metric(self):
        pass

    def batch_scoring(self, synthetic, radiologist):
        raise NotImplementedError

    def accumulate_scores(self, df, epoch):
        raise NotImplementedError

    def metric_scoring(self, rows: Optional[list] = None):
        self.init_metric()
        input_rows, rows = rows, []
        for batch in self.mini_batch(input_rows, self.mbatch_size):

            synthetic = [i['synthetic'] for i in batch]
            radiologist = [i['radiologist'] for i in batch]
            study_ids = [i['study_id'] for i in batch]

            with torch.no_grad():
                try:
                    scores = self.batch_scoring(synthetic, radiologist)
                except Exception as e:
                    print(f'Error occurred while scoring batch for metric {self.metric_name}: {e}')
                    scores = [{} for _ in range(len(batch))]

            assert isinstance(scores, list), f'"scores" must be a list: {scores}.'
            assert all(isinstance(i, dict) for i in scores), f'Each element of "scores" must be a dict: {scores}.'

            scores = [{k: v if not torch.is_tensor(v) else v.item() for k, v in i.items()} for i in scores]

            scores = [{'study_id': study_id, **score_dict} for study_id, score_dict in zip(study_ids, scores)]

            rows.extend(scores)
        self.cleanup_metric()
        return rows

    def convert_lists_to_rows(self):
        raise NotImplementedError

    def compute(self, epoch: Optional[int] = None):
        self.epoch = epoch
        rows = self.convert_lists_to_rows()

        rows = self.metric_scoring(rows)

        if self.save_dir is None:
            raise ValueError('save_dir must be set before calling compute().')
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self.save_time = time.strftime("%d-%m-%Y_%H-%M-%S")
            else:
                self.save_time = None
            obj_list = [self.save_time]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            self.save_time = obj_list[0]
        else:
            self.save_time = time.strftime("%d-%m-%Y_%H-%M-%S")

        rank_csv_path = os.path.join(self.save_dir, f'rank_{rank}_{self.save_time}.csv')
        pd.DataFrame(rows).to_csv(rank_csv_path, index=False)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        all_rank_paths = [os.path.join(self.save_dir, f'rank_{r}_{self.save_time}.csv') for r in range(world_size)]
        df_list = []
        for p in all_rank_paths:
            df_list.append(pd.read_csv(p))

        df = pd.concat(df_list, ignore_index=True) if len(df_list) > 1 else df_list[0]

        return self.accumulate_scores(df, epoch)
    
    def reset(self):
        raise NotImplementedError


class CXRReportGenerationMetric(NLGMetric):
    """
    Torchmetric for metrics for CXR report generation evaluation.
    """
    def __init__(self, metric_name: str, split: str, exp_dir: str, **kwargs):
        """
        Argument/s:
            metric_name - name of the metric.
            split - dataset split.
            exp_dir - experiment directory where outputs will be saved.
        """
        super().__init__(**kwargs)

        self.metric_name = metric_name
        self.split = split
        self.exp_dir = exp_dir

        self.synthetic = []
        self.radiologist = []
        self.study_ids = []

        self.save_dir = os.path.join(self.exp_dir, 'metric_outputs', self.metric_name)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def update(self, synthetic, radiologist, study_ids):

        assert isinstance(synthetic, list), f'"synthetic" must be a list of strings: {synthetic}.'
        assert all(isinstance(i, str) for i in synthetic), f'Each element of "synthetic" must be a string: {synthetic}.'
        assert isinstance(radiologist, list), f'"labels" must be a list of lists, where each sub-list has a multiple strings: {radiologist}.'
        assert all(isinstance(i, str) for i in radiologist), f'Each element of "radiologist" must be a list of strings: {radiologist}.'

        self.synthetic.extend(synthetic)
        self.radiologist.extend(radiologist)
        self.study_ids.extend(study_ids)

    def convert_lists_to_rows(self):
        rows = []
        for (i_1, i_2, i_3) in zip(self.synthetic, self.radiologist, self.study_ids):
            rows.append(
                {
                    'synthetic': i_1,
                    'radiologist': i_2,
                    'study_id': i_3,
                }
            )

        return rows

    def accumulate_scores(self, df, epoch):

        df = df.fillna(0.0)

        # Drop duplicates caused by distributed:
        key = 'study_id'
        df = df.drop_duplicates(subset=[key])
        df = df.drop(columns=['synthetic', 'radiologist'], axis=1, errors='ignore')

        # Save the scores:
        def save_scores():
            df.to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_scores_{self.save_time}.csv',
                ),
                index=False,
            )
        if not torch.distributed.is_initialized():
            save_scores()
        elif torch.distributed.get_rank() == 0:
            save_scores()

        # Number of examples:
        prefix = f'{self.split}_{self.metric_name}_'
        scores = {f'{prefix}num_study_ids': float(df.study_id.nunique())}

        df = df.drop(['study_id'], axis=1)
        mean_scores = {f'{prefix}{k}': v for k, v in df.mean().to_dict().items()}
        scores = {**mean_scores, **scores}
        scores.pop('study_id', None)

        return scores
    
    def reset(self):
        self.synthetic = []
        self.radiologist = []
        self.study_ids = []