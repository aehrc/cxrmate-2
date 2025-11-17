import itertools
import os
import time
from pathlib import Path

import pandas as pd
import torch
from metrics.base import CXRReportGenerationMetric
from torchmetrics import Metric


class SectionsLogger(CXRReportGenerationMetric):
    def __init__(self, metric_name='sections', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)

        self.rows = []

    def update(self, findings, impression, study_ids):

        self.findings.extend(findings)
        self.impression.extend(impression)
        self.study_ids.extend(study_ids)
    
    def compute(self, epoch):

        rows = []
        for (i_1, i_2, i_3) in zip(self.findings, self.impression, self.study_ids):
            rows.append(
                {
                    'findings': i_1,
                    'impression': i_2,
                    'study_id': i_3,
                }
            )

        if torch.distributed.is_initialized():  # If DDP
            rows_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rows_gathered, rows)
            rows = [j for i in rows_gathered for j in i]

        return self.log(epoch, rows)

    def log(self, epoch, rows):

        def save():

            key = 'study_id'
            df = pd.DataFrame(rows).drop_duplicates(subset=key)

            df.to_csv(
                os.path.join(self.save_dir, f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()

    def reset(self):
        super(SectionsLogger, self).reset()
        self.rows = []


class ReportLogger(CXRReportGenerationMetric):
    """
    Logs the findings and impression sections of a report to a .csv.
    """

    def __init__(self, metric_name='reports', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)

        self.findings = []
        self.impression = []
        self.study_ids = []

    def update(self, findings, impression, study_ids):
        """
        Argument/s:
            findings - the findings section.
            impression - the impression section.
            study_ids - list of study identifiers.
        """

        assert isinstance(findings, list), '"findings" must be a list.'
        assert isinstance(impression, list), '"impression" must be a list.'

        self.findings.extend(findings)
        self.impression.extend(impression)
        self.study_ids.extend(study_ids)
    
    def compute(self, epoch):

        rows = []
        for (i_1, i_2, i_3) in zip(self.findings, self.impression, self.study_ids):
            rows.append(
                {
                    'findings': i_1,
                    'impression': i_2,
                    'study_id': i_3,
                }
            )

        if torch.distributed.is_initialized():  # If DDP
            rows_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rows_gathered, rows)
            rows = [j for i in rows_gathered for j in i]

        return self.log(epoch, rows)

    def log(self, epoch, rows):

        def save():

            key = 'study_id'
            df = pd.DataFrame(rows).drop_duplicates(subset=key)

            df.to_csv(
                os.path.join(self.save_dir, f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()

    def reset(self):
        super(ReportLogger, self).reset()
        self.findings = []
        self.impression = []
        self.study_ids = []


class ReportReasoningLogger(CXRReportGenerationMetric):
    """
    Logs the findings and impression sections of a report to a .csv. Also logs reasoning.
    """

    def __init__(self, metric_name='reports', **kwargs):

        super().__init__(metric_name=metric_name, **kwargs)

        self.findings = []
        self.impression = []
        self.reasoning = []
        self.study_ids = []

    def update(self, findings, impression, reasoning, study_ids):
        assert isinstance(findings, list), '"findings" must be a list.'
        assert isinstance(impression, list), '"impression" must be a list.'

        self.findings.extend(findings)
        self.impression.extend(impression)
        self.reasoning.extend(reasoning)
        self.study_ids.extend(study_ids)

    def compute(self, epoch):

        rows = []
        for (i_1, i_2, i_3, i_4) in zip(self.findings, self.impression, self.reasoning, self.study_ids):
            rows.append(
                {
                    'findings': i_1,
                    'impression': i_2,
                    'reasoning': i_3,
                    'study_id': i_4,
                }
            )

        if torch.distributed.is_initialized():  # If DDP
            rows_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rows_gathered, rows)
            rows = [j for i in rows_gathered for j in i]

        return self.log(epoch, rows)

    def log(self, epoch, rows):

        def save():

            key = 'study_id'
            df = pd.DataFrame(rows).drop_duplicates(subset=key)

            df.to_csv(
                os.path.join(self.save_dir, f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()

    def reset(self):
        super(ReportReasoningLogger, self).reset()
        self.findings = []
        self.impression = []
        self.reasoning = []
        self.study_ids = []


class ReportTokenIdentifiersLogger(Metric):
    """
    Logs the findings and impression section token identifiers of a report to a .csv.
    """

    def __init__(self, exp_dir: str, split: str, metric_name='report_ids', **kwargs):
        super().__init__(**kwargs)

        self.exp_dir = exp_dir
        self.split = split
        self.metric_name = metric_name

        self.report_ids = []
        self.study_ids = []

        self.save_dir = os.path.join(self.exp_dir, 'metric_outputs', self.metric_name)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def dist_reduce_fx(state):
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, state)
        gathered = list(itertools.chain.from_iterable(gathered))
        return gathered

    def update(self, report_ids, study_ids):
        """
        Argument/s:
            report_ids - report identifiers.
            study_ids - list of study identifiers.
        """

        assert isinstance(report_ids, torch.Tensor), '"report_ids" must be a torch.Tensor.'
        report_ids = report_ids.tolist()

        assert isinstance(report_ids, list)
        assert all(isinstance(i, list) for i in report_ids)

        self.report_ids.append(report_ids)
        self.study_ids.append(study_ids)

    def compute(self, epoch):
        report_ids = self.report_ids.tolist() if not isinstance(self.report_ids, list) else self.report_ids

        rows = []
        for (i, j) in zip(report_ids, self.study_ids):
            rows.append(
                {
                    'report_ids': i,
                    'study_id': j,
                }
            )

        return self.log(epoch, rows)

    def log(self, epoch, rows):

        def save():

            key = 'study_id'
            df = pd.DataFrame(rows).drop_duplicates(subset=key)

            df.to_csv(
                os.path.join(self.save_dir, f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()

    def reset(self):
        super(ReportTokenIdentifiersLogger, self).reset()
        self.report_ids = []
        self.study_ids = []

class SizeLogger(CXRReportGenerationMetric):

    def __init__(self, metric_name='size', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)

        self.size = []
        self.study_ids = []

    def update(self, sizes, study_ids):
        """
        Argument/s:
            sizes - sizes.
            study_ids - list of study identifiers.
        """
        assert isinstance(sizes, list), '"sizes" must be a list.'

        self.size.extend(sizes)
        self.study_ids.extend(study_ids)

    def metric_scoring(self, batch):

        mbatch_rows = []
        for x, y in zip(self.study_ids, self.size):
            mbatch_rows.append({'study_id': x, 'size': y})

        return mbatch_rows

    def convert_lists_to_rows(self):
        size = self.size.tolist() if not isinstance(self.size, list) else self.size

        rows = []
        for (i, j) in zip(size, self.study_ids):
            rows.append(
                {
                    'size': i,
                    'study_id': j,
                }
            )

        if torch.distributed.is_initialized():  # If DDP
            rows_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rows_gathered, rows)
            rows = [j for i in rows_gathered for j in i]
        return rows

    def accumulate_scores(self, df, epoch):

        # Drop duplicates caused by DDP:
        key = 'study_id'
        df = df.drop_duplicates(subset=[key])

        # Save the scores:
        def save_scores():
            df.to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )
        if not torch.distributed.is_initialized():
            save_scores()
        elif torch.distributed.get_rank() == 0:
            save_scores()

        # Number of examples:
        prefix = f'{self.split}_{self.metric_name}_'
        scores = {f'{prefix}num_study_ids': 0 if df.empty else float(df.study_id.nunique())}

        df = df.drop(['study_id'], axis=1)
        mean_scores = {f'{prefix}{k}': v for k, v in df.mean().to_dict().items()}
        scores = {**mean_scores, **scores}
        scores.pop('study_id', None)

        return scores
    
    def reset(self):
        super(SizeLogger, self).reset()
        self.size = []
        self.study_ids = []
