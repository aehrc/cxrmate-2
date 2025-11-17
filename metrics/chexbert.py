import os
import time

from collections import OrderedDict
from appdirs import user_cache_dir

import accelerate
import examples
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
)
from metrics.base import CXRReportGenerationMetric

CACHE_DIR = user_cache_dir('chexbert')


"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""


CLASSES = {
    0: 'not mentioned',
    1: 'positive',
    2: 'negative',
    3: 'uncertain',
}


PATHOLOGIES = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]


def download_model(repo_id, cache_dir, filename=None):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    if filename is not None:
        files = [filename]
    else:
        files = list(set(list_repo_files(repo_id=repo_id)).difference({'README.md', '.gitattributes'}))
    for f in files:
        try:
            hf_hub_download(repo_id=repo_id, filename=f, cache_dir=cache_dir, force_filename=f)
        except Exception as e:
            print(e)


class CheXbert(nn.Module):
    def __init__(self, device, p=0.1):
        super(CheXbert, self).__init__()

        self.device = device
        
        # Downloading pretrain model from huggingface
        ckpt_path = os.path.join(CACHE_DIR, "chexbert.pth")
        _ = hf_hub_download(repo_id='StanfordAIMI/RRG_scorers', local_dir=CACHE_DIR, filename='chexbert.pth')

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig().from_pretrained('bert-base-uncased')

        with torch.no_grad():

            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)

            hidden_size = self.bert.pooler.dense.in_features

            # Classes: present, absent, unknown, blank for 12 conditions + support devices:
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])

            # Classes: yes, no for the 'no finding' observation:
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            # Load CheXbert checkpoint:
            assert os.path.exists(ckpt_path)
            state_dict = torch.load(ckpt_path, map_location=device)['model_state_dict']

            new_state_dict = OrderedDict()
            # new_state_dict['bert.embeddings.position_ids'] = torch.arange(config.max_position_embeddings).expand((1, -1))
            for key, value in state_dict.items():
                if 'bert' in key:
                    new_key = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key:
                    new_key = key.replace('module.linear_heads.', 'linear_heads.')
                new_state_dict[new_key] = value

            self.load_state_dict(new_state_dict)

        self.eval()

    def forward(self, reports):

        for i in range(len(reports)):
            reports[i] = reports[i].strip()
            reports[i] = reports[i].replace(r"\n", " ")
            reports[i] = reports[i].replace(r"\s+", " ")
            reports[i] = reports[i].replace(r"\s+(?=[\.,])", "")
            reports[i] = reports[i].strip()

        with torch.no_grad():

            tokenized = self.tokenizer(
                reports,
                padding='longest',
                return_tensors='pt',
                truncation=True,
                max_length=self.bert.config.max_position_embeddings,
            )

            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            last_hidden_state = self.bert(**tokenized)[0]

            cls = last_hidden_state[:, 0, :]
            cls = self.dropout(cls)

            predictions = []
            for i in range(14):
                predictions.append(self.linear_heads[i](cls).argmax(dim=1))

        return torch.stack(predictions, dim=1)


class CheXbertMetric(CXRReportGenerationMetric):
    def __init__(self, metric_name='chexbert', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        assert self.accelerator is not None, 'An accelertator instance must be provided.'

    def init_metric(self):
        self.chexbert = CheXbert(device=self.accelerator.device).to(self.accelerator.device)
        if self.accelerator.state.distributed_type.name == 'FSDP':
            self.chexbert = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.chexbert)

    def cleanup_metric(self):
        del self.chexbert

    def batch_scoring(self, synthetic, radiologist):

        with torch.no_grad():
            y_hat_chexbert = self.chexbert(list(synthetic)).tolist()
            y_chexbert = self.chexbert(list(radiologist)).tolist()

        rows = []
        for i, j in zip(y_hat_chexbert, y_chexbert):
            rows.append(
                {
                    **{f'y_hat_{k}': v for k, v in zip(PATHOLOGIES, i)},
                    **{f'y_label_{k}': v for k, v in zip(PATHOLOGIES, j)},
                }
            )

        return rows

    def accumulate_scores(self, df, epoch):

        examples = df.to_dict(orient='records')

        y_hat_rows = [{k.replace('y_hat_', ''): v for k, v in i.items() if 'y_label_' not in k} for i in examples]
        y_rows = [{k.replace('y_label_', ''): v for k, v in i.items() if 'y_hat_' not in k} for i in examples]

        scores = {'y_hat': pd.DataFrame(y_hat_rows), 'y': pd.DataFrame(y_rows)}

        # Drop duplicates caused by DDP:
        key = 'study_id'
        scores['y_hat'] = scores['y_hat'].drop_duplicates(subset=[key])
        scores['y'] = scores['y'].drop_duplicates(subset=[key])

        def save_chexbert_outputs():
            scores['y_hat'].to_csv(
                os.path.join(
                    self.save_dir, f'{self.split}_epoch-{epoch}_y_hat_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'
                ),
                index=False,
            )
            scores['y'].to_csv(
                os.path.join(
                    self.save_dir, f'{self.split}_epoch-{epoch}_y_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'
                ),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save_chexbert_outputs()
        elif torch.distributed.get_rank() == 0:
            save_chexbert_outputs()

        # Positive is 1/positive, negative is 0/not mentioned, 2/negative, and 3/uncertain:
        scores['y_hat'][PATHOLOGIES] = (scores['y_hat'][PATHOLOGIES] == 1)
        scores['y'][PATHOLOGIES] = (scores['y'][PATHOLOGIES] == 1)

        # Create dataframes for each error type:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores[i] = scores['y'][['study_id']].copy()

        # Calculate errors:
        scores['tp'][PATHOLOGIES] = \
            (scores['y_hat'][PATHOLOGIES]).astype(float) * (scores['y'][PATHOLOGIES]).astype(float)
        scores['tn'][PATHOLOGIES] = \
            (~scores['y_hat'][PATHOLOGIES]).astype(float) * (~scores['y'][PATHOLOGIES]).astype(float)
        scores['fp'][PATHOLOGIES] = \
            (scores['y_hat'][PATHOLOGIES]).astype(float) * (~scores['y'][PATHOLOGIES]).astype(float)
        scores['fn'][PATHOLOGIES] = \
            (~scores['y_hat'][PATHOLOGIES]).astype(float) * (scores['y'][PATHOLOGIES]).astype(float)

        # Initialise example scores dataframe:
        scores['example'] = scores['tp'][['study_id']].copy()

        # Errors per study_id:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores['example'][f'{i}'] = scores[i][PATHOLOGIES].sum(1)

        # Initialise class scores dataframe:
        scores['class'] = pd.DataFrame()

        # Sum over study_ids for class scores:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores['class'][i] = scores[i][PATHOLOGIES].sum()

        # Accuracy:
        scores['class']['accuracy'] = np.where(
            (scores['class']['tp'] + scores['class']['tn'] + scores['class']['fp'] + scores['class']['fn']) == 0,
            np.nan,  # Undefined when there are no true/false positives or negatives.
            (scores['class']['tp'] + scores['class']['tn']) / 
            (scores['class']['tp'] + scores['class']['tn'] + scores['class']['fp'] + scores['class']['fn'])
        )

        # Precision:
        scores['class']['precision'] = np.where(
            (scores['class']['tp'] + scores['class']['fp']) == 0,
            np.nan,  # Undefined when there are no true or false positives.
            scores['class']['tp'] / (scores['class']['tp'] + scores['class']['fp'])
        )

        # Recall:
        scores['class']['recall'] = np.where(
            (scores['class']['tp'] + scores['class']['fn']) == 0,
            np.nan,  # Undefined when there are no true positives or false negatives.
            scores['class']['tp'] / (scores['class']['tp'] + scores['class']['fn'])
        )

        # F1 Score:
        scores['class']['f1'] = np.where(
            (scores['class']['tp'] + 0.5 * (scores['class']['fp'] + scores['class']['fn'])) == 0,
            np.nan,  # Undefined when the denominator for F1 is zero.
            scores['class']['tp'] / (scores['class']['tp'] + 0.5 * (scores['class']['fp'] + scores['class']['fn']))
        )

        # Alternate F1 Score:
        scores['class']['f1_alternate'] = np.where(
            (scores['class']['precision'] + scores['class']['recall']) == 0,
            np.nan,  # Undefined when precision + recall is zero.
            (2 * scores['class']['precision'] * scores['class']['recall']) / (scores['class']['precision'] + scores['class']['recall'])
        )

        # Macro-averaging:
        scores['averaged'] = pd.DataFrame()
        for i in ['accuracy', 'precision', 'recall', 'f1', 'f1_alternate']:
            scores['averaged'][f'{i}_macro'] = [scores['class'][i].mean()]

        # Micro-averaged over the classes:
        scores['averaged']['accuracy_micro'] = (scores['class']['tp'].sum() + scores['class']['tn'].sum()) / (
            scores['class']['tp'].sum() + scores['class']['tn'].sum() +
            scores['class']['fp'].sum() + scores['class']['fn'].sum()
        )
        scores['averaged']['precision_micro'] = scores['class']['tp'].sum() / (
            scores['class']['tp'].sum() + scores['class']['fp'].sum()
        )
        scores['averaged']['recall_micro'] = scores['class']['tp'].sum() / (
            scores['class']['tp'].sum() + scores['class']['fn'].sum()
        )
        scores['averaged']['f1_micro'] = scores['class']['tp'].sum() / (
            scores['class']['tp'].sum() + 0.5 * (scores['class']['fp'].sum() + scores['class']['fn'].sum())
        )

        # Reformat classification scores for individual pathologies:
        scores['class'].insert(loc=0, column='pathology', value=scores['class'].index)
        scores['class'] = scores['class'].drop(['tp', 'tn', 'fp', 'fn'], axis=1).melt(
            id_vars=['pathology'],
            var_name='metric',
            value_name='score',
        )
        scores['class']['metric'] = scores['class']['metric'] + '_' + scores['class']['pathology']
        scores['class'] = pd.DataFrame([scores['class']['score'].tolist()], columns=scores['class']['metric'].tolist())

        # Save the example and class scores:
        def save_scores():
            scores['class'].to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_class_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )
            scores['example'].to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_example_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save_scores()
        elif torch.distributed.get_rank() == 0:
            save_scores()

        score_dict = {
            **scores['averaged'].to_dict(orient='records')[0],
            **scores['class'].to_dict(orient='records')[0],
            'num_study_ids': float(scores['y'].study_id.nunique()),
        }
            
        prefix = f'{self.split}_{self.metric_name}_'
        score_dict = {f'{prefix}{k}': v for k, v in score_dict.items()}

        return score_dict

