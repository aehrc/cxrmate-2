import os
import time
import accelerate
import numpy as np
import pandas as pd
import torch
from metrics.base import CXRReportGenerationMetric
import transformers


class SRRMetric(CXRReportGenerationMetric):

    # https://huggingface.co/StanfordAIMI/SRR-BERT-Leaves

    idx_2_label = {
        0: 'No Finding',
        1: 'Lung Lesion',
        2: 'Edema',
        3: 'Pneumonia',
        4: 'Atelectasis',
        5: 'Aspiration',
        6: 'Lung collapse',
        7: 'Perihilar airspace opacity',
        8: 'Air space opacity–multifocal',
        9: 'Mass/Solitary lung mass',
        10: 'Nodule/Solitary lung nodule',
        11: 'Cavitating mass with content',
        12: 'Cavitating masses',
        13: 'Emphysema',
        14: 'Fibrosis',
        15: 'Pulmonary congestion',
        16: 'Hilar lymphadenopathy',
        17: 'Bronchiectasis',
        18: 'Simple pneumothorax',
        19: 'Loculated pneumothorax',
        20: 'Tension pneumothorax',
        21: 'Simple pleural effusion',
        22: 'Loculated pleural effusion',
        23: 'Pleural scarring',
        24: 'Hydropneumothorax',
        25: 'Pleural Other',
        26: 'Cardiomegaly',
        27: 'Pericardial effusion',
        28: 'Inferior mediastinal mass',
        29: 'Superior mediastinal mass',
        30: 'Tortuous Aorta',
        31: 'Calcification of the Aorta',
        32: 'Enlarged pulmonary artery',
        33: 'Hernia',
        34: 'Pneumomediastinum',
        35: 'Tracheal deviation',
        36: 'Acute humerus fracture',
        37: 'Acute rib fracture',
        38: 'Acute clavicle fracture',
        39: 'Acute scapula fracture',
        40: 'Compression fracture',
        41: 'Shoulder dislocation',
        42: 'Subcutaneous Emphysema',
        43: 'Suboptimal central line',
        44: 'Suboptimal endotracheal tube',
        45: 'Suboptimal nasogastric tube',
        46: 'Suboptimal pulmonary arterial catheter',
        47: 'Pleural tube',
        48: 'PICC line',
        49: 'Port catheter',
        50: 'Pacemaker',
        51: 'Implantable defibrillator',
        52: 'LVAD',
        53: 'Intraaortic balloon pump',
        54: 'Pneumoperitoneum',
    }
    
    def __init__(self, metric_name='srr', **kwargs):

        super().__init__(metric_name=metric_name, **kwargs)
        assert self.accelerator is not None, 'An accelertator instance must be provided.'
        assert self.mbatch_size == 1, 'SRRMetric only supports mbatch_size=1.'

    def init_metric(self):

        self.tokenizer = transformers.BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-general')
        self.model = transformers.BertForSequenceClassification.from_pretrained('StanfordAIMI/SRR-BERT-Leaves', num_labels=len(self.idx_2_label)).to(device='cuda')


        if self.accelerator.state.distributed_type.name == 'FSDP':
            self.model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.model)

        self.model.eval()

    def cleanup_metric(self):
        del self.model

    def batch_scoring(self, synthetic, radiologist):

        assert len(synthetic) == 1
        assert len(radiologist) == 1

        y_hat_inputs = self.tokenizer(
            synthetic[0],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device=self.model.device)
        y_inputs = self.tokenizer(
            radiologist[0],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device=self.model.device)

        with torch.no_grad():

            logits = self.model(**y_hat_inputs).logits
            y_hat_preds = (torch.sigmoid(logits)[0].cpu().numpy() > 0.5).astype(int)

            logits = self.model(**y_inputs).logits
            y_preds = (torch.sigmoid(logits)[0].cpu().numpy() > 0.5).astype(int)

        mbatch_row = [
            {
                **{f'Generated {k}': v for k, v in zip(self.idx_2_label.values(), y_hat_preds)},
                **{f'GT {k}': v for k, v in zip(self.idx_2_label.values(), y_preds)},
            }
        ]

        return mbatch_row

    def accumulate_scores(self, df, epoch):

        examples = df.to_dict(orient='records')

        y_hat_rows = [{k.replace('Generated ', ''): v for k, v in i.items() if 'GT ' not in k} for i in examples]
        y_rows = [{k.replace('GT ', ''): v for k, v in i.items() if 'Generated ' not in k} for i in examples]

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
        scores['y_hat'][list(self.idx_2_label.values())] = (scores['y_hat'][list(self.idx_2_label.values())] == 1)
        scores['y'][list(self.idx_2_label.values())] = (scores['y'][list(self.idx_2_label.values())] == 1)

        # Create dataframes for each error type:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores[i] = scores['y'][['study_id']].copy()

        # Calculate errors:
        scores['tp'][list(self.idx_2_label.values())] = \
            (scores['y_hat'][list(self.idx_2_label.values())]).astype(float) * (scores['y'][list(self.idx_2_label.values())]).astype(float)
        scores['tn'][list(self.idx_2_label.values())] = \
            (~scores['y_hat'][list(self.idx_2_label.values())]).astype(float) * (~scores['y'][list(self.idx_2_label.values())]).astype(float)
        scores['fp'][list(self.idx_2_label.values())] = \
            (scores['y_hat'][list(self.idx_2_label.values())]).astype(float) * (~scores['y'][list(self.idx_2_label.values())]).astype(float)
        scores['fn'][list(self.idx_2_label.values())] = \
            (~scores['y_hat'][list(self.idx_2_label.values())]).astype(float) * (scores['y'][list(self.idx_2_label.values())]).astype(float)

        # Initialise example scores dataframe:
        scores['example'] = scores['tp'][['study_id']].copy()

        # Errors per study_id:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores['example'][f'{i}'] = scores[i][list(self.idx_2_label.values())].sum(1)

        # Initialise class scores dataframe:
        scores['class'] = pd.DataFrame()

        # Sum over study_ids for class scores:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores['class'][i] = scores[i][list(self.idx_2_label.values())].sum()

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

