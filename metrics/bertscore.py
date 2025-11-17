import accelerate
import torch
from bert_score import BERTScorer
from metrics.base import CXRReportGenerationMetric


class BERTScoreRoBERTaLargeMetric(CXRReportGenerationMetric):
    def __init__(self, num_workers, metric_name='bertscore', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        self.num_workers = num_workers
        assert self.accelerator is not None, 'An accelertator instance must be provided.'

    def init_metric(self):
        self.bert_scorer = BERTScorer(
            model_type='roberta-large',
            num_layers=17,
            batch_size=self.mbatch_size,
            nthreads=self.num_workers,
            all_layers=False,
            idf=False,
            lang='en',
            device=self.accelerator.device,
            rescale_with_baseline=True,
        )
        if self.accelerator.state.distributed_type.name == 'FSDP':
            self.bert_scorer._model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.bert_scorer._model)

    def cleanup_metric(self):
        del self.bert_scorer

    def batch_scoring(self, synthetic, radiologist):

        with torch.no_grad():
            bert_scores, _ = self.bert_scorer.score(synthetic, radiologist, batch_size=self.mbatch_size, return_hash=True)

        precision = bert_scores[0].tolist()
        recall = bert_scores[1].tolist()
        f1 = bert_scores[2].tolist()

        return [{'f1': i, 'precision': j, 'recall': k} for i, j, k in zip(f1, precision, recall)]

