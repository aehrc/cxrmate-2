from rouge_score import rouge_scorer
from metrics.base import CXRReportGenerationMetric


class ROUGELMetric(CXRReportGenerationMetric):
    def __init__(self, metric_name='rouge_l', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)

    def init_metric(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def cleanup_metric(self):
        del self.scorer

    def batch_scoring(self, synthetic, radiologist):
        scores = []
        for y_hat, y in zip(synthetic, radiologist):
            scores.append({
                'f1': self.scorer.score(y_hat, y)['rougeL'].fmeasure
            })
        return scores
