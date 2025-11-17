import os
from metrics.base import CXRReportGenerationMetric
from RadEval.factual.RadCliQv1.radcliq import CompositeMetric


class RadCliQMetric(CXRReportGenerationMetric):

    def __init__(self, metric_name='radcliq', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        assert self.accelerator is not None, 'An accelertator instance must be provided.'
        raise NotImplementedError('RadCliQ metric does not work with multi-device setups currently.')

    def init_metric(self):
        self.radcliq = CompositeMetric()

    def cleanup_metric(self):
        del self.radcliq

    def batch_scoring(self, synthetic, radiologist):

        _, detail_scores = self.radcliq.predict(radiologist, synthetic)

        return [{'similarity': score} for score in detail_scores]
    

if __name__ == '__main__':

    import accelerate

    refs = [
        'No evidence of pneumothorax following chest tube removal.',
        'There is a left pleural effusion.',
        'There is a left pleural effusion.'
    ]
    hyps = [
        'No pneumothorax detected.',
        'Left pleural effusion is present.',
        'No pneumothorax detected.',
    ]
    refs = refs * 100
    hyps = hyps * 100
    study_ids = list(range(len(refs)))

    accelerator = accelerate.Accelerator()

    metric = RadCliQMetric(split='tmp', exp_dir='/scratch3/nic261/experiments/test/radcliq', accelerator=accelerator)
    metric.update(refs, hyps, study_ids)
    metric.compute()
    metric.reset()
