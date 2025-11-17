import bleuscore
from metrics.base import CXRReportGenerationMetric


class BLEUMetric(CXRReportGenerationMetric):
    def __init__(self, metric_name='bleu', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)

    def batch_scoring(self, synthetic, radiologist):

        scores = []
        for i, j in zip(synthetic, radiologist):
            scores.append({'bleu_4': bleuscore.compute(predictions=[i], references=[[j]])['bleu']})
        return scores


if __name__ == '__main__':
    import accelerate

    refs = [
        'This is exactly the same text to be evaluated.',
        'No evidence of pneumothorax following chest tube removal.',
        'There is a left pleural effusion.',
        'There is a left pleural effusion.',
        'There is a left pleural effusion.',
    ]
    hyps = [
        'This is exactly the same text to be evaluated.',
        'No pneumothorax detected, there is no evidence.',
        'Left pleural effusion is present.',
        'No pneumothorax detected.',
        'There is a left pleural effusion and right pleural effusion.',
    ]
    refs = refs * 100
    hyps = hyps * 100
    study_ids = list(range(len(refs)))

    accelerator = accelerate.Accelerator()

    metric = BLEUMetric(mbatch_size=5, split='tmp', exp_dir='/scratch3/nic261/experiments/test/bleu', accelerator=accelerator)
    metric.update(refs, hyps, study_ids)
    metric.compute()
    metric.reset()