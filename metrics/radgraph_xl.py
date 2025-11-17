import torch
from radgraph import F1RadGraph
from metrics.base import CXRReportGenerationMetric


class RadGraphXLMetric(CXRReportGenerationMetric):
    def __init__(self, metric_name='radgraph-xl', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        assert self.accelerator is not None, 'An accelertator instance must be provided.'

    def init_metric(self):

        self.f1radgraph = F1RadGraph(reward_level='all', model_type='radgraph-xl', cuda=self.accelerator.device.index)
        if self.accelerator.state.distributed_type.name == 'FSDP':
            raise NotImplementedError('Need to implement FSDP for RadGraphXLMetric.')

    def cleanup_metric(self):
        del self.f1radgraph

    def batch_scoring(self, synthetic, radiologist):

        with torch.no_grad():
            _, scores, _, _ = self.f1radgraph(synthetic, radiologist)

        return [{'rg_xl_rg_e': rg_e, 'rg_xl_rg_er': rg_er, 'rg_xl_rg_bar_er': rg_bar_er} for rg_e, rg_er, rg_bar_er in zip(*scores)]


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

    metric = RadGraphXLMetric(mbatch_size=3, split='tmp', exp_dir='/scratch3/nic261/experiments/test/radgraph_xl', accelerator=accelerator)
    metric.update(refs, hyps, study_ids)
    metric.compute()
    metric.reset()
