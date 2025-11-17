import accelerate
from RaTEScore import RaTEScore
from loguru import logger
from metrics.base import CXRReportGenerationMetric
import torch 

logger.disable('PyRuSH')


class RaTEScoreMetric(CXRReportGenerationMetric):
    def __init__(self, metric_name='ratescore', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        assert self.accelerator is not None, 'An accelertator instance must be provided.'

    def init_metric(self):
        self.ratescore = RaTEScore()
        self.ratescore.model = self.ratescore.model.to(self.accelerator.device)
        self.ratescore.eval_model = self.ratescore.eval_model.to(self.accelerator.device)
        if self.accelerator.state.distributed_type.name == 'FSDP':
            self.ratescore.model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.ratescore.model)
            self.ratescore.eval_model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.ratescore.eval_model)

    def cleanup_metric(self):
        del self.ratescore

    def batch_scoring(self, synthetic, radiologist):
        with torch.no_grad():
            scores = self.ratescore.compute_score(synthetic, radiologist)
        return [{'ratescore': score} for score in scores]
