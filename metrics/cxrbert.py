import accelerate
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from metrics.base import CXRReportGenerationMetric


class CXRBERTMetric(CXRReportGenerationMetric):

    def __init__(self, metric_name='cxrbert', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        assert self.accelerator is not None, 'An accelertator instance must be provided.'

    def init_metric(self):
        ckpt_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(ckpt_name, trust_remote_code=True).to(self.accelerator.device)
        self.model.eval()
        if self.accelerator.state.distributed_type.name == 'FSDP':
            self.model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.model)

    def cleanup_metric(self):
        del self.tokenizer, self.model

    def batch_scoring(self, synthetic, radiologist):

        # Tokenize and compute the sentence embeddings
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=synthetic,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )

        prediction_embeddings = self.model(
            input_ids=tokenizer_output.input_ids.to(self.accelerator.device),
            attention_mask=tokenizer_output.attention_mask.to(self.accelerator.device),
            output_cls_projected_embedding=True,
            return_dict=False,
        )

        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=radiologist,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )

        label_embeddings = self.model(
            input_ids=tokenizer_output.input_ids.to(self.accelerator.device),
            attention_mask=tokenizer_output.attention_mask.to(self.accelerator.device),
            output_cls_projected_embedding=True,
            return_dict=False,
        )

        # Compute the cosine similarity of sentence embeddings obtained from input text prompts.
        sim = torch.nn.functional.cosine_similarity(
            prediction_embeddings[2],
            label_embeddings[2],
        )

        return [{'similarity': i} for i in sim.tolist()]
