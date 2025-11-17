import transformers
from metrics.base import CXRReportGenerationMetric
import torch


class AbsenceOfRepeatedNGramesMetric(CXRReportGenerationMetric):

    def __init__(self, metric_name='arn', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        self.n = 3
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', trust_remote_code=True)
        assert self.accelerator is not None, 'An accelertator instance must be provided.'

    def batch_scoring(self, synthetic, radiologist):
        
        arng = []  # Absence of repeated n-grams scores.
        for i in synthetic:
            
            tokens = self.tokenizer.tokenize(i)
            
            # If the sequence is shorter than the n-gram size, maximum reward is given:
            if len(tokens) < self.n:
                arng.append(1.0)
                continue
            
            # Count the occurrences of n-grams:
            ngram_counts = {}
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                if ngram in ngram_counts:
                    ngram_counts[ngram] += 1
                else:
                    ngram_counts[ngram] = 1
            
            # Total number of n-grams:
            total_ngrams = len(tokens) - self.n + 1

            # Calculate the number of repeated n-grams:
            repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)

            # Calculate the reward as the absence of repeated n-grams:
            arng.append(1.0 - (repeated_ngrams / total_ngrams)) # Invert the penalty to get reward:
        arng = torch.tensor(arng, device=self.accelerator.device)    
                
        score = arng.tolist()

        return [{'score': i} for i in score]
