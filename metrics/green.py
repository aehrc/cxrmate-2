import re
import warnings


import accelerate
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from appdirs import user_cache_dir
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from loguru import logger
from metrics.base import CXRReportGenerationMetric

logger.disable('PyRuSH')
CACHE_DIR = user_cache_dir('chexbert')


def compute_largest_cluster(sentences):
    """
    Computes the largest cluster of sentences using K-means clustering, finds the sentences within the largest cluster, and orders them by their distance to the cluster center.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - sentences_of_largest_cluster (list): Sentences in the largest cluster, ordered by their proximity
              to the cluster center.
    """
    if len(sentences) == 0:
        return None, None
    embeddings, kmeans = compute_kmeans(sentences)
    cluster_sizes = np.bincount(kmeans.labels_)
    largest_cluster_idx = np.argmax(cluster_sizes)
    cluster_member_ids = np.where(kmeans.labels_ == largest_cluster_idx)[0]
    sentences_of_largest_cluster = [sentences[i] for i in cluster_member_ids]

    largest_cluster_mean = kmeans.cluster_centers_[largest_cluster_idx]
    embeddings_of_largest_cluster = [embeddings[i] for i in cluster_member_ids]
    distances = distance.cdist(
        embeddings_of_largest_cluster, [largest_cluster_mean], "cosine"
    ).flatten()
    closest_point_indices = np.argsort(distances)[0]

    sentences_of_largest_cluster = sentences_of_largest_cluster[closest_point_indices]

    return embeddings, sentences_of_largest_cluster


def compute_kmeans(sentences):
    """
    Computes K-means clustering for a list of sentences by generating their embeddings, normalizing the embeddings, and determining the optimal number of clusters using binary search.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - kmeans (KMeans): The KMeans object with the optimal number of clusters determined.
    """
    # sentence embeddings
    embeddings = sentence_transformer.encode(sentences)
    # normalize the embeddings for equivalent computation of the cosine distance
    embeddings = preprocessing.normalize(embeddings)
    # compute the number of clusters with binary search
    kmeans = binary_search_optimal_kmeans(embeddings, min_k=0, max_k=len(sentences))
    return embeddings, kmeans


"""
Below is updated to handle when the length of data is one:
"""
def binary_search_optimal_kmeans(data, min_k, max_k):
    """
    Finds the optimal k for KMeans clustering using binary search on the silhouette score.

    Args:
        data (ndarray): Data to cluster (e.g., sentence embeddings).
        min_k (int): Minimum number of clusters for binary search.
        max_k (int): Maximum number of clusters for binary search.

    Returns:
        KMeans: The KMeans object with the optimal number of clusters determined.
    """
    if len(data) < 2:
        # If less than 2 data points, return a trivial KMeans model with 1 cluster:
        return KMeans(n_clusters=1, random_state=42).fit(data)

    best_score = -1
    best_kmeans = None

    # Binary search for the optimal k:
    while min_k <= max_k:
        mid_k = (min_k + max_k) // 2
        if mid_k < 2:  # Silhouette score requires at least 2 clusters.
            break

        # Fit KMeans with mid_k clusters:
        kmeans = KMeans(n_clusters=mid_k, random_state=42).fit(data)
        labels = kmeans.labels_

        # Compute silhouette score:
        score = silhouette_score(data, labels)

        # Update best model if score improves:
        if score > best_score:
            best_score = score
            best_kmeans = kmeans

        # Adjust binary search bounds:
        if score > best_score:
            min_k = mid_k + 1
        else:
            max_k = mid_k - 1

    # If no valid KMeans was created, fallback to 1 cluster:
    if best_kmeans is None:
        return KMeans(n_clusters=1, random_state=42).fit(data)

    return best_kmeans


def flatten_values_lists_of_list_dicts_to_dict(item):
    """
    Flattens a list of dictionaries containing lists of values into a single dictionary.

    Args:
        item (list): List of dictionaries, where each dictionary's values are lists. If any element of the list is itself a list, the function will consider only the first dictionary in that sublist.

    Returns:
        dict: A dictionary where each key corresponds to the keys in the input dictionaries, and each value is a flattened list of all values associated with that key across all input dictionaries.
    """
    result = {}
    for i in item:
        if isinstance(i, list):
            i = i[0]
        for key, lists in i.items():
            if key not in result:
                result[key] = []
            result[key].extend(lists)

    return result


def clean_responses(response):
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        if (
            "[Explanation]:\n    <Explanation>\n" or "[Explanation]:\n<Explanation>"
        ) in response:
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")


def make_prompt(text1, text2, max_len=300):
    """
    Creates a prompt for evaluating the accuracy of a candidate radiology report in comparison to a reference radiology report.

    Args:
        text1 (str): Reference radiology report.
        text2 (str): Candidate radiology report.

    Returns:
        str: Formatted prompt string.
    """
    text1 = " ".join(text1.split()[:max_len])
    text2 = " ".join(text2.split()[:max_len])
    prompt = f"Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n    Process Overview: You will be presented with:\n\n    1. The criteria for making a judgment.\n    2. The reference radiology report.\n    3. The candidate radiology report.\n    4. The desired format for your assessment.\n\n    1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n    The count of clinically significant errors.\n    The count of clinically insignificant errors.\n\n    Errors can fall into one of these categories:\n\n    a) False report of a finding in the candidate.\n    b) Missing a finding present in the reference.\n    c) Misidentification of a finding's anatomic location/position.\n    d) Misassessment of the severity of a finding.\n    e) Mentioning a comparison that isn't in the reference.\n    f) Omitting a comparison detailing a change from a prior study.\n    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n    2. Reference Report:\n    {text1}\n\n    3. Candidate Report:\n    {text2}\n\n    4. Reporting Your Assessment:\n\n    Follow this specific format for your output, even if no errors are found:\n    ```\n    [Explanation]:\n    <Explanation>\n\n    [Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n"
    return prompt


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class GREEN:
    def __init__(self, model_name, device):
        super().__init__()
        warnings.filterwarnings(
            "ignore", message="A decoder-only architecture is being used*"
        )
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
            message="Number of distinct clusters.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="transformers.tokenization_utils_base",
        )
        self.model_name = model_name.split("/")[-1]
        self.batch_size = 4
        self.max_length = 2048
        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]
        self.prompts = None
        self.completions = None
        self.green_scores = None
        self.error_counts = None
        
        self.is_fsdp = False

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False if "Phi" in model_name else True,
            device_map=device,
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=True,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )

        chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

        self.tokenizer.chat_template = chat_template
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.clean_up_tokenization_spaces = True
        self.tokenizer.padding_side = "left"

    def __call__(self, refs, hyps):
        dataset = Dataset.from_dict({"reference": refs, "prediction": hyps})
        dataset = self.process_data(dataset)
        self.dataset = dataset
        mean, std, green_scores, summary, results_df = self.infer()
        return mean, std, green_scores, summary, results_df

    def process_data(self, dataset):
        def prompting(examples):
            return {
                "prompt": [
                    make_prompt(r, p)
                    for r, p in zip(examples["reference"], examples["prediction"])
                ]
            }

        dataset = dataset.map(prompting, batched=True)
        return dataset

    @torch.inference_mode()
    def infer(self):
        dataset_dist = self.dataset

        local_completions = []
        local_references = []

        for batch in dataset_dist.iter(batch_size=self.batch_size):
            local_references.extend(batch["prompt"])
            local_completions.extend(self.get_response(batch))

        self.completions = local_completions
        self.prompts = local_references

        if len(self.completions) != len(self.prompts):
            print("Length of prompts and completions are not equal!")

        return self.process_results()

    def tokenize_batch_as_chat(self, batch):
        batch = [
            self.tokenizer.apply_chat_template(
                i, tokenize=False, add_generation_prompt=True
            )
            for i in batch
        ]

        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return batch

    def get_response(self, batch):
        assert "prompt" in batch.keys(), "prompt is not in batch keys"

        batch = [
            [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
            for prompt in batch["prompt"]
        ]

        batch = self.tokenize_batch_as_chat(batch)

        gen_model = self.model.module if hasattr(self.model, 'module') else self.model
        outputs = gen_model.generate(
            input_ids=batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=2048 + 512,  # Add 512 to allow for prompts of length 2048.
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response_list = []
        if isinstance(responses, list):
            for response in responses:
                response = clean_responses(response)
                response_list.append(response)
        else:
            responses = clean_responses(responses)
            response_list.append(responses)

        return response_list

    def process_results(self):
        self.green_scores = [
            self.compute_green(response) for response in self.completions
        ]
        self.error_counts = pd.DataFrame(
            [self.compute_error_count(response) for response in self.completions],
            columns=self.sub_categories + ["Matched Findings"],
        )

        results_df = pd.DataFrame(
            {
                "reference": self.dataset["reference"],
                "predictions": self.dataset["prediction"],
                "green_analysis": self.completions,
                "green_score": self.green_scores,
                **self.error_counts,
            }
        )

        mean, std, summary = self.compute_summary()

        return mean, std, self.green_scores, summary, results_df

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            return 0

        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        if not category_text:
            if for_reward:
                return None, None
            return sum_counts, sub_counts
        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts
        else:
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def parse_error_sentences(self, response, category):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, response, re.DOTALL)
        sub_category_dict_sentences = {}
        for sub_category in self.sub_categories:
            sub_category_dict_sentences[sub_category] = []

        if not category_text:
            return sub_category_dict_sentences
        if category_text.group(1).startswith("No"):
            return sub_category_dict_sentences

        if category == "Matched Findings":
            return (
                category_text.group(1).rsplit(":", 1)[-1].rsplit(".", 1)[-1].split(";")
            )

        matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

        if len(matches) == 0:
            matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
            self.sub_categories = [
                f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
            ]

        for position, sub_category in enumerate(self.sub_categories):
            for match in range(len(matches)):
                if matches[match].startswith(sub_category):
                    sentences_list = (
                        matches[match].rsplit(":", 1)[-1].split(".", 1)[-1].split(";")
                    )
                    sub_category_dict_sentences[self.sub_categories[position]] = (
                        sentences_list
                    )

        return sub_category_dict_sentences

    def compute_sentences(self, response):
        return self.parse_error_sentences(response, self.categories[0])

    def get_representative_sentences(self, responses):
        list_sentences = []
        for i in responses:
            sentences = self.compute_sentences(i)
            list_sentences.append(sentences)

        dict_sentences = flatten_values_lists_of_list_dicts_to_dict(list_sentences)

        result_sentences_dict = {}

        for i in self.sub_categories:
            sentences = dict_sentences[i]
            sentences = [i for i in sentences if i.strip() != ""]
            _, sentences_of_largest_cluster = compute_largest_cluster(sentences)
            result_sentences_dict[i] = sentences_of_largest_cluster

        return result_sentences_dict

    def compute_accuracy(self, responses):
        counts = []
        for response in responses:
            _, sig_errors = self.parse_error_counts(response, self.categories[0])
            counts.append(sig_errors)

        counts = np.array(counts)

        dict_acc = {}
        for i in range(len(self.sub_categories)):
            error_counts = counts[:, i]
            accuracy = np.mean(error_counts == 0)
            dict_acc[self.sub_categories[i]] = accuracy

        return dict_acc

    def compute_summary(self):
        representative_sentences = self.get_representative_sentences(self.completions)
        accuracies = self.compute_accuracy(self.completions)
        mean = np.mean(self.green_scores)
        std = np.std(self.green_scores)

        summary = f"\n-------------{self.model_name}----------------\n [Summary]: Green average {mean} and standard deviation {std} \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n"
        for idx, sub_category in enumerate(self.sub_categories):
            accuracy = accuracies[sub_category]
            sentences = representative_sentences[sub_category]
            summary += f"{sub_category}: {accuracy}. \n {sentences} \n\n"
        summary += "----------------------------------\n"

        return mean, std, summary
    

class GREENMetric(CXRReportGenerationMetric):
    """
    GREEN for CXR report generation evaluation.
    """

    def __init__(self, metric_name='green', **kwargs):
        super().__init__(metric_name=metric_name, **kwargs)
        disable_progress_bar()
        assert self.accelerator is not None, 'An accelertator instance must be provided.'

    def init_metric(self):
        self.metric = GREEN('StanfordAIMI/GREEN-radllama2-7b', device=self.accelerator.device)
        global sentence_transformer
        sentence_transformer = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2", device=self.accelerator.device)

        if self.accelerator.state.distributed_type.name == 'FSDP':
            self.metric.model = accelerate.utils.fsdp2_prepare_model(self.accelerator, self.metric.model)
            sentence_transformer[0].auto_model = accelerate.utils.fsdp2_prepare_model(self.accelerator, sentence_transformer[0].auto_model)

    def cleanup_metric(self):
        del self.metric
        del globals()['sentence_transformer']

    def batch_scoring(self, synthetic, radiologist):

        _, _, scores, _, _ = self.metric(radiologist, synthetic)

        return [{'green': i} for i in scores]
