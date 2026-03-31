import io
import math
import random
from io import BytesIO
from typing import Dict, List, Union

import cv2
import numpy as np
import pydicom
import requests
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

try:
    from .dataset import CXRMate2Dataset
except ImportError:
    from dataset import CXRMate2Dataset

# Ordered by oblique, lateral, AP, and then PA views so that PA views are closest in position to the generated tokens (and oblique is furtherest).
VIEW_ORDER = [
    None,        
    'nan',  # PadChest.
    'SWIMMERS',
    'LPO', 
    'RAO', 
    'LAO',
    'OBLICUA', # PadChest.
    'AP LLD',
    'AP RLD',
    'PA LLD',
    'PA RLD',
    'LLD',  # PadChest.
    'XTABLE LATERAL',
    'RL',
    'LL',   
    'Lateral',
    'LATERAL',
    'AP AXIAL',
    'ANTEROPOSTERIOR',  # PadChest.
    'AP',
    'GENERICA',  # PadChest (PA).
    'POSTEROANTERIOR',  # PadChest.
    'PA',
]


def compute_time_delta(event_time, reference_time, to_tensor=True):
    time_delta = reference_time - event_time
    time_delta = time_delta.total_seconds()
    assert isinstance(time_delta, float), f'time_delta should be float, not {type(time_delta)}.'
    if time_delta < 0:
        raise ValueError(f'time_delta should be greater than or equal to zero, not {time_delta}.')
    if to_tensor: 
        time_delta = torch.tensor(time_delta)
    return time_delta


class CXRMate2Processor(transformers.ProcessorMixin):

    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'AutoImageProcessor'
    tokenizer_class = 'AutoTokenizer'
    valid_kwargs = [
        'token_type_to_token_type_id',
        'max_generated_tokens',
    ]
    
    def __init__(
        self,
        image_processor,
        tokenizer,
        token_type_to_token: Dict[str, int],
        max_generated_tokens: int,
        embeddings_per_image: int,
        image_token: str,
        max_train_images_per_study: int,  # This includes current and prior images.
        generate_findings_token: str,
        generate_impression_token: str,
        convert_to_rgb: bool = False,
        mimic_cxr_normalisation: bool = True,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer)
        
        self.token_type_to_token = token_type_to_token
        self.max_generated_tokens = max_generated_tokens
        self.embeddings_per_image = embeddings_per_image
        self.image_token = image_token
        self.max_train_images_per_study = max_train_images_per_study
        
        self.generate_findings_token = generate_findings_token
        self.generate_impression_token = generate_impression_token

        self.convert_to_rgb = convert_to_rgb
        self.mimic_cxr_normalisation = mimic_cxr_normalisation
        
        self.generate_findings_token_id = self.tokenizer.convert_tokens_to_ids(self.generate_findings_token)
        self.generate_impression_token_id = self.tokenizer.convert_tokens_to_ids(self.generate_impression_token)
        
        self.time_delta_map = lambda x: 1 / math.sqrt((x / 3600) + 1)
        self.time_delta_monotonic_inversion = True
        self.zero_time_delta_value = self.time_delta_map(0.0)
        self.inf_time_delta_value = self.time_delta_map(float('inf'))

        self.prior_section_token_type_ids = [self.tokenizer.convert_tokens_to_ids(self.token_type_to_token[i]) for i in ['prior_findings', 'prior_impression']]
        self.section_token_type_ids = [self.tokenizer.convert_tokens_to_ids(self.token_type_to_token[i]) for i in ['indication', 'history', 'comparison', 'technique']]
    
        assert self.tokenizer.bos_token_id is not None, 'Tokenizer must have a bos_token_id.'
        assert self.tokenizer.sep_token_id is not None, 'Tokenizer must have a sep_token_id.'
        assert self.tokenizer.eos_token_id is not None, 'Tokenizer must have a eos_token_id.'
        assert self.tokenizer.pad_token_id is not None, 'Tokenizer must have a pad_token_id.'

    def __call__(
        self,
        images: Union[ImageInput, str, list[str], bytes, list[bytes]],
        image_datetime: Union[List[float], None] = None,
        findings: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        impression: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        views: Union[List[str]] = None,
        indication: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        history: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        comparison: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        technique: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        
        study_datetime: Union[float, None] = None,

        # Priors:
        prior_findings: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        prior_impression: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        prior_study_datetime: Union[List[float], None] = None,

        train: bool = False,
        **kwargs,
    ) -> BatchFeature:
        
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim == 4:
                images = images.unsqueeze(0)
        elif isinstance(images, list):
            if isinstance(images[0], (str, bytes)):
                images = [images]
        elif isinstance(images, (str, bytes)):
            images = [[images]]

        if image_datetime is not None and not all(isinstance(x, list) for x in image_datetime):
            image_datetime = [image_datetime]
        if views is not None and not all(isinstance(x, list) for x in views):
            views = [views]

        if indication is not None and not isinstance(indication, list):
            indication = [indication]
        if history is not None and not isinstance(history, list):
            history = [history]
        if comparison is not None and not isinstance(comparison, list):
            comparison = [comparison]
        if technique is not None and not isinstance(technique, list):
            technique = [technique]
        if prior_findings is not None and not isinstance(prior_findings, list):
            prior_findings = [[prior_findings]]
        if prior_findings is not None and isinstance(prior_findings, list) and not isinstance(prior_findings[0], list):
            prior_findings = [prior_findings]
        if prior_impression is not None and not isinstance(prior_impression, list):
            prior_impression = [[prior_impression]]
        if prior_impression is not None and isinstance(prior_impression, list) and not isinstance(prior_impression[0], list):
            prior_impression = [prior_impression]
        if study_datetime is not None and not isinstance(study_datetime, list):
            study_datetime = [study_datetime]

        if prior_study_datetime is not None and not all(isinstance(x, list) for x in prior_study_datetime):
            prior_study_datetime = [prior_study_datetime]

        batch_size = len(images)

        if views is None:
            views = [[None for _, _ in enumerate(i)] for i in images]

        batch = {
            'input_ids': {i: [] for i in range(batch_size)},
            'token_type_ids': {i: [] for i in range(batch_size)},
            'time_deltas': {i: [] for i in range(batch_size)},
            'time_deltas_mask': {i: [] for i in range(batch_size)},
            'attention_mask': [],
        }
        
        non_causal_2d_attention_mask = {i: [] for i in range(batch_size)}
        causal_2d_attention_mask = []
        
        # Map the prior study time delta values using the time delta map:
        if prior_study_datetime is not None:
            prior_study_time_deltas = [
                [self.time_delta_map(compute_time_delta(j, k)) if j is not None else float('nan') for j in i] for i, k in zip(prior_study_datetime, study_datetime, strict=True)
            ]

        # Findings and impression sections from prior studies:
        for i, token_type_id in zip([prior_findings, prior_impression], self.prior_section_token_type_ids, strict=True):
            if not i:
                continue
            assert len(i) == batch_size, f'Length of {i} must be equal to the batch size: {batch_size}.'
            for j in range(len(i)):
                if not i[j]:
                    continue
                for k in range(len(i[j])):
                    if not i[j][k]:
                        continue
                    batch['input_ids'][j].append(self.tokenizer.encode(i[j][k], add_special_tokens=False, return_tensors='pt')[0])
                    batch['token_type_ids'][j].append(torch.full((batch['input_ids'][j][-1].shape[-1],), token_type_id, dtype=torch.long))
                    non_causal_2d_attention_mask[j].append((batch['input_ids'][j][-1] != self.tokenizer.pad_token_id).long())
                    batch['time_deltas'][j].append(
                        torch.full(
                            (batch['input_ids'][j][-1].shape[-1],), 
                            prior_study_time_deltas[j][k] if prior_study_time_deltas is not None and prior_study_time_deltas[j][k] is not None else float('nan'), 
                            dtype=torch.float32,
                        ),
                    )
                    batch['time_deltas_mask'][j].append(torch.full((batch['input_ids'][j][-1].shape[-1],), 1.0, dtype=torch.float32))

        # Sections of the report for the prompt:
        for i, token_type_id in zip([indication, history, comparison, technique], self.section_token_type_ids, strict=True):
            if not i:
                continue
            assert len(i) == batch_size, f'Length of {i} must be equal to the batch size: {batch_size}.'
            for j, k in enumerate(i):
                if not k:
                    continue
                batch['input_ids'][j].append(self.tokenizer.encode(k, add_special_tokens=False, return_tensors='pt')[0])
                batch['token_type_ids'][j].append(torch.full((batch['input_ids'][j][-1].shape[-1],), token_type_id, dtype=torch.long))
                non_causal_2d_attention_mask[j].append((batch['input_ids'][j][-1] != self.tokenizer.pad_token_id).long())
                batch['time_deltas'][j].append(
                    torch.full((batch['input_ids'][j][-1].shape[-1],), self.zero_time_delta_value, dtype=torch.float32),
                )
                batch['time_deltas_mask'][j].append(torch.full((batch['input_ids'][j][-1].shape[-1],), 1.0, dtype=torch.float32))

        # Labels; findings and impression:
        if train:
            batch['label_ids'] = []
            for i, (j, k) in enumerate(zip(findings, impression, strict=True)):
                
                if j is not None and k is not None:
                    report = f'{self.tokenizer.bos_token}{j}{self.tokenizer.sep_token}{k}{self.tokenizer.eos_token}'
                elif j is not None and k is None:
                    report = f'{self.generate_findings_token}{j}{self.tokenizer.eos_token}'
                elif j is None and k is not None:
                    report = f'{self.generate_impression_token}{k}{self.tokenizer.eos_token}'
                else:
                    raise ValueError('Both findings and impression cannot be None.')
                
                report_ids = self.tokenizer.encode(
                    report,
                    truncation=True,
                    max_length=self.max_generated_tokens + 1,  # +1 to account for the bias between input and target.
                    return_tensors='pt',
                    add_special_tokens=False,
                )[0]
            
                # Labels for the decoder (shifted right by one for autoregression):
                batch['label_ids'].append(report_ids[1:].clone())

                # Remove last token identifier to match the sequence length of the labels:
                batch['input_ids'][i].append(report_ids[:-1])
            
                report_token_type_ids = self.token_ids_to_token_type_ids(token_ids=batch['input_ids'][i][-1])
                batch['token_type_ids'][i].append(report_token_type_ids)
                
                causal_2d_attention_mask.append((batch['input_ids'][i][-1] != self.tokenizer.pad_token_id).long())
                
                batch['time_deltas'][i].append(
                    torch.full((batch['input_ids'][i][-1].shape[-1],), self.zero_time_delta_value, dtype=torch.float32),
                )
                
                batch['time_deltas_mask'][i].append(torch.full((batch['input_ids'][i][-1].shape[-1],), 0.0, dtype=torch.float32))

        else:  # Add special tokens for generation:
            for i in range(batch_size):
                
                bos_token_id = self.tokenizer.bos_token_id
                batch['token_type_ids'][i].append(torch.tensor([self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['findings'])], dtype=torch.long))
                                
                batch['input_ids'][i].append(torch.tensor([bos_token_id], dtype=torch.long))
                                
                causal_2d_attention_mask.append(torch.tensor([1], dtype=torch.long))
                
                batch['time_deltas'][i].append(torch.tensor([self.zero_time_delta_value], dtype=torch.float32))
                batch['time_deltas_mask'][i].append(torch.tensor([0.0], dtype=torch.float32))

        # Map the image time delta values using the time delta map:
        if study_datetime is not None:
            image_time_deltas = [[self.time_delta_map(compute_time_delta(j, k)) if j is not None else float('nan') for j in i] for i, k in zip(image_datetime, study_datetime, strict=True)] 
        else:
            image_time_deltas = [[float('nan') for _ in range(len(i))] for i in images]

        # Randomly select max_train_images_per_study if the number of images for a study exceeds max_train_images_per_study.
        for i in range(len(images)):
            if len(images[i]) > self.max_train_images_per_study:
                paired = list(zip(images[i], views[i], image_time_deltas[i], strict=True))
                sampled_pairs = random.sample(paired, self.max_train_images_per_study)
                images[i], views[i], image_time_deltas[i] = map(list, zip(*sampled_pairs, strict=True))

        # Sort based on views:
        images, views, image_time_deltas = self.sort_images(images, views, image_time_deltas)
                
        # Images:
        max_images = max(len(i) for i in images)
        for i in range(batch_size):
            for j in range(max_images):
                if j < len(images[i]):

                    image_np = None

                    if isinstance(images[i][j], bytes):
                        image = Image.open(io.BytesIO(images[i][j]))

                    elif isinstance(images[i][j], str):
                        if images[i][j].endswith('.dcm'):
                            assert self.mimic_cxr_normalisation, 'MIMIC-CXR normalisation must be True when using DICOM images.'
                            ds = pydicom.dcmread(images[i][j])
                            image_np = ds.pixel_array.astype(float)
                    
                        else:
                            if images[i][j].startswith('http://') or images[i][j].startswith('https://'):
                                response = requests.get(images[i][j], stream=True)
                                image = Image.open(BytesIO(response.content))
                            else:
                                image = Image.open(images[i][j])
                    
                    elif isinstance(images[i][j], Image.Image):
                        image = images[i][j]

                    if self.mimic_cxr_normalisation:

                        # MIMIC-CXR normalisation:
                        if image_np is None:
                            image_np = np.array(image.convert('L'), dtype=np.float32)
                        assert image_np.ndim == 2
                        min_val = image_np.min()
                        denom = image_np.max() - min_val
                        if denom == 0:
                            raise ValueError(f'Cannot normalise image with zero dynamic range (min and max both {min_val}).')
                        image_np = (image_np - min_val) / denom
                        image_uint8 = (image_np * 255).astype(np.uint8)
                        image_eq = cv2.equalizeHist(image_uint8)
                        image = Image.fromarray(image_eq)

                    if self.convert_to_rgb:
                        image = image.convert('RGB')

                    images[i][j] = self.image_processor(image, return_tensors='pt')['pixel_values'].squeeze(0)

                    batch['time_deltas'][i].insert(j, torch.full((self.embeddings_per_image,), image_time_deltas[i][j]))
                    batch['time_deltas_mask'][i].insert(j, torch.full((self.embeddings_per_image,), 1.0))

                    token_type_id = self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['image']) if image_time_deltas[i][j] == self.zero_time_delta_value else self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['prior_image'])
                    batch['token_type_ids'][i].insert(j, torch.full((self.embeddings_per_image,), token_type_id))
                    
                    non_causal_2d_attention_mask[i].insert(j, torch.full((self.embeddings_per_image,), 1))
                    
                else:
                    
                    batch['time_deltas'][i].insert(j, torch.full((self.embeddings_per_image,), 0.0))
                    batch['time_deltas_mask'][i].insert(j, torch.full((self.embeddings_per_image,), 0.0))

                    batch['token_type_ids'][i].insert(j, torch.full((self.embeddings_per_image,), self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['image'])))
                    
                    non_causal_2d_attention_mask[i].insert(j, torch.full((self.embeddings_per_image,), 0))

                    
            images[i] = torch.stack(images[i])
            batch['input_ids'][i].insert(0, self.tokenizer.encode(self.image_token * self.embeddings_per_image * max_images, add_special_tokens=False, return_tensors='pt')[0])
                                                        
        batch['pixel_values'] = pad_sequence(images, batch_first=True, padding_value=0.0)
                
        # Concatenate input_ids, token_type_ids, time_deltas, and time_deltas_mask:
        batch['input_ids'] = [torch.cat(j, dim=0) for j in batch['input_ids'].values()]
        batch['token_type_ids'] = [torch.cat(j, dim=0) for j in batch['token_type_ids'].values()]
        batch['time_deltas'] = [torch.cat(j, dim=0) for j in batch['time_deltas'].values()]
        batch['time_deltas_mask'] = [torch.cat(j, dim=0) for j in batch['time_deltas_mask'].values()]
    
        # Concatentate, and convert label_ids into padded sequences:
        if train:
            batch['label_ids'] = [F.pad(i, (len(j) - len(i), 0), 'constant', self.tokenizer.pad_token_id) for i, j in zip(batch['label_ids'], batch['input_ids'], strict=True)]
            batch['label_ids'] = pad_sequence(batch['label_ids'], batch_first=True, padding_value=self.tokenizer.pad_token_id)   

        # Convert input_ids, token_type_ids, time_deltas, and time_deltas_mask into padded sequences:
        batch['input_ids'] = pad_sequence(batch['input_ids'], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch['token_type_ids'] = pad_sequence(batch['token_type_ids'], batch_first=True, padding_value=0)
        batch['time_deltas'] = pad_sequence(batch['time_deltas'], batch_first=True, padding_value=0)
        batch['time_deltas_mask'] = pad_sequence(batch['time_deltas_mask'], batch_first=True, padding_value=0)    
        
        # Assert that time_delta values are between zero_time_delta_value and inf_time_delta_value:        
        check_1 = torch.all((batch['time_deltas'][~torch.isnan(batch['time_deltas'])] <= max([self.zero_time_delta_value, self.inf_time_delta_value])))
        check_2 = torch.all((batch['time_deltas'][~torch.isnan(batch['time_deltas'])] >= min([self.zero_time_delta_value, self.inf_time_delta_value])))                
        assert check_1 & check_2, 'Time delta values must be between zero_time_delta_value and inf_time_delta_value, or NaN if the time delta is missing.'
        
        # Mixed causality mask:
        non_causal_2d_attention_mask = [torch.cat(j, dim=0) for j in non_causal_2d_attention_mask.values()]
        batch['attention_mask'] = self.create_4d_mixed_causality_attention_mask(
            non_causal_2d_attention_mask, 
            causal_2d_attention_mask, 
            dtype=batch['pixel_values'].dtype,
        )        
            
        if not train:
            batch['initial_attention_mask'] = batch['attention_mask'].clone()  # For the first iteration of generation.
            batch['attention_mask'] = (batch['attention_mask'].squeeze(1).diagonal(dim1=1, dim2=2) == 0.0).long()
            
        # Create position_ids from time_deltas and attention_mask:
        batch['position_ids'] = self.position_ids_from_time_deltas_and_attention_mask(batch['time_deltas'], batch['attention_mask'])

        rows, cols = (batch['input_ids'] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)
        assert all(batch['token_type_ids'][rows, cols] == self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['findings']))
        
        rows, cols = (batch['input_ids'] == self.tokenizer.bos_token_id).nonzero(as_tuple=True)
        assert all(batch['token_type_ids'][rows, cols] == self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['findings']))             

        return BatchFeature(data=batch)

    @staticmethod
    def sort_images(images, views, image_time_deltas):
        def sort_by_view(images, views, time_deltas):
            paired = list(zip(images, views, time_deltas, strict=True))
            sorted_pairs = sorted(paired, key=lambda x: VIEW_ORDER.index(x[1]))
            sorted_images, sorted_views, sorted_time_deltas = map(list, zip(*sorted_pairs, strict=True))
            return sorted_images, sorted_views, sorted_time_deltas

        # Apply sorting to each set of images, views, and time deltas:
        sorted_results = [sort_by_view(i, j, k) for i, j, k in zip(images, views, image_time_deltas, strict=True)]

        sorted_images = [result[0] for result in sorted_results]
        sorted_views = [result[1] for result in sorted_results]
        sorted_time_deltas = [result[2] for result in sorted_results]

        return sorted_images, sorted_views, sorted_time_deltas

    def token_ids_to_token_type_ids(self, token_ids, num_report_tokens=None):
        findings_id   = self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['findings'])
        impression_id = self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['impression'])

        # Initialize all as 'findings':
        token_type_ids = torch.full_like(token_ids, findings_id)

        # Detect sep_token_id positions:
        sep_positions = (token_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]

        if sep_positions.numel() > 0:
            # Use the first sep_token_id as the split point; change anything after it to 'impression' (this is fine as more than one sep_token_id will be treated as invalid for RL):
            first_sep_token_id = sep_positions[0].item()
            if first_sep_token_id + 1 < token_type_ids.numel():
                token_type_ids[first_sep_token_id + 1:] = impression_id

        return token_type_ids if num_report_tokens is None else token_type_ids[-num_report_tokens:]
        
    def create_4d_mixed_causality_attention_mask(self, non_causal_attention_mask, causal_attention_mask, dtype=torch.float32):
        attention_mask = []
        
        max_len = max([len(i) + len(j) for i, j in zip(non_causal_attention_mask, causal_attention_mask, strict=True)])
            
        for i in range(len(non_causal_attention_mask)):
            attention_mask.append(
                self.create_3d_mixed_causality_attention_mask(
                    non_causal_attention_mask[i], 
                    causal_attention_mask[i], 
                    dtype=dtype,
                )
            )
            pad_len = max_len - attention_mask[-1].shape[-1]
            attention_mask[-1] = F.pad(attention_mask[-1], (0, pad_len, 0, pad_len, 0, 0), 'constant', torch.finfo(dtype).min)
        attention_mask = torch.stack(attention_mask)
        
        return attention_mask

    @staticmethod
    def create_3d_mixed_causality_attention_mask(non_causal_1d_attention_mask, causal_1d_attention_mask, dtype=torch.float32):
        
        # Expand to 2D (seq_len x seq_len):
        upper_left = non_causal_1d_attention_mask[:, None] * non_causal_1d_attention_mask[None, :]
        
        if causal_1d_attention_mask is not None:

            prompt_seq_len = non_causal_1d_attention_mask.shape[-1] 
            report_seq_len = causal_1d_attention_mask.shape[-1]

            # Lower right of attention matrix (causal attention with lower triangular masking):
            causal_mask = torch.tril(torch.ones(report_seq_len, report_seq_len, device=causal_1d_attention_mask.device))
            lower_right = causal_1d_attention_mask[:, None] * causal_1d_attention_mask[None, :]
            lower_right = lower_right * causal_mask

            # Upper right of attention matrix (zeroes):
            upper_right = torch.zeros(prompt_seq_len, report_seq_len, dtype=torch.long, device=causal_1d_attention_mask.device)

            # Lower left of attention matrix:
            lower_left = non_causal_1d_attention_mask[None, :] * causal_1d_attention_mask[:, None]

            # Concatenate blocks:
            left = torch.cat((upper_left, lower_left), dim=0)
            right = torch.cat((upper_right, lower_right), dim=0)
            mixed_causality_3d_attention_mask = torch.cat((left, right), dim=-1)
        else:
            mixed_causality_3d_attention_mask = upper_left

        # Convert dtype and apply masking rules:
        mixed_causality_3d_attention_mask = mixed_causality_3d_attention_mask.to(dtype=dtype)
        mixed_causality_3d_attention_mask[mixed_causality_3d_attention_mask == 0] = torch.finfo(mixed_causality_3d_attention_mask.dtype).min
        mixed_causality_3d_attention_mask[mixed_causality_3d_attention_mask == 1] = 0.0
        
        # Add head dimension:
        mixed_causality_3d_attention_mask = mixed_causality_3d_attention_mask.unsqueeze(0)

        return mixed_causality_3d_attention_mask

    def position_ids_from_time_deltas_and_attention_mask(self, time_deltas, attention_mask):
        
        # Set NaNs to inf_time_delta_value:
        time_deltas = torch.nan_to_num(time_deltas, nan=self.inf_time_delta_value)

        # Convert attention mask to 2D if it is 4D:
        if attention_mask.dim() == 4:
            attention_mask = (attention_mask.squeeze(1).diagonal(dim1=1, dim2=2) == 0.0).long()

        # Set time deltas to NaN where the attention mask is 0:
        mask_value = float('inf') if self.time_delta_monotonic_inversion else -float('inf')
        masked_time_deltas = torch.where(attention_mask == 1, time_deltas, mask_value)

        # Sort time deltas and get indices
        sorted_time_deltas, col_indices = masked_time_deltas.sort(
            dim=1, descending=not self.time_delta_monotonic_inversion, stable=True
        )

        num_rows, num_cols = time_deltas.shape

        row_indices = torch.arange(num_rows, device=time_deltas.device).view(-1, 1).repeat(1, num_cols).view(-1)
        position_ids = torch.zeros_like(col_indices, device=time_deltas.device)
        position_ids[row_indices, col_indices.flatten()] = torch.arange(num_cols, device=time_deltas.device)[None, :].expand(num_rows, -1).flatten()

        # Apply the attention mask to zero out invalid positions
        position_ids = position_ids.masked_fill(attention_mask == 0, 1) # Following: https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/llama/modeling_llama.py#L1285.

        for i in range(position_ids.shape[0]):
            assert self.validate_position_ids(position_ids[i])
            
        return position_ids
    
    @staticmethod
    def validate_position_ids(tensor, repeat_value=1):
        unique, counts = torch.unique(tensor, return_counts=True)

        # Check if all integers from 0 to tensor.max() exist:
        full_range = torch.arange(0, tensor.max() + 1, device=tensor.device)
        if not torch.equal(unique.sort()[0], full_range):
            return False

        # Check for repeated values except for repeat_value:
        repeated = unique[counts > 1]
        if repeated.nelement() == 0:
            return True
        if not (repeated.numel() == 1 and repeated.item() == repeat_value):
            return False

        return True
    
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def split_and_decode_sections(self, token_ids):
        """
        Split the token identifiers into sections, then convert the token identifiers into strings.

        Argument/s:
            token_ids - token identifiers.

        Returns:
            token_type_ids - token type identifiers.
        """
        
        sections = {'findings': [], 'impression': []}
        for i in token_ids:
            findings_start_idx = (i == self.tokenizer.bos_token_id).int().argmax().item()
            findings_end_idx = (i == self.tokenizer.sep_token_id).int().argmax().item()
            sections['findings'].append(self.tokenizer.decode(i[findings_start_idx:findings_end_idx], skip_special_tokens=True))
            impression_start_idx = findings_end_idx + 1
            impression_end_idx = (i == self.tokenizer.eos_token_id).int().argmax().item()
            sections['impression'].append(self.tokenizer.decode(i[impression_start_idx:impression_end_idx], skip_special_tokens=True))

        return tuple(sections.values())

    def update_batch_for_rl(self, batch, completion_ids):

        batch_size, prompt_len = batch['token_type_ids'].shape

        # Number of completion tokens:
        num_completion_tokens = completion_ids.shape[1] - prompt_len - 1  # -1 for offset between input and label ids.
        
        # Update mask for completion tokens:
        completion_mask = (completion_ids[:,-(num_completion_tokens + 1):] != self.tokenizer.pad_token_id).float()  # +1 to ignore offset.
        batch['completion_mask'] = completion_mask
        completion_mask_expanded = completion_mask[:, None, None, 1:]  # Start from 1 to reintroduce offset.
        completion_mask_expanded_t = completion_mask[:, None, 1:, None]  # Start from 1 to reintroduce offset.
        
        upper_right = torch.zeros(batch_size, 1, prompt_len, num_completion_tokens, dtype=batch['initial_attention_mask'].dtype, device=completion_ids.device)

        bottom_right = torch.tril(torch.ones(num_completion_tokens, num_completion_tokens, device=completion_ids.device)).bool()
        bottom_right = bottom_right.unsqueeze(0).unsqueeze(0)
        bottom_right = bottom_right.expand(batch_size, -1, -1, -1) 
        bottom_right = bottom_right * completion_mask_expanded * completion_mask_expanded_t
        
        lower_left = batch['attention_mask'][:, None, None, :]
        lower_left = lower_left.expand(-1, -1, num_completion_tokens, -1)
        lower_left = lower_left * completion_mask_expanded_t
        
        right = torch.cat((upper_right, bottom_right), dim=2)
        right[right == 0] = torch.finfo(right.dtype).min
        right[right == 1] = 0.0
        
        lower_left[lower_left == 0] = torch.finfo(lower_left.dtype).min
        lower_left[lower_left == 1] = 0.0
        
        batch['attention_mask'] = torch.cat((batch['initial_attention_mask'], lower_left), dim=2)
        batch['attention_mask'] = torch.cat((batch['attention_mask'], right), dim=3)
        
        # initial_attention_mask was the 4D attention mask, whereas attention_mask was the 2D attention mask (i.e., not needed now that attention_mask is 4D):
        batch.pop('initial_attention_mask', None)

        # Convert remaining batch elements:
        new_token_type_ids = torch.stack([self.token_ids_to_token_type_ids(
            token_ids=i[-num_completion_tokens:], 
            # special_token_ids=[self.tokenizer.sep_token_id],
            # token_type_id_sections=[self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['findings']), self.tokenizer.convert_tokens_to_ids(self.token_type_to_token['impression'])],
        ) for i in completion_ids])
        batch['token_type_ids'] = torch.cat((batch['token_type_ids'], new_token_type_ids), dim=1)
        batch['time_deltas'] = torch.nn.functional.pad(batch['time_deltas'], (0, num_completion_tokens), value=0.0)
        batch['time_deltas_mask'] = torch.nn.functional.pad(batch['time_deltas_mask'], (0, num_completion_tokens), value=0.0)
        
        start_values = batch['position_ids'].max(dim=1).values + 1
        end_values = start_values + num_completion_tokens
        position_ids = torch.stack([torch.arange(i, j, device=batch['position_ids'].device) for i, j in zip(start_values, end_values)])
        batch['position_ids'] = torch.cat((batch['position_ids'], position_ids), dim=1)
        
        batch['label_ids'] = completion_ids[:, 1:].clone()
        batch['input_ids'] = completion_ids[:, :-1]

        # Convert token identifiers that weren't sampled to pad_token_id:
        for i in range(batch_size):
            idx = (batch['label_ids'][i] == self.tokenizer.bos_token_id).nonzero(as_tuple=False)[0, 0].item()
            batch['label_ids'][i][:idx+1] = self.tokenizer.pad_token_id

        
        return batch

    def wrap_dataset(self, dataset):
        return CXRMate2Dataset(dataset)
