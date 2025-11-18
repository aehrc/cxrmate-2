import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoBackbone, AutoModelForCausalLM, RoFormerConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.models.roformer.modeling_roformer import RoFormerLayer
from transformers.utils import check_min_version, logging

try:
    from .configuration_cxrmate2 import CXRMate2Config
except ImportError:
    from configuration_cxrmate2 import CXRMate2Config

logger = logging.get_logger(__name__)      


class CXRMate2FNNEncoder(torch.nn.Module):
    def __init__(self, num_features, intermediate_size, hidden_size):
        super().__init__()
        self.up_proj = torch.nn.Linear(num_features, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = torch.nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class CXRMate2QAdapter(torch.nn.Module):  # Inspired by the Perceiver Resampler (e.g., used by Flamingo).
    def __init__(self, config):
        super().__init__()
        
        self.num_q_adapter_queries = config.num_q_adapter_queries
        
        # https://huggingface.co/docs/transformers/en/model_doc/roformer#transformers.RoFormerConfig
        roformer_config = RoFormerConfig(
            num_hidden_layers=config.num_q_adapter_layers,
            max_position_embeddings=config.num_q_adapter_positions,
            is_decoder=False
        )
        
        self.queries = torch.nn.Parameter(torch.empty(config.num_q_adapter_queries, roformer_config.hidden_size))
        torch.nn.init.xavier_uniform_(self.queries)        

        self.layers = torch.nn.ModuleList(
            [RoFormerLayer(roformer_config) for _ in range(roformer_config.num_hidden_layers)]
        )

        self.projection = torch.nn.Linear(roformer_config.hidden_size, config.text_config.hidden_size)

    def forward(self, x):

        queries = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat((queries, x), dim=1)

        for layer in self.layers:
            x, *_ = layer(hidden_states=x)

        x = x[:, :self.num_q_adapter_queries, :]
        
        x = self.projection(x)

        return x
    

class CXRMate2PreTrainedModel(PreTrainedModel):
    config: CXRMate2Config
    base_model_prefix = ''
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = 'past_key_values'

    _supports_flash_attn = False
    _supports_sdpa = False

    _can_compile_fullgraph = False
    _supports_flex_attn = False
    _supports_attention_backend = False


class CXRMate2ForConditionalGeneration(CXRMate2PreTrainedModel, GenerationMixin):

    config_class = CXRMate2Config

    def __init__(self, config: CXRMate2Config) -> None:

        super(CXRMate2PreTrainedModel, self).__init__(config)
        
        assert self.config.sep_token_id is not None
        assert self.config.bos_token_id is not None
        
        self.permute_encoder_last_hidden_state = config.permute_encoder_last_hidden_state
        
        self.vision_tower = AutoBackbone.from_config(
            config.vision_config,
            torch_dtype=config.vision_config.torch_dtype,
        )
        self.multi_modal_projector = CXRMate2QAdapter(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config,
            attn_implementation=config._attn_implementation,
            trust_remote_code=True,
            torch_dtype=config.text_config.torch_dtype,
        )
                
        self.time_delta_encoder = CXRMate2FNNEncoder(
            num_features=1, 
            intermediate_size=config.time_delta_encoder_intermediate_size, 
            hidden_size=config.text_config.hidden_size,
        )  

        self.register_buffer('missing_time_delta_token_id', torch.tensor(self.config.missing_time_delta_token_id), persistent=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_image_features(
        self, pixel_values: torch.FloatTensor, vision_feature_layer: int, vision_feature_select_strategy: str
    ) -> torch.Tensor:

        # Flatten the batch and study_id dimensions:
        assert len(pixel_values.shape) == 5, 'pixel_values must be B, S, C, H, W, where S is the max number of images for a study in the batch.'
        image_outputs = self.vision_tower(pixel_values.view(-1, *pixel_values.shape[2:]), output_hidden_states=True)
        image_features = image_outputs.feature_maps[vision_feature_layer]

        # Flatten h x w:
        image_features = torch.flatten(image_features, 2) if image_features.dim() > 3 else image_features
        image_features = torch.permute(image_features, [0, 2, 1]) if self.permute_encoder_last_hidden_state else image_features

        if vision_feature_select_strategy == 'default':
            image_features = image_features[:, 1:]
        elif vision_feature_select_strategy == 'full':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature strategy: {self.config.vision_feature_select_strategy}')

        image_features = self.multi_modal_projector(image_features)
        
        # Concatenate the features for each chest X-ray:
        image_features = image_features.view(pixel_values.shape[0], -1, image_features.shape[-1])
        
        return image_features 

    def forward(
        self,
        token_type_ids: torch.LongTensor,
        time_deltas: torch.FloatTensor,
        time_deltas_mask: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        pixel_values: Optional[torch.FloatTensor] = None,
        initial_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one'
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            n_image_features = image_features.shape[0] * image_features.shape[1]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f'Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}'
                )
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Add token type embeddings:
        token_type_embeddings = self.get_input_embeddings()(token_type_ids)
        inputs_embeds += token_type_embeddings
         
        # Add time delta embeddings:
        missing_time_delta_mask = time_deltas.isnan()
        time_deltas = time_deltas.nan_to_num(0) # Replace NaN with dummy value before projection.
        time_delta_embeddings = self.time_delta_encoder(time_deltas.unsqueeze(-1)) 
        time_delta_embeddings[missing_time_delta_mask] = self.get_input_embeddings()(self.missing_time_delta_token_id)
        time_delta_embeddings *= time_deltas_mask.unsqueeze(-1)
        inputs_embeds += time_delta_embeddings

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = {}          

        model_inputs.update(
            self.language_model.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **kwargs,
            ),
        )

        if cache_position[0] == 0:
            model_inputs['pixel_values'] = pixel_values
            model_inputs['attention_mask'] = kwargs['initial_attention_mask']  # Use the 4D attention mask for the initial iteration.
        else:

            # Always ensure that the token_type_ids are computed from all of the input_ids, not just the last to be generated:
            model_inputs['token_type_ids'] = self.token_ids_to_token_type_ids(input_ids)
            
            # Time deltas:
            model_inputs['time_deltas'] = torch.zeros_like(
                model_inputs['token_type_ids'], dtype=torch.float, device=model_inputs['token_type_ids'].device
            )  # These will be masked; no need to set this to inf_time_delta_value.
            model_inputs['time_deltas_mask'] = torch.zeros_like(model_inputs['token_type_ids'], dtype=torch.float, device=model_inputs['token_type_ids'].device)
            
            # Position identifiers:
            model_inputs['position_ids'] = kwargs['position_ids'].max(dim=1).values.unsqueeze(-1) + (input_ids.shape[1] - kwargs['position_ids'].shape[1])
            
            # Validate that the findings token type identifier is used for sep_token_id:
            mask = input_ids[:, -1] == self.config.sep_token_id
            if mask.any():
                assert (model_inputs['token_type_ids'][mask] == self.config.findings_token_type_id).all()
            
            # Validate that the impression token type identifier is used after sep_token_id:
            mask = (input_ids[:, :-1] == self.config.sep_token_id).any(dim=1)
            if mask.any():
                assert (model_inputs['token_type_ids'][mask] == self.config.impression_token_type_id).all()
   
        model_inputs.pop('initial_attention_mask', None)
            
        return model_inputs

    def token_ids_to_token_type_ids(self, token_ids):

        assert token_ids.ndim == 2

        token_type_ids = []
        for i in token_ids:
            if self.config.sep_token_id in i[:-1]:
                token_type_ids.append(self.config.impression_token_type_id)
            else:
                token_type_ids.append(self.config.findings_token_type_id)

        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=token_ids.device)[:, None]

        return token_type_ids
