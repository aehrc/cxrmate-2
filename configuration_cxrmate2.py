from typing import Any, Optional

from transformers import CONFIG_MAPPING, AutoConfig
from transformers.configuration_utils import PretrainedConfig


class CXRMate2Config(PretrainedConfig):
    model_type = 'cxrmate-2'

    sub_configs = {'text_config': AutoConfig, 'vision_config': AutoConfig}
    is_composition = True

    def __init__(
        self,
        vision_config: PretrainedConfig = None,
        text_config: PretrainedConfig = None,
        num_token_types: int = None,
        num_q_adapter_queries: int = None,
        num_q_adapter_layers: int = None,
        num_q_adapter_positions: int = None,
        findings_token_type_id: int = None,
        impression_token_type_id: int = None,
        image_token: str = None,
        image_token_index: int = None,
        permute_encoder_last_hidden_state: bool = False,
        time_delta_encoder_intermediate_size: int = 2048,
        time_delta_monotonic_inversion: bool = True,
        vision_feature_layer=-1,
        vision_feature_select_strategy='full',
        missing_time_delta_token_id: int = None,
        generate_both_sections_token_id: str = None,
        generate_findings_token_id: str = None,
        generate_impression_token_id: str = None,
        multimodal_projector_bias: bool = False,
        projector_hidden_act: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.num_token_types = num_token_types
        self.num_q_adapter_queries = num_q_adapter_queries
        self.num_q_adapter_layers = num_q_adapter_layers
        self.num_q_adapter_positions = num_q_adapter_positions
        self.findings_token_type_id = findings_token_type_id
        self.impression_token_type_id = impression_token_type_id
        self.image_token = image_token
        self.image_token_index = image_token_index
        self.permute_encoder_last_hidden_state = permute_encoder_last_hidden_state
        self.time_delta_encoder_intermediate_size = time_delta_encoder_intermediate_size
        self.time_delta_monotonic_inversion = time_delta_monotonic_inversion
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.missing_time_delta_token_id = missing_time_delta_token_id
        self.generate_both_sections_token_id = generate_both_sections_token_id
        self.generate_findings_token_id = generate_findings_token_id
        self.generate_impression_token_id = generate_impression_token_id
        self.multimodal_projector_bias = multimodal_projector_bias
        self.projector_hidden_act = projector_hidden_act

        if isinstance(vision_config, dict):
            vision_config = CONFIG_MAPPING[vision_config['model_type']](**vision_config)
        if isinstance(text_config, dict):
            text_config = CONFIG_MAPPING[text_config['model_type']](**text_config)
        self.vision_config = vision_config
        self.text_config = text_config

        super().__init__(**kwargs)
