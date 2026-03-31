from peft import LoraConfig, get_peft_model
from stages_cxrmate2 import Stages as BaseStages
from stages_cxrmate2_dpo import Stages as DPOStages


class Stages(DPOStages):

    def __init__(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    def init_model(self):
        super().init_model()
        super().warm_start()

        # Apply LoRA to the language model only:
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=['q_proj', 'v_proj'],
            bias='none',
            task_type='CAUSAL_LM',
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        self.model.language_model.print_trainable_parameters()

        # Freeze everything except LoRA parameters:
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

    def warm_start(self):
        pass

    def training_epoch(self, epoch):
        
        return BaseStages.training_epoch(self, epoch)
    