import logging
from transformers import AutoTokenizer

from ..types import PytorchModelConfig

logger = logging.getLogger(__name__)


def get_device_preferred_dtype(device: str) -> Union[torch.dtype, None]:
    if device == "cpu":
        return torch.float32
    elif device == "cuda" or device == "mps" or device == "npu":
        return torch.float16
    elif device == "xpu":
        return torch.bfloat16

    return None

class PytorchModel(LLM):
    def __init__(
        self,
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
        ):
        super().__init__(quantization, model_path)
        self._use_fast_tokenizer = True
        self._pytorch_model_config: PytorchModelConfig = self._sanitize_model_config(
            pytorch_model_config
        )

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        if pytorch_model_config is None:
            pytorch_model_config = PytorchModelConfig()
        pytorch_model_config.setdefault("revision", self.model_spec.model_revision)
        pytorch_model_config.setdefault("gptq_ckpt", None)
        pytorch_model_config.setdefault("gptq_wbits", 16)
        pytorch_model_config.setdefault("gptq_groupsize", -1)
        pytorch_model_config.setdefault("gptq_act_order", False)
        pytorch_model_config.setdefault("device", "auto")
        pytorch_model_config.setdefault("trust_remote_code", True)
        pytorch_model_config.setdefault("max_num_seqs", 16)
        pytorch_model_config.setdefault("enable_tensorizer", False)
        return pytorch_model_config

    def _load_model(self, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            **kwargs,
        )

        return model, tokenizer

    def load(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
            )

        quantization = self.quantization
        device = self._pytorch_model_config.get("device", "auto")
        self._pytorch_model_config["device"] = select_device(device)
        self._device = self._pytorch_model_config["device"]

        kwargs = {}

        dtype = get_device_preferred_dtype(self._device)

        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        else:
            raise ValueError(f"Device {self._device} is not supported in temporary")

        kwargs["revision"] = self._pytorch_model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._pytorch_model_config.get(
            "trust_remote_code"
        )

        (
            self._model,
            self._tokenizer,
        ) = load_compress_model(
            model_path=self.model_path,
            device=self._device,
            torch_dtype=kwargs["torch_dtype"],
            use_fast=self._use_fast_tokenizer,
            revision=kwargs["revision"],
        )
        return 