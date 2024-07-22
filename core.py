import abc
from abc import abstractmethod
from typing import Tuple

class LLM(abc.ABC):
    def __init__(
        self,
        quantization: str,
        model_path: str,
        *args,
        **kwargs,
        ):
        pass
    self.quantization = quantization
    self.model_path = model_path
    if args:
        raise ValueError(f"Unrecognized positional arguments: {args}")
    if kwargs:
        raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    @abstractmethod
    def load(self):
        raise NotImplementedError


class LLMDescription(ModelDescription):
    def __init__(self):
        pass

def create_llm_model_instance(
    model_name: str,
    model_engine: Optional[str],
    model_format: Optional[str] = None,
    quantization: Optional[str] = None,
    **kwargs,
) -> Tuple[LLM, LLMDescription]:
    pass