import abc

from abc import abstractmethod

from ...core.utils import parse_replica_model_uid

class LLM(abc.ABC):
    def __init__(
        self,
        replica_model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        *args,
        **kwargs,
    ):
        self.model_uid, self.replica, self.rep_id = parse_replica_model_uid(
            replica_model_uid
        )
        self.model_family = model_family
        self.model_spec = model_spec
        self.quantization = quantization
        self.model_path = model_path
        if args:
            raise ValueError(f"Unrecognized positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")