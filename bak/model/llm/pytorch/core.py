

class PytorchLLM(LLM):
    def __init__(
        self,
        model_uid: int,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._use_fast_tokenizer = True
        self._pytorch_model_config: PytorchModelConfig = self._sanitize_model_config(
            pytorch_model_config
        )
        self._peft_model = peft_model

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

    def generate(
        self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        from .utils import generate_stream

        def generator_wrapper(
            prompt: str, generate_config: PytorchGenerateConfig
        ) -> Iterator[CompletionChunk]:
            for completion_chunk, completion_usage in generate_stream(
                self.model_uid,
                self._model,
                self._tokenizer,
                prompt,
                self._device,
                generate_config,
            ):
                completion_chunk["usage"] = completion_usage
                yield completion_chunk

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        generate_config = self._sanitize_generate_config(generate_config)

        assert self._model is not None
        assert self._tokenizer is not None

        lora_model = generate_config.pop("lora_name")

        if lora_model is not None and self._peft_model is not None:
            for lora in self._peft_model:
                if lora_model == lora.lora_name:
                    self._model.set_adapter(lora_model)
                    logger.info(f"Set lora model to {lora_model}")
                    break
            else:
                self._model.disable_adapter()
                logger.info(f"No lora model {lora_model} found, skip setting")

        stream = generate_config.get("stream", False)
        if not stream:
            for completion_chunk, completion_usage in generate_stream(
                self.model_uid,
                self._model,
                self._tokenizer,
                prompt,
                self._device,
                generate_config,
            ):
                pass
            completion = Completion(
                id=completion_chunk["id"],
                object=completion_chunk["object"],
                created=completion_chunk["created"],
                model=completion_chunk["model"],
                choices=completion_chunk["choices"],
                usage=completion_usage,
            )
            return completion
        else:
            return generator_wrapper(prompt, generate_config)