from typing import Iterator, Tuple

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from ....types import CompletionChunk, CompletionUsage
from ....fields import max_tokens_field


def get_context_length(config) -> int:
    """
    Get the context length of a model from a huggingface model config.

    This function examines various attributes of the config object to determine
    the appropriate context length. It looks for 'max_sequence_length', 'seq_length',
    and 'max_position_embeddings', using default values if these are not present.

    Args:
        config: An object containing configuration parameters.

    Returns:
        int: The maximum context length, which is the largest value among
             max_sequence_length, seq_length, and max_position_embeddings.
    """
    if (
        hasattr(config, "max_sequence_length")
        and config.max_sequence_length is not None
    ):
        max_sequence_length = config.max_sequence_length
    else:
        max_sequence_length = 2048
    if hasattr(config, "seq_length") and config.seq_length is not None:
        seq_length = config.seq_length
    else:
        seq_length = 2048
    if (
        hasattr(config, "max_position_embeddings")
        and config.max_position_embeddings is not None
    ):
        max_position_embeddings = config.max_position_embeddings
    else:
        max_position_embeddings = 2048
    return max(max_sequence_length, seq_length, max_position_embeddings)

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    """
    Prepare a list of logits processors based on the given parameters.

    This function creates a LogitsProcessorList with various processors
    to modify the logits distribution during text generation.

    Args:
        temperature (float): Controls randomness in generation. Values > 0.0 and != 1.0 are applied.
        repetition_penalty (float): Penalizes repetition in generated text. Applied if > 1.0.
        top_p (float): Nucleus sampling parameter. Applied if between 1e-8 and 1.0 (exclusive).
        top_k (int): Limits consideration to top k tokens. Applied if > 0.

    Returns:
        LogitsProcessorList: A list of logits processors to be applied during generation.

    Note:
        The function selectively adds processors based on the parameter values,
        skipping those that would have no effect (e.g., temperature of 1.0).
    """
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

def is_sentence_complete(output: str):
    """Check whether the output is a complete sentence."""
    end_symbols = (".", "?", "!", "...", "。", "？", "！", "…", '"', "'", "”")
    return output.endswith(end_symbols)

def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False

@torch.generate_mode()
def generate_stream(
    model_uid: str,
    model,
    tokenizer,
    prompt: str,
    device: str,
    generate_config: "PytorchGenerateConfig",
    judge_sent_end: bool=False,
) -> Iterator[Tuple[CompletionChunk, CompletionUsage]]:
    context_length = get_context_length(model.config)
    stream_interval = generate_config.get("stream_interval", 2)
    stream = generate_config.get("stream", False)
    stream_options = generate_config.pop("stream_options", None)
    include_usage = (
        generate_config["include_usage"] if isinstance(generate_config, dict) else False
    )

    len_prompt = len(prompt)

    temperature = float(generate_config.get("temperature", 1.0))
    repetition_penalty = float(generate_config.get("repetition_penalty", 1.0))
    top_p = float(generate_config.get("top_p", 1.0))
    top_k = int(generate_config.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(generate_config.get("max_tokens", max_tokens_field.default))
    echo = bool(generate_config.get("echo", False))
    stop_str = generate_config.get("stop", None)
    stop_token_ids = generate_config.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)
    chunk_id = str(uuid.uuid4())

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    # Seems Qwen need special tokens
    if ".modeling_qwen." in str(type(model)).lower():
        # TODO: hacky
        input_ids = tokenizer(prompt, allowed_special="all").input_ids
    else:
        input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    # Determine max source length based on model architecture
    # For encoder-decoder models, use full context length
    # For autoregressive models, reserve space for new tokens and special tokens
    # Ensure the requested generation length doesn't exceed model's capacity
    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8
        if max_src_len < 0:
            raise ValueError("Max tokens exceeds model's max length")

    # max_src_len is the max input token number
    # Truncate input_ids to max_src_len, keeping the most recent (rightmost) tokens
    # If max_src_len is greater than input_ids length, do noting.
    input_ids = input_ids[-max_src_len:]
    # Store the actual length of the input after truncation
    input_echo_len = len(input_ids)

    # For encoder-decoder models: encode input and prepare decoder start token
    # There are two steps for encoder-decoder models: 1. encode 2. decode
    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    start = time.time()
    past_key_values = out = None
    sent_interrupt = False
    token = None
    last_output_length = 0
    for i in range(max_new_tokens):
        # Decode
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        # Process the logits for the last token:
        # - Apply logits processor if available (e.g., for repetition penalty)
        # - Extract logits for the last token in the sequence
        # logits(batch_size, sequence_length, config.vocab_size)
        # The line 'last_token_logits = logits[0, -1, :]' means:
        # 1. Select the first (or only) sample from the model output
        # 2. Choose the predictions for the last position in that sample
        # 3. Get the prediction scores for all possible tokens in the vocabulary for that position
        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            if stream:
                output = output.strip("�")
                tmp_output_length = len(output)
                output = output[last_output_length:]
                last_output_length = tmp_output_length

            # prevent yielding partial stop sequence
            if not partially_stopped:
                completion_choice = CompletionChoice(
                    text=output, index=0, logprobs=None, finish_reason=None
                )
                completion_chunk = CompletionChunk(
                    id=chunk_id,
                    object="text_completion",
                    created=int(time.time()),
                    model=model_uid,
                    choices=[completion_choice],
                )
                completion_usage = CompletionUsage(
                    prompt_tokens=input_echo_len,
                    completion_tokens=i,
                    total_tokens=(input_echo_len + i),
                )

                yield completion_chunk, completion_usage

        if stopped:
            break