from typing import Any, Callable, Dict, ForwardRef, Iterable, List, Optional, Union
from typing_extensions import TypedDict, Literal, NotRequired

class CompletionLogprobs(TypedDict):
    """
    Contains log probability information for a completion.

    Attributes:
        text_offset (List[int]): List of token offsets in the text.
        token_logprobs (List[Optional[float]]): Log probabilities for each token.
        tokens (List[str]): List of generated tokens.
        top_logprobs (List[Optional[Dict[str, float]]]): Top N tokens and their log probabilities for each position.
    """
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]

class ToolCallFunction(TypedDict):
    """
    Represents the details of a function call made by a tool during the completion process.

    Attributes:
        name (str): The name of the function that was called.
        arguments (str): A JSON-encoded string containing the arguments passed to the function.
    """
    name: str
    arguments: str

class ToolCalls(TypedDict):
    """
    Represents a tool call made during completion.

    Attributes:
        id (str): Unique identifier for the tool call.
        type (Literal["function"]): Type of the tool call, currently only "function" is supported.
        function (ToolCallFunction): Details of the function call.
    """
    id: str
    type: Literal["function"]
    function: ToolCallFunction

class CompletionChoice(TypedDict):
    """
    Represents a single completion choice.

    Attributes:
        text (str): The generated text.
        index (int): Index of this choice in the list of choices.
        logprobs (Optional[CompletionLogprobs]): Log probability information, if requested.
        finish_reason (Optional[str]): Reason why the generation stopped.
        tool_calls (NotRequired[List[ToolCalls]]): List of tool calls made during generation, if any.
    """
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[str]
    tool_calls: NotRequired[List[ToolCalls]]

class CompletionUsage(TypedDict):
    """
    Contains token usage information for a completion.

    Attributes:
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.
        total_tokens (int): Total number of tokens used (prompt + completion).
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionChunk(TypedDict):
    """
    Represents a chunk of a completion response.

    Attributes:
        id (str): Unique identifier for this completion chunk.
        object (Literal["text_completion"]): Object type, always "text_completion".
        created (int): Unix timestamp of when this chunk was created.
        model (str): Name of the model used for completion.
        choices (List[CompletionChoice]): List of completion choices in this chunk.
        usage (NotRequired[CompletionUsage]): Token usage statistics, if available.
    """
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: NotRequired[CompletionUsage]

class Completion(TypedDict):
    """
    Represents a complete response from a text completion API.

    Attributes:
        id (str): A unique identifier for this completion.
        object (Literal["text_completion"]): The object type, always "text_completion".
        created (int): The Unix timestamp (in seconds) of when the completion was created.
        model (str): The name of the model used to generate the completion.
        choices (List[CompletionChoice]): A list of generated completion choices.
        usage (CompletionUsage): Statistics about the token usage for this completion.
    """
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class PytorchGenerateConfig(TypedDict, total=False):
    """
    Configuration options for text generation using PyTorch models.

    This TypedDict allows for flexible configuration with optional fields.
    The 'total=False' parameter indicates that all fields are optional.

    Attributes:
        temperature (float): Controls randomness in generation. Higher values (e.g., 1.0) make output more random,
                             lower values (e.g., 0.1) make it more deterministic.
        repetition_penalty (float): Penalizes repetition in generated text. Values > 1.0 discourage repetition.
        top_p (float): Nucleus sampling parameter. Only consider tokens with cumulative probability < top_p.
        top_k (int): Only consider the top k tokens for generation at each step.
        stream (bool): If True, enable streaming of generated text.
        max_tokens (int): Maximum number of tokens to generate.
        echo (bool): If True, include the prompt in the generated output.
        stop (Optional[Union[str, List[str]]]): Stop sequence(s) to end generation.
        stop_token_ids (Optional[Union[int, List[int]]]): Token ID(s) that signal to stop generation.
        stream_interval (int): Interval for yielding tokens when streaming.
        model (Optional[str]): Name or path of the specific model to use.
        tools (Optional[List[Dict]]): List of tool configurations for function calling.
        lora_name (Optional[str]): Name of the LoRA adapter to use, if applicable.
        stream_options (Optional[Union[dict, None]]): Additional options for streaming.
        request_id (Optional[str]): Unique identifier for the generation request.
    """
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int
    stream: bool
    max_tokens: int
    echo: bool
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[Union[int, List[int]]]
    stream_interval: int
    model: Optional[str]
    tools: Optional[List[Dict]]
    lora_name: Optional[str]
    stream_options: Optional[Union[dict, None]]
    request_id: Optional[str]


class PytorchModelConfig(TypedDict, total=False):
    """
    Configuration options for PyTorch models.

    This TypedDict allows for flexible configuration with optional fields.
    The 'total=False' parameter indicates that all fields are optional.

    Attributes:
        revision (Optional[str]): The specific model revision to use, if applicable.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        gpus (Optional[str]): Specific GPUs to use, typically a comma-separated list of device ids.
        num_gpus (int): Number of GPUs to use for model parallelism.
        max_gpu_memory (str): Maximum GPU memory to use per device (e.g., '16GiB').
        gptq_ckpt (Optional[str]): Path to the GPTQ checkpoint file, if using GPTQ quantization.
        gptq_wbits (int): Number of bits to use in GPTQ quantization.
        gptq_groupsize (int): Group size for GPTQ quantization.
        gptq_act_order (bool): Whether to use activation order in GPTQ.
        trust_remote_code (bool): Whether to trust and execute code from remote sources.
        max_num_seqs (int): Maximum number of sequences to process in parallel.
        enable_tensorizer (Optional[bool]): Whether to enable tensorizer optimizations.
    """
    revision: Optional[str]
    device: str
    gpus: Optional[str]
    num_gpus: int
    max_gpu_memory: str
    gptq_ckpt: Optional[str]
    gptq_wbits: int
    gptq_groupsize: int
    gptq_act_order: bool
    trust_remote_code: bool
    max_num_seqs: int
    enable_tensorizer: Optional[bool]

