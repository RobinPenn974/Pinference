import glob
import tqdm
import gc
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def empty_cache():
    """
    清空各种计算设备的缓存。
    
    此函数会检查并清空可用的计算设备的缓存，包括：
    - CUDA(GPU)
    - MPS(Metal Performance Shaders,用于苹果芯片)
    - XPU(英特尔特定加速器)
    - NPU(神经网络处理器)
    
    这有助于释放内存，特别是在处理大型模型或数据集时。
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if is_xpu_available():
        torch.xpu.empty_cache()
    if is_npu_available():
        torch.npu.empty_cache()



def load_compress_model(
    model_path: str,
    device: str,
    torch_dtype: torch.dtype,
    use_fast: bool,
    trust_remote_code: bool=True,
    low_cpu_mem_usage: bool=True,
    revision: str = "main"
):
    """
    加载并可选压缩预训练的语言模型。

    此函数执行以下操作：
    1. 初始化分词器
    2. 使用空权重创建模型结构
    3. 逐个加载模型权重文件
    4. 可选地压缩大型线性层(待实现)
    5. 将加载的权重应用到模型

    参数:
    model_path (str): 模型路径或Hugging Face模型名称
    device (str): 目标设备(如 'cuda', 'cpu')
    torch_dtype (torch.dtype): PyTorch数据类型
    use_fast (bool): 是否使用快速分词器
    trust_remote_code (bool): 是否信任远程代码(默认为True)
    low_cpu_mem_usage (bool): 是否使用低CPU内存模式(默认为True)
    revision (str): 模型版本，默认为"main"

    返回:
    tuple: 包含加载(和可能压缩)的模型及其分词器

    注意:
    - 此函数目前不执行实际的模型压缩，压缩功能在TODO注释中标记待实现。
    - 函数假设模型文件存在于本地路径，否则会抛出FileNotFoundError。
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
        revision=revision
    )

    # 加载空权重创建模型结构
    with init_empty_weights():
        config = AutoConfig.from_pretrained(
            model_path,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            revision=revision
        )
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code
        )

    # 确定权重文件的位置
    if os.path.exists(model_path):
        base_pattern = os.path.join(model_path, "pytorch_model*.bin")
    else:
        raise FileNotFoundError(
            f"Model file {base_pattern} does not exist in {model_path}."
        )

    files = glob.glob(base_pattern)
    
    # TODO: 使用 compress 函数对大型线性层进行压缩。
    compressed_state_dict = {}
    
    # 分段加载
    for filename in tqdm(files):
        tmp_state_dict = torch.load(filename, map_location=torch.device(device))
        for name in tmp_state_dict:
            # TODO: 判断如果是线性层进行压缩
            compressed_state_dict[name] = tmp_state_dict[name].to(device)
            # 缓存清除
            tmp_state_dict[name] = None
            gc.collect()
            empty_cache()

    # 逐个将权重加载到模型
    for name in model.state_dict():
        # TODO: 对线性层特殊处理
        set_module_tensor_to_device(
                model, name, device, value=compressed_state_dict[name]
            )

    model.to(device)
    return model, tokenizer