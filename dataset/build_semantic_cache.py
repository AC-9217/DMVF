"""独立加载本地 Qwen3-Embedding-4B，将全英文元数据编码为训练用缓存。"""

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm


# 直接运行脚本时补充项目根目录，不依赖当前终端所在位置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.semantic_cache import (
    CACHE_FILENAME,
    ENCODER_ID,
    TEXT_DIM,
    build_description,
    make_semantic_cache,
    read_metadata_file,
    save_semantic_cache,
)


# 解析独立预处理命令行，不读取训练配置或创建训练日志。
def parse_args() -> argparse.Namespace:
    """
    功能描述:
        获取本地模型、数据目录、设备、编码批量和输出文件配置。
    参数:
        无显式参数；读取终端命令行参数。
    返回值:
        argparse.Namespace：经过基本合法性检查的参数对象，标量。
    """
    # 默认路径均相对脚本位置确定，模型目录必须提前下载。
    parser = argparse.ArgumentParser(description="独立生成 Qwen3-Embedding-4B 全英文语义缓存。")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "Qwen3-Embedding-4B", help="提前下载好的本地模型目录。")
    parser.add_argument("--model-revision", default="", help="下载时使用的模型提交版本，记录到缓存供追溯。")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "Supplement_WSDREAM_dataset2", help="包含 data 子目录的数据集目录。")
    parser.add_argument("--output", type=Path, default=None, help="输出缓存文件，默认位于数据集目录。")
    parser.add_argument("--device", default="cuda:0", help="编码设备，例如 cuda:0 或 cpu。")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto", help="模型计算精度，自动模式按设备选择。")
    parser.add_argument("--batch-size", type=int, default=4, help="一次编码的去重描述数量。")
    parser.add_argument("--max-length", type=int, default=256, help="允许的最大输入长度，超过时直接报错以防字段被截断。")
    parser.add_argument("--ip-octets", type=int, choices=range(1, 5), default=2, help="IP 前缀保留段数，必须与训练配置一致。")
    parser.add_argument("--overwrite", action="store_true", help="显式允许重建并覆盖已有缓存。")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1 or args.max_length > 32768:
        parser.error("batch-size 必须为正整数，max-length 必须在 1 至 32768 之间。")
    args.model_path = args.model_path.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output = (args.output or args.data_dir / CACHE_FILENAME).expanduser().resolve()
    return args


# 按设备选择模型精度，并提前报告不可用的设备。
def resolve_dtype(device: torch.device, name: str) -> torch.dtype:
    """
    功能描述:
        自动模式下优先使用受支持的低精度，中央处理器默认使用单精度。
    参数:
        device：torch.device，编码设备，标量。
        name：str，命令行指定的精度名称，标量。
    返回值:
        torch.dtype：模型加载和计算使用的数据类型，标量。
    """
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("当前环境没有可用的 CUDA，请安装对应 PyTorch 或使用 --device cpu。")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA 设备编号不存在：{device}。")
        if name == "auto":
            with torch.cuda.device(device):
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name == "auto":
        return torch.float32
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


# 仅从本地目录加载编码模型，避免预处理阶段隐式联网下载。
def load_encoder(model_path: Path, device: torch.device, dtype: torch.dtype) -> tuple[Any, Any]:
    """
    功能描述:
        延迟导入编码依赖，检查模型架构和输出维度，再冻结全部参数。
    参数:
        model_path：Path，本地 Qwen3-Embedding-4B 模型目录，标量。
        device：torch.device，模型运行设备，标量。
        dtype：torch.dtype，模型参数精度，标量。
    返回值:
        tuple[Any, Any]：冻结的编码模型与对应分词器，两个对象标量。
    """
    # 编码库只在独立脚本中导入，训练进程不接触这些依赖。
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("缺少编码依赖，请安装 requirements-semantic-cache.txt 中的依赖。") from error

    # 校验本地模型，拒绝混用零点六十亿参数版本的缓存维度。
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "qwen3" or config.hidden_size != TEXT_DIM:
        raise ValueError(f"本地目录必须包含 {ENCODER_ID}，其输出维度应为 {TEXT_DIM}。")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 使用原生缩放点积注意力，无需额外安装快速注意力扩展。
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    return model, tokenizer


# 从每条序列中提取最后一个有效位置，兼容左右两种填充。
def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    功能描述:
        依据注意力掩码定位最后一个有效词元，避免把填充值作为句向量。
    参数:
        hidden_states：torch.Tensor，最后一层隐藏状态，形状为 [B, L, 2560]。
        attention_mask：torch.Tensor，有效词元掩码，形状为 [B, L]。
    返回值:
        torch.Tensor：每条描述的句向量，形状为 [B, 2560]。
    """
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
    last_positions = positions.masked_fill(~attention_mask.bool(), -1).amax(dim=1)
    if (last_positions < 0).any():
        raise ValueError("编码输入中存在没有有效词元的空序列。")
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_positions]


# 分批执行冻结编码，并将归一化向量直接搬到处理器内存。
def encode_descriptions(
    descriptions: list[str], model: Any, tokenizer: Any, device: torch.device, batch_size: int, max_length: int
) -> torch.Tensor:
    """
    功能描述:
        编码去重后的英文描述，使用最后有效词元池化和单精度归一化。
    参数:
        descriptions：list[str]，去重后的全英文描述，形状为 [N]。
        model：Any，已冻结且处于评估模式的文本编码器，标量。
        tokenizer：Any，对应分词器，标量。
        device：torch.device，编码设备，标量。
        batch_size：int，每批描述条数，标量。
        max_length：int，允许的最大词元长度，标量。
    返回值:
        torch.Tensor：处理器上的单精度归一化句向量，形状为 [N, 2560]。
    """
    embeddings = torch.empty(len(descriptions), TEXT_DIM, dtype=torch.float32)
    with torch.inference_mode():
        for start in tqdm(range(0, len(descriptions), batch_size), desc="编码英文元数据"):
            # 禁止静默截断末尾字段，输入过长时让用户显式提高长度配置。
            texts = descriptions[start:start + batch_size]
            batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
            if batch["input_ids"].shape[1] > max_length:
                raise ValueError(f"英文描述长度超过 {max_length}，请提高 --max-length 后重新编码。")
            batch = {name: tensor.to(device) for name, tensor in batch.items()}

            # 不生成文本、不保留键值缓存，只提取最后一层表示。
            output = model(**batch, use_cache=False, return_dict=True)
            pooled = last_token_pool(output.last_hidden_state, batch["attention_mask"])
            normalized = F.normalize(pooled.float(), p=2, dim=-1)
            if normalized.shape != (len(texts), TEXT_DIM) or not torch.isfinite(normalized).all():
                raise ValueError("编码器输出的向量维度或数值异常。")
            embeddings[start:start + len(texts)].copy_(normalized.cpu())
    return embeddings


# 为缓存记录本地模型版本与配置文件指纹。
def encoder_provenance(model_path: Path, revision: str, dtype: torch.dtype, max_length: int) -> dict[str, Any]:
    """
    功能描述:
        保存模型来源、指定版本及配置指纹，使离线缓存可以追溯。
    参数:
        model_path：Path，本地模型目录，标量。
        revision：str，下载模型时使用的提交版本，标量。
        dtype：torch.dtype，编码计算精度，标量。
        max_length：int，编码长度上限，标量。
    返回值:
        dict[str, Any]：仅包含基础类型的模型追溯信息，字典标量。
    """
    fingerprints = {}
    for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        path = model_path / name
        if path.is_file():
            fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "model_path": str(model_path),
        "revision": revision or "local",
        "config_fingerprints": fingerprints,
        "dtype": str(dtype),
        "max_length": max_length,
    }


# 独立执行缓存生成，用户和服务之间共享相同描述的编码结果。
def main() -> None:
    """
    功能描述:
        读取全部元数据、去重英文描述、加载本地模型并原子写入缓存。
    参数:
        无显式参数；配置由 parse_args 读取，元数据分别为 [U, 9] 和 [S, 9]。
    返回值:
        None：成功后在数据集目录生成包含两侧 [N, 2560] 向量的缓存文件。
    """
    # 在加载模型之前完成文件检查，避免无效输入占用设备资源。
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在：{args.model_path}，请先下载 {ENCODER_ID}。")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"缓存已存在：{args.output}；重建时请添加 --overwrite。")
    user_rows = read_metadata_file(args.data_dir / "data" / "userlist_grid.txt")
    service_rows = read_metadata_file(args.data_dir / "data" / "wslist_grid.txt")
    rows = user_rows + service_rows
    records = [build_description(row, args.ip_octets) for row in rows]
    descriptions = sorted({text for text, mask in records if any(mask)})
    description_ids = {text: index for index, text in enumerate(descriptions)}
    print(f"用户 {len(user_rows)} 行，服务 {len(service_rows)} 行，有效描述去重后 {len(descriptions)} 条。", flush=True)

    # 所有字段缺失的实体保留零向量，不浪费模型编码计算。
    device = torch.device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    embeddings = torch.zeros(len(rows), TEXT_DIM, dtype=torch.float32)
    if descriptions:
        print(f"从本地目录加载 {ENCODER_ID}，设备 {device}，精度 {dtype}。", flush=True)
        model, tokenizer = load_encoder(args.model_path, device, dtype)
        encoded = encode_descriptions(descriptions, model, tokenizer, device, args.batch_size, args.max_length)
        active_rows = [index for index, (_, mask) in enumerate(records) if any(mask)]
        source_ids = [description_ids[records[index][0]] for index in active_rows]
        embeddings[active_rows] = encoded[source_ids]

    # 缓存全部源实体，训练端按自己的实体配置校验并选择所需子集。
    user_count = len(user_rows)
    cache = make_semantic_cache(
        user_rows,
        service_rows,
        embeddings[:user_count],
        embeddings[user_count:],
        ip_octets=args.ip_octets,
        encoder_metadata=encoder_provenance(args.model_path, args.model_revision, dtype, args.max_length),
    )
    save_semantic_cache(cache, args.output, overwrite=args.overwrite)
    print(f"语义缓存已保存到 {args.output}，向量维度为 {TEXT_DIM}。", flush=True)


# 只有显式执行本脚本时才加载模型或写入缓存。
if __name__ == "__main__":
    main()
