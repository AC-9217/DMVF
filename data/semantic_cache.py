"""管理全英文语义描述和离线文本缓存，仅依赖标准库与 PyTorch。"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


# 固定输入列序、文本字段顺序与缓存协议。
COLUMN_ID, COLUMN_IP, COLUMN_COUNTRY, COLUMN_STATE, COLUMN_CITY, COLUMN_LAT, COLUMN_LON, COLUMN_ASN, COLUMN_GRID = range(9)
COLUMN_NUM = 9
NONE_TOKEN = "None"
TEXT_FIELD_NAMES = ("ip", "country", "state", "city", "asn")
FIELD_NAMES = ("id", "ip", "country", "state", "city", "geo", "asn")
ENCODER_ID = "Qwen/Qwen3-Embedding-4B"
TEXT_DIM = 2560
CACHE_FILENAME = "semantic_qwen3_4b_cache.pt"
CACHE_VERSION = 1
TEMPLATE_VERSION = "english-v1"
TEXT_INSTRUCTION = "Encode the geographic location and network attributes of an entity."


# 清洗单个文本字段，保持原有空串和缺失占位符语义。
def clean_value(row: list[str], column: int) -> str | None:
    """
    功能描述:
        去除字段首尾空白，将空串和字符串 None 识别为缺失。
    参数:
        row：list[str]，一行元数据，形状为 [9]。
        column：int，需要读取的列号，标量。
    返回值:
        str | None：清洗后的字段值，标量；缺失时为 None。
    """
    value = row[column].strip() if column < len(row) else ""
    return None if value in ("", NONE_TOKEN) else value


## 校验列数和ID匹配
def validate_rows(rows: list[list[str]], label: str) -> None:
    """
    功能描述:
        要求元数据非空、每行九列，且第零列严格等于从零开始的行号。
    参数:
        rows：list[list[str]]，用户或服务元数据，形状为 [N, 9]。
        label：str，错误消息中的数据名称，标量。
    返回值:
        None：校验失败时抛出 ValueError。
    """
    if not rows:
        raise ValueError(f"{label}元数据不能为空。")
    for index, row in enumerate(rows):
        if len(row) != COLUMN_NUM:
            raise ValueError(f"{label}第 {index} 行有 {len(row)} 列，应为 {COLUMN_NUM} 列。")
        if row[COLUMN_ID] != str(index):
            raise ValueError(f"{label}第 {index} 行的 ID={row[COLUMN_ID]!r}，必须等于行号。")


# 读取独立预处理脚本使用的完整元数据文件。
def read_metadata_file(path: str | Path) -> list[list[str]]:
    """
    功能描述:
        读取制表符分隔的九列元数据，跳过空行并校验实体编号。
    参数:
        path：str | Path，元数据文件路径，标量。
    返回值:
        list[list[str]]：按实体编号排列的全部元数据，形状为 [N, 9]。
    """
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        rows = [[value.strip() for value in line.rstrip("\r\n").split("\t")] for line in source if line.strip()]
    validate_rows(rows, str(path))
    return rows


# 生成只含英文标签和英文地名的确定性联合描述。
def build_description(row: list[str], ip_octets: int = 2) -> tuple[str, list[bool]]:
    """
    功能描述:
        校验 IP 和 ASN，将五字段拼成英文指令与短句，并返回各字段有效性。
        非英文字符需要先在源数据中规范为英文名称，函数不会猜测翻译。
    参数:
        row：list[str]，九列元数据，形状为 [9]。
        ip_octets：int，IP 前缀保留的段数，标量，范围为一至四。
    返回值:
        tuple[str, list[bool]]：英文描述标量，以及按文本字段顺序排列的 [5] 有效性标志。
    """
    # 沿用 IP 前缀和 ASN 的规范化规则。
    if not 1 <= ip_octets <= 4:
        raise ValueError("IP 前缀段数必须在 1 至 4 之间。")
    ip = clean_value(row, COLUMN_IP)
    octets = ip.split(".") if ip is not None else []
    valid_ip = len(octets) == 4 and all(part.isascii() and part.isdigit() and 0 <= int(part) <= 255 for part in octets)
    prefix = ".".join(octets[:ip_octets]) if valid_ip else None
    asn = clean_value(row, COLUMN_ASN)
    asn = str(int(asn)) if asn is not None and asn.isascii() and asn.isdigit() else None

    # 固定地名空白格式，禁止在全英文模板中混入未规范化字符。
    locations = []
    for column in (COLUMN_COUNTRY, COLUMN_STATE, COLUMN_CITY):
        value = clean_value(row, column)
        if value is not None:
            value = " ".join(value.split())
            if not value.isascii():
                raise ValueError(f"实体 ID={row[COLUMN_ID]} 的地名 {value!r} 含非英文字符，请先规范为英文名称。")
        locations.append(value)

    # 仅使用五个文本字段，实体 ID 和经纬度不写入描述。
    values = [prefix, *locations, asn]
    mask = [value is not None for value in values]
    ip_text, country, state, city, asn_text = [value if value is not None else "Unknown" for value in values]
    sentence = (
        f"IP prefix: {ip_text}; country: {country}; state: {state}; "
        f"city: {city}; autonomous system number: {asn_text}."
    )
    return f"Instruct: {TEXT_INSTRUCTION}\nQuery: {sentence}", mask


# 为原始行生成确定性指纹，支持缓存全部实体而训练仅取部分实体。
def rows_fingerprint(rows: list[list[str]]) -> str:
    """
    功能描述:
        对规范序列化后的元数据计算指纹，用于检测源数据变化。
    参数:
        rows：list[list[str]]，任意数量的元数据行，形状为 [N, 9]。
    返回值:
        str：十六进制 SHA256 指纹，标量。
    """
    serialized = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# 构建离线脚本和训练端共同遵守的缓存元信息。
def cache_metadata(ip_octets: int) -> dict[str, Any]:
    """
    功能描述:
        返回模型、模板、字段顺序和编码方式的固定协议。
    参数:
        ip_octets：int，IP 前缀保留的段数，标量。
    返回值:
        dict[str, Any]：仅包含基础类型的缓存元信息，字典标量。
    """
    return {
        "cache_version": CACHE_VERSION,
        "encoder_id": ENCODER_ID,
        "embedding_dim": TEXT_DIM,
        "field_names": list(TEXT_FIELD_NAMES),
        "template_version": TEMPLATE_VERSION,
        "instruction": TEXT_INSTRUCTION,
        "pooling": "last_valid_token",
        "normalization": "l2",
        "ip_octets": ip_octets,
    }


# 将一侧实体的离线向量整理为可独立校验的缓存分区。
def make_cache_section(rows: list[list[str]], embeddings: torch.Tensor, ip_octets: int) -> dict[str, Any]:
    """
    功能描述:
        为每个实体记录 ID、冻结向量、字段掩码、原始行指纹和英文描述。
    参数:
        rows：list[list[str]]，按 ID 排列的元数据，形状为 [N, 9]。
        embeddings：torch.Tensor，已归一化的文本向量，形状为 [N, 2560]。
        ip_octets：int，IP 前缀段数，标量。
    返回值:
        dict[str, Any]：可由安全权重加载模式读取的缓存分区，字典标量。
    """
    validate_rows(rows, "缓存")
    descriptions, masks = zip(*(build_description(row, ip_octets) for row in rows))
    return {
        "ids": torch.arange(len(rows), dtype=torch.long),
        "text_embeddings": embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        "field_valid_mask": torch.tensor(masks, dtype=torch.bool),
        "row_fingerprints": [rows_fingerprint([row]) for row in rows],
        "descriptions": list(descriptions),
    }


# 校验缓存分区并按当前模型所需的实体 ID 重排。
def select_cache_section(section: dict[str, Any], rows: list[list[str]], ip_octets: int, label: str) -> dict[str, torch.Tensor]:
    """
    功能描述:
        检查向量、掩码、ID 和源数据指纹，允许缓存包含多于当前配置的实体。
    参数:
        section：dict[str, Any]，包含 N 个缓存实体的分区，字典标量。
        rows：list[list[str]]，当前模型需要的 M 行元数据，形状为 [M, 9]。
        ip_octets：int，当前配置的 IP 前缀段数，标量。
        label：str，错误消息中的分区名称，标量。
    返回值:
        dict[str, torch.Tensor]：按目标 ID 排序的 [M, 2560] 向量及 [M, 5] 布尔掩码。
    """
    # 检查张量类型和形状，拒绝隐式截断或把浮点 ID 转成整数。
    ids = section["ids"]
    embeddings = section["text_embeddings"]
    masks = section["field_valid_mask"]
    fingerprints = section["row_fingerprints"]
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 1:
        raise ValueError(f"{label}缓存 ID 必须是一维整数张量。")
    count = ids.numel()
    if count == 0 or (ids < 0).any() or ids.unique().numel() != count:
        raise ValueError(f"{label}缓存 ID 必须非空、非负且互不重复。")
    if not isinstance(embeddings, torch.Tensor) or not embeddings.is_floating_point() or embeddings.shape != (count, TEXT_DIM):
        raise ValueError(f"{label}缓存向量形状必须为 [{count}, {TEXT_DIM}]。")
    if not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool or masks.shape != (count, len(TEXT_FIELD_NAMES)):
        raise ValueError(f"{label}缓存掩码必须为 [{count}, {len(TEXT_FIELD_NAMES)}] 的布尔张量。")
    if not isinstance(fingerprints, list) or len(fingerprints) != count:
        raise ValueError(f"{label}缓存缺少逐行源数据指纹。")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{label}缓存向量含非有限数值。")
    norms = torch.linalg.vector_norm(embeddings.float(), dim=1)
    active = masks.any(dim=1)
    if not torch.allclose(norms[active], torch.ones_like(norms[active]), atol=1e-3, rtol=1e-3):
        raise ValueError(f"{label}有效文本向量必须经过 L2 归一化。")

    # 按 ID 查找而非直接切片，确保分区行序变化不会改变实体对应关系。
    sorted_ids, order = ids.sort()
    required = torch.arange(len(rows), dtype=torch.long)
    positions = torch.searchsorted(sorted_ids, required)
    if (positions >= count).any() or not torch.equal(sorted_ids[positions], required):
        raise ValueError(f"{label}缓存没有覆盖当前模型需要的全部实体 ID。")
    indices = order[positions]
    expected_masks = torch.tensor([build_description(row, ip_octets)[1] for row in rows], dtype=torch.bool)
    if not torch.equal(masks[indices], expected_masks):
        raise ValueError(f"{label}缓存字段有效性与当前元数据不一致。")
    for row, index in zip(rows, indices.tolist()):
        if fingerprints[index] != rows_fingerprint([row]):
            raise ValueError(f"{label}实体 ID={row[COLUMN_ID]} 的元数据已变化，缓存需要重新生成。")
    return {
        "text_embeddings": embeddings[indices].float().contiguous(),
        "field_valid_mask": masks[indices].contiguous(),
    }


# 组合用户和服务向量，生成完整缓存内容。
def make_semantic_cache(
    user_rows: list[list[str]],
    service_rows: list[list[str]],
    user_embeddings: torch.Tensor,
    service_embeddings: torch.Tensor,
    ip_octets: int = 2,
    encoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    功能描述:
        组装并验证两侧实体的离线缓存，不调用文本编码器。
    参数:
        user_rows：list[list[str]]，用户元数据，形状为 [U, 9]。
        service_rows：list[list[str]]，服务元数据，形状为 [S, 9]。
        user_embeddings：torch.Tensor，用户文本向量，形状为 [U, 2560]。
        service_embeddings：torch.Tensor，服务文本向量，形状为 [S, 2560]。
        ip_octets：int，IP 前缀段数，标量。
        encoder_metadata：dict[str, Any] | None，模型版本等附加信息，字典标量。
    返回值:
        dict[str, Any]：包含元信息、用户分区和服务分区的完整缓存。
    """
    metadata = cache_metadata(ip_octets)
    metadata["encoder"] = encoder_metadata or {}
    metadata["source_fingerprints"] = {
        "users": rows_fingerprint(user_rows),
        "services": rows_fingerprint(service_rows),
    }
    cache = {
        "metadata": metadata,
        "users": make_cache_section(user_rows, user_embeddings, ip_octets),
        "services": make_cache_section(service_rows, service_embeddings, ip_octets),
    }
    select_cache_section(cache["users"], user_rows, ip_octets, "用户")
    select_cache_section(cache["services"], service_rows, ip_octets, "服务")
    return cache


# 原子写入缓存，编码失败时不留下半个可加载文件。
def save_semantic_cache(cache: dict[str, Any], path: str | Path, overwrite: bool = False) -> None:
    """
    功能描述:
        先写同目录临时文件，再替换目标文件，默认拒绝覆盖已有缓存。
    参数:
        cache：dict[str, Any]，完整缓存内容，字典标量。
        path：str | Path，输出文件路径，标量。
        overwrite：bool，是否允许覆盖已有缓存，标量。
    返回值:
        None：成功后目标路径包含完整缓存。
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"缓存已存在：{path}；需要重建时请显式添加 --overwrite。")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(cache, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


# 训练端仅加载缓存并校验协议，完全不依赖文本编码库。
def load_semantic_cache(
    path: str | Path, user_rows: list[list[str]], service_rows: list[list[str]], ip_octets: int
) -> dict[str, Any]:
    """
    功能描述:
        安全读取缓存，校验模型与模板协议，并选择当前配置要求的实体。
        缓存缺失或失效时仅报错，不联网、不调用编码器、不生成缓存。
    参数:
        path：str | Path，缓存文件路径，标量。
        user_rows：list[list[str]]，当前用户元数据，形状为 [U, 9]。
        service_rows：list[list[str]]，当前服务元数据，形状为 [S, 9]。
        ip_octets：int，当前 IP 前缀配置，标量。
    返回值:
        dict[str, Any]：缓存元信息与按实体 ID 对齐的两个张量分区。
    """
    # 在读取大型缓存之前检查原始元数据和文件是否就绪。
    validate_rows(user_rows, "用户")
    validate_rows(service_rows, "服务")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"语义缓存不存在：{path}。请先独立运行 dataset/build_semantic_cache.py 生成缓存。"
        )

    # 使用仅权重加载模式，缓存只包含张量和基础类型。
    cache = torch.load(path, map_location="cpu", weights_only=True)
    try:
        metadata = cache["metadata"]
        for key, expected in cache_metadata(ip_octets).items():
            if metadata.get(key) != expected:
                raise ValueError(f"缓存协议 {key}={metadata.get(key)!r}，当前要求 {expected!r}。")
        return {
            "metadata": metadata,
            "users": select_cache_section(cache["users"], user_rows, ip_octets, "用户"),
            "services": select_cache_section(cache["services"], service_rows, ip_octets, "服务"),
        }
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(
            f"语义缓存校验失败：{error} 请重新独立运行 dataset/build_semantic_cache.py --overwrite。"
        ) from error
