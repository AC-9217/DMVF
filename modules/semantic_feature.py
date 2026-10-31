import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from data.semantic_cache import TEXT_DIM, load_semantic_cache


# 使用冻结 Qwen 相似度选择近邻，并通过共享 Adapter 校准观测聚合分数。
class SemanticFeature(nn.Module):
    """
    功能描述:
        从离线缓存读取冻结文本向量，保留 Qwen 余弦相似度作为近邻选择依据，
        用共享 Adapter 提供小幅可训练评分修正。
    参数:
        cfg：Config，缓存、输出维度、隐藏层和 Dropout 配置，标量。
    返回值:
        SemanticFeature：按用户或服务输出 [B, N] 邻居评分的文本模块。
    """

    # 初始化共享文本 Adapter、冻结向量及整句有效性缓存。
    def __init__(self, cfg: Config) -> None:
        """
        功能描述:
            沿用现有缓存协议，用两层投影、Dropout 和 LayerNorm 适配联合文本表示。
        参数:
            cfg：Config，文本语义配置，标量。
        返回值:
            None：完成网络及运行时缓存初始化。
        """
        super().__init__()
        self.cfg = cfg
        self.ip_octets = int(cfg.semantic_ip_octets)
        if not 1 <= self.ip_octets <= 4:
            raise ValueError("semantic_ip_octets 必须在 1 至 4 之间。")
        if cfg.semantic_text_hidden < 1 or cfg.e < 1:
            raise ValueError("semantic_text_hidden 和 e 必须为正整数。")
        if not math.isfinite(cfg.semantic_text_dropout) or not 0 <= cfg.semantic_text_dropout < 1:
            raise ValueError("semantic_text_dropout 必须为零至一之间的有限数值，且不能等于一。")
        if not math.isfinite(cfg.semantic_temperature) or cfg.semantic_temperature <= 0:
            raise ValueError("semantic_temperature 必须为正的有限数值。")
        # 联合向量保持整体语义，不再人为划分为五个字段。
        self.text_projection = nn.Sequential(
            nn.Linear(TEXT_DIM, cfg.semantic_text_hidden),
            nn.GELU(),
            nn.Dropout(cfg.semantic_text_dropout),
            nn.Linear(cfg.semantic_text_hidden, cfg.e),
            nn.LayerNorm(cfg.e),
        )

        # 仅保留整句有效性；原始五字段掩码继续由缓存协议校验。
        for label, count in (("user", cfg.user_num), ("service", cfg.service_num)):
            self.register_buffer(f"{label}_text_embeddings", torch.zeros(count, TEXT_DIM), persistent=False)
            self.register_buffer(f"{label}_similarities", torch.zeros(count, count), persistent=False)
            self.register_buffer(
                f"{label}_text_valid_mask", torch.zeros(count, dtype=torch.bool), persistent=False
            )
        self.cache_metadata: dict = {}

    # 使用既有校验协议加载实体对齐后的离线向量。
    def build_metadata(
        self, user_text: list[list[str]], service_text: list[list[str]], cache_path: str | Path | None = None
    ) -> None:
        """
        功能描述:
            校验缓存指纹、模型、字段和实体映射，恢复冻结文本及有效性掩码。
        参数:
            user_text：list[list[str]]，用户元数据，形状为 [U, 9]。
            service_text：list[list[str]]，服务元数据，形状为 [S, 9]。
            cache_path：str | Path | None，显式缓存路径，标量。
        返回值:
            None：更新冻结缓存和缓存版本信息。
        """
        cache = load_semantic_cache(
            cache_path if cache_path is not None else self.cfg.resolved_semantic_cache_path,
            user_text, service_text, self.ip_octets,
        )
        for label in ("user", "service"):
            section = cache["users" if label == "user" else "services"]
            target = getattr(self, f"{label}_text_embeddings")
            target.copy_(section["text_embeddings"])
            mask = getattr(self, f"{label}_text_valid_mask")
            mask.copy_(section["field_valid_mask"].any(dim=1))
            # 冻结相似度直接来自完整 Qwen 向量，训练中的 Adapter 不改变近邻候选边界。
            normalized = F.normalize(target, dim=-1)
            similarities = getattr(self, f"{label}_similarities")
            similarities.copy_((normalized @ normalized.T).clamp(-1.0, 1.0))
        self.cache_metadata = cache["metadata"]

    # 适配完整联合向量，并在投影和归一化后清零整句缺失的实体。
    def encode_text(self, text_embeddings: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        功能描述:
            将冻结联合向量映射到模型维度，整句缺失时屏蔽投影与归一化层的偏置。
        参数:
            text_embeddings：torch.Tensor，冻结文本向量，形状为 [B, 2560]。
            valid_mask：torch.Tensor，整句布尔有效性，形状为 [B]。
        返回值:
            torch.Tensor：完整实体文本表示，形状为 [B, E]。
        """
        projected = self.text_projection(text_embeddings)
        return projected.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    # 对一侧全部实体评分，供检索模块在有观测的实体中选择近邻。
    def forward(
        self, indices: torch.Tensor, entity_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        功能描述:
            在 Qwen 相似度上叠加幅度为 0.1 的 Adapter 余弦修正，之后按温度缩放。
            整句文本缺失和实体自身均不提供邻居证据。
        参数:
            indices：torch.Tensor，当前侧查询实体索引，形状为 [B]。
            entity_type：str，实体侧名称，仅支持 user 或 service，标量。
        返回值:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]：可训练分数、冻结余弦相似度、布尔有效性，各为 [B, N]。
        """
        if entity_type not in ("user", "service"):
            raise ValueError("entity_type 必须为 user 或 service。")
        embeddings = getattr(self, f"{entity_type}_text_embeddings")
        valid = getattr(self, f"{entity_type}_text_valid_mask")
        similarities = getattr(self, f"{entity_type}_similarities")[indices]
        adapted = F.normalize(self.encode_text(embeddings, valid), dim=-1)
        correction = adapted[indices] @ adapted.T
        scores = (similarities + 0.1 * correction) / self.cfg.semantic_temperature
        neighbors = torch.arange(embeddings.shape[0], device=embeddings.device)
        eligible = valid[indices].unsqueeze(1) & valid.unsqueeze(0) & (indices.unsqueeze(1) != neighbors)
        return scores, similarities, eligible
