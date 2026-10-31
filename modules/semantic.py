from pathlib import Path

import torch
from torch import nn

from config import Config
from modules.fusion import TokenFusion
from modules.semantic_feature import SemanticFeature
from modules.semantic_retrieval import RETRIEVAL_STAT_NAMES, SemanticRetrieval, SemanticRetrievalResult
from modules.semantic_value import SemanticValue
from util.tensor_utils import as_index


# 组织 Qwen 邻居评分、训练观测检索和残差统计，输出两条检索路径的语义 Token。
class Semantic_Feature(nn.Module):
    """
    功能描述:
        Qwen 负责选择可参考的实体，检索模块读取其他配对在相近时间的训练观测，
        将相对自身留一法中位数的变化量和可靠性统计组合为两个动态语义 Token。
    参数:
        cfg：Config，文本、经纬度、检索及注意力配置，标量。
    返回值:
        Semantic_Feature：前向返回 Token [B, 2, E] 和支持有效性 [B, 2]。
    """

    # 保留经纬度编码，建立文本评分、训练支持库及两侧统计投影。
    def __init__(self, cfg: Config) -> None:
        """
        功能描述:
            两条检索路径各输出一个 Token，继续交给现有 Predictor，不直接叠加到最终预测。
        参数:
            cfg：Config，表示维度 E、坐标维度 D 和检索配置，标量。
        返回值:
            None：完成子模块和运行时状态初始化。
        """
        super().__init__()
        self.cfg = cfg
        self.value = SemanticValue(cfg)
        self.feature = SemanticFeature(cfg)
        self.retrieval = SemanticRetrieval(cfg)
        input_dim = cfg.semantic_dim + len(RETRIEVAL_STAT_NAMES)
        self.user_projection = nn.Linear(input_dim, cfg.e)
        self.service_projection = nn.Linear(input_dim, cfg.e)
        self.fusion = TokenFusion(cfg, 2)
        self.metadata_ready = False
        self.cache_metadata: dict = {}

    # 分别恢复 Qwen 关系及原有经纬度缓存，不调用文本编码器。
    def build_metadata(
        self, user_text: list[list[str]], service_text: list[list[str]], cache_path: str | Path | None = None,
    ) -> None:
        """
        功能描述:
            校验实体数量并加载原有离线缓存，训练观测缓存独立于元数据的加载顺序。
        参数:
            user_text：list[list[str]]，用户九列元数据，形状为 [U, 9]。
            service_text：list[list[str]]，服务九列元数据，形状为 [S, 9]。
            cache_path：str | Path | None，显式 Qwen 缓存路径，标量。
        返回值:
            None：恢复文本、坐标及冻结相似度。
        """
        self.metadata_ready = False
        if len(user_text) != self.cfg.user_num:
            raise ValueError(f"user_text 有 {len(user_text)} 行，与 cfg.user_num={self.cfg.user_num} 不一致。")
        if len(service_text) != self.cfg.service_num:
            raise ValueError(f"service_text 有 {len(service_text)} 行，与 cfg.service_num={self.cfg.service_num} 不一致。")
        self.feature.build_metadata(user_text, service_text, cache_path)
        self.value.build_metadata(user_text, service_text)
        self.cache_metadata = self.feature.cache_metadata
        self.metadata_ready = True

    # 使用与中位数底座相同的训练划分，独立准备跨实体支持库。
    def build_observations(
        self, train_info: torch.Tensor, train_rt: torch.Tensor,
        anchor_loo: torch.Tensor, pair_counts: torch.Tensor,
    ) -> int:
        """
        功能描述:
            转交检索子模块缓存逐观测残差，关闭 Local 时仍可独立工作。
        参数:
            train_info：torch.Tensor，时间、用户、服务索引，形状为 [N, 3]。
            train_rt：torch.Tensor，训练响应时间，形状为 [N] 或 [N, 1]。
            anchor_loo：torch.Tensor，训练留一法中位数，形状为 [U, S, T]。
            pair_counts：torch.Tensor，训练配对计数，形状为 [U, S]。
        返回值:
            int：有效支持观测数量。
        """
        return self.retrieval.build_observations(train_info, train_rt, anchor_loo, pair_counts)

    # 返回文本评分和统计投影的线性权重，保持原有正则分组方式。
    def regularized_parameters(self) -> tuple[nn.Parameter, ...]:
        """
        功能描述:
            对文本 Adapter 和两侧统计投影施加语义专属权重衰减。
        参数:
            无显式参数；读取当前模块可训练投影。
        返回值:
            tuple[nn.Parameter, ...]：四个线性权重，不包含偏置、归一化、坐标和内部注意力。
        """
        text_weights = tuple(layer.weight for layer in self.feature.text_projection if isinstance(layer, nn.Linear))
        return text_weights + (self.user_projection.weight, self.service_projection.weight)

    # 提供可直接检查的两侧残差和可靠性，不经过 Token 投影或内部融合。
    def retrieve(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor,
        *, indices_validated: bool = False,
    ) -> SemanticRetrievalResult:
        """
        功能描述:
            对有训练观测的文本邻居进行时间相关检索，输出前三条改进对应的结构化结果。
        参数:
            user_indices、service_indices、time_indices：torch.Tensor，当前查询索引，各为 [B]。
        返回值:
            SemanticRetrievalResult：残差 [B, 2]、可靠性统计 [B, 2, F] 和有效性 [B, 2]。
        """
        if not self.metadata_ready:
            raise RuntimeError("语义元数据尚未就绪，请先调用 build_metadata 加载离线缓存。")
        if not self.retrieval.observations_ready:
            raise RuntimeError("语义训练支持库尚未就绪，请先调用 build_anchor 构建训练观测缓存。")
        user, service, time = self.retrieval.prepare_indices(
            user_indices, service_indices, time_indices, check_bounds=not indices_validated,
        )
        user_scores = self.feature(user, "user")
        service_scores = self.feature(service, "service")
        return self.retrieval(user, service, time, user_scores, service_scores, indices_prepared=True)

    # 将坐标表示与两路检索统计组合为动态语义 Token，并屏蔽无支持路径。
    def forward(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor,
        *, indices_validated: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        功能描述:
            在现有三模块结构中，用用户邻居和服务邻居两个 Token 替换静态实体语义 Token。
            无支持观测时即使坐标有效也输出无效零 Token，避免投影偏置形成伪检索证据。
        参数:
            user_indices、service_indices、time_indices：torch.Tensor，当前查询索引，各为 [B]。
        返回值:
            tuple[torch.Tensor, torch.Tensor]：内部融合后的 Token [B, 2, E] 和支持有效性 [B, 2]。
        """
        device = self.value.user_geo.device
        user, service, time = (as_index(index, device) for index in (user_indices, service_indices, time_indices))
        result = self.retrieve(user, service, time, indices_validated=indices_validated)
        user_geo, service_geo, _, _ = self.value(user, service)
        user_token = self.user_projection(torch.cat([user_geo, result.statistics[:, 0]], dim=-1))
        service_token = self.service_projection(torch.cat([service_geo, result.statistics[:, 1]], dim=-1))
        tokens = torch.stack([user_token, service_token], dim=1)
        return self.fusion(tokens, result.valid_mask), result.valid_mask
