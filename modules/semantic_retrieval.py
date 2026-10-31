import math
from dataclasses import dataclass

import torch
from torch import nn

from config import Config
from util.tensor_utils import as_index


# 固定两条检索路径的统计顺序，首列始终保留原始秒数单位的变化量。
RETRIEVAL_STAT_NAMES = (
    "residual", "residual_std", "log_neighbors", "log_observations",
    "similarity", "time_distance", "same_time_weight", "log_history",
)


# 保存用户邻居、服务邻居两条路径的变化量、可靠性和有效性。
@dataclass(frozen=True)
class SemanticRetrievalResult:
    """
    功能描述:
        组织两条检索路径的输出，供语义 Token 投影及后续预测头使用。
    参数:
        statistics：torch.Tensor，按 RETRIEVAL_STAT_NAMES 排列的统计，形状为 [B, 2, F]。
        valid_mask：torch.Tensor，两条路径是否有有效支持观测，布尔形状为 [B, 2]。
    返回值:
        SemanticRetrievalResult：路径顺序固定为用户邻居、服务邻居。
    """

    statistics: torch.Tensor
    valid_mask: torch.Tensor

    # 从统计首列读取两条路径的加权 QoS 变化量。
    @property
    def residuals(self) -> torch.Tensor:
        """
        功能描述:
            提供未直接加入最终预测的两侧检索残差。
        参数:
            无显式参数；读取 statistics，形状为 [B, 2, F]。
        返回值:
            torch.Tensor：单位为秒的变化量，形状为 [B, 2]，无支持路径为零。
        """
        return self.statistics[..., 0]


# 只缓存训练观测，并从其他配对检索相近时间的留一法残差。
class SemanticRetrieval(nn.Module):
    """
    功能描述:
        用户路径查询其他用户对目标服务的观测，服务路径查询目标用户对其他服务的观测。
        以 Qwen 相似度选择有观测的邻居，再按可训练分数和时间距离聚合残差。
    参数:
        cfg：Config，实体数量、时间范围、邻居数量及时间衰减配置，标量。
    返回值:
        SemanticRetrieval：前向输出 SemanticRetrievalResult 的无参数检索模块。
    """

    # 初始化与 Local 独立的训练残差、支持掩码和历史观测数缓存。
    def __init__(self, cfg: Config) -> None:
        """
        功能描述:
            配置包含当前时间片的检索窗口，缓存不写入模型权重文件。
        参数:
            cfg：Config，用户数 U、服务数 S、时间片数 T 及检索超参数，标量。
        返回值:
            None：创建 [U, S, T] 的残差和支持掩码及 [U, S] 的历史计数。
        """
        super().__init__()
        if cfg.semantic_neighbors < 1:
            raise ValueError("semantic_neighbors 必须为正整数。")
        if cfg.semantic_time_radius < 0:
            raise ValueError("semantic_time_radius 必须为非负整数。")
        if not math.isfinite(cfg.semantic_time_decay) or cfg.semantic_time_decay < 0:
            raise ValueError("semantic_time_decay 必须为非负有限数值。")
        self.cfg = cfg
        shape = (cfg.user_num, cfg.service_num, cfg.time_num)
        self.register_buffer("residual_matrix", torch.zeros(shape), persistent=False)
        self.register_buffer("support_mask", torch.zeros(shape, dtype=torch.bool), persistent=False)
        self.register_buffer("pair_counts", torch.zeros(shape[:2]), persistent=False)
        self.register_buffer(
            "offsets", torch.arange(-cfg.semantic_time_radius, cfg.semantic_time_radius + 1), persistent=False,
        )
        self.observations_ready = False

    # 仅用训练样本及其逐观测留一法中位数构建支持库。
    @torch.no_grad()
    def build_observations(
        self, train_info: torch.Tensor, train_rt: torch.Tensor,
        anchor_loo: torch.Tensor, pair_counts: torch.Tensor,
    ) -> int:
        """
        功能描述:
            缓存 y 减去同配对其他时间观测中位数的残差；单观测配对不使用全局兜底参与检索。
            支持库与训练、推理状态无关，均只包含调用方提供的训练观测。
        参数:
            train_info：torch.Tensor，整数索引，形状为 [N, 3]，列顺序为时间、用户、服务。
            train_rt：torch.Tensor，正响应时间，形状为 [N] 或 [N, 1]。
            anchor_loo：torch.Tensor，训练逐观测留一法中位数，形状为 [U, S, T]。
            pair_counts：torch.Tensor，训练配对观测数，形状为 [U, S]。
        返回值:
            int：具有同配对其他历史观测的有效支持记录数量。
        """
        self.observations_ready = False
        device = self.residual_matrix.device
        if train_info.ndim != 2 or train_info.shape[1] != 3 or train_info.dtype != torch.long:
            raise ValueError("语义训练索引必须是形状 [N, 3] 的 long 张量。")
        info = train_info.detach().to(device)
        rt = train_rt.detach().to(device=device, dtype=torch.float32).reshape(-1)
        if info.shape[0] == 0 or rt.numel() != info.shape[0]:
            raise ValueError("语义支持库需要非空且索引与响应时间数量一致的训练集。")
        if not torch.isfinite(rt).all() or (rt <= 0).any():
            raise ValueError("语义支持观测必须是正的有限响应时间。")
        limits = info.new_tensor([self.cfg.time_num, self.cfg.user_num, self.cfg.service_num])
        if ((info < 0) | (info >= limits)).any():
            raise ValueError("语义训练索引超出时间、用户或服务范围。")
        if anchor_loo.shape != self.residual_matrix.shape or pair_counts.shape != self.pair_counts.shape:
            raise ValueError("语义留一法中位数或配对计数的形状与配置不一致。")

        # 校验支持库与中位数统计来自同一套无重复训练观测。
        time, user, service = info.unbind(dim=1)
        pair = user * self.cfg.service_num + service
        cell = pair * self.cfg.time_num + time
        if torch.unique(cell).numel() != info.shape[0]:
            raise ValueError("语义支持库不接受重复的时间、用户、服务索引。")
        counts = pair_counts.detach().to(device=device, dtype=torch.float32)
        actual_counts = torch.bincount(pair, minlength=counts.numel()).reshape_as(counts)
        if not torch.equal(counts, actual_counts.to(counts.dtype)):
            raise ValueError("语义配对计数与传入训练观测不一致。")
        usable = counts[user, service] >= 2
        support_user, support_service, support_time = user[usable], service[usable], time[usable]
        centers = anchor_loo.detach().to(device)[support_user, support_service, support_time]
        if not torch.isfinite(centers).all() or (centers <= 0).any():
            raise ValueError("有效语义支持记录的留一法中位数必须为正的有限数值。")

        # 清理旧划分残留，未观测位置和单观测配对均保持零值、无效掩码。
        self.residual_matrix.zero_()
        self.support_mask.zero_()
        self.pair_counts.copy_(counts)
        self.residual_matrix[support_user, support_service, support_time] = rt[usable] - centers
        self.support_mask[support_user, support_service, support_time] = True
        self.observations_ready = True
        return int(usable.sum().item())

    # 统一查询索引的设备、形状及合法范围。
    def prepare_indices(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor,
        *, check_bounds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        功能描述:
            将查询索引搬到支持库所在设备，拒绝负索引及不匹配的 batch。
        参数:
            user_indices、service_indices、time_indices：torch.Tensor，一维整数索引，各为 [B]。
        返回值:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]：设备对齐的 long 用户、服务、时间索引，各为 [B]。
        """
        indices = tuple(as_index(index, self.residual_matrix.device) for index in (user_indices, service_indices, time_indices))
        if indices[0].ndim != 1 or any(index.shape != indices[0].shape for index in indices[1:]):
            raise ValueError("语义查询的用户、服务和时间必须是相同长度的一维张量。")
        if check_bounds:
            for name, index, limit in zip(("用户", "服务", "时间"), indices, (self.cfg.user_num, self.cfg.service_num, self.cfg.time_num)):
                if index.numel() and ((index < 0) | (index >= limit)).any():
                    raise ValueError(f"语义{name}索引超出 [0, {limit - 1}] 范围。")
        return indices

    # 在存在支持观测的实体中选择文本近邻，并聚合单侧残差及可靠性。
    def _aggregate_path(
        self, user: torch.Tensor, service: torch.Tensor, time: torch.Tensor,
        scores: torch.Tensor, similarities: torch.Tensor, eligible: torch.Tensor, user_side: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        功能描述:
            查询包含当前时间片的短窗口，排除目标配对；相似度位于第 K 名边界的并列邻居全部保留。
            在有效观测上对评分减去时间距离惩罚执行归一化，避免全空路径产生非数值。
        参数:
            user、service、time：torch.Tensor，已对齐查询索引，各为 [B]。
            scores：torch.Tensor，可训练邻居分数，形状为 [B, N]。
            similarities：torch.Tensor，冻结 Qwen 余弦相似度，形状为 [B, N]。
            eligible：torch.Tensor，文本有效且不是目标实体的布尔标记，形状为 [B, N]。
            user_side：bool，True 查询其他用户，False 查询其他服务，标量。
        返回值:
            tuple[torch.Tensor, torch.Tensor]：统计 [B, F] 和支持有效性 [B]。
        """
        neighbor_count = self.cfg.user_num if user_side else self.cfg.service_num
        expected_shape = (user.numel(), neighbor_count)
        if scores.shape != expected_shape or similarities.shape != expected_shape or eligible.shape != expected_shape:
            raise ValueError("语义邻居分数、相似度及有效性形状与查询不一致。")
        if eligible.dtype != torch.bool:
            raise ValueError("语义邻居有效性必须是布尔张量。")

        # 两条路径均排除目标实体，因此所有时间片的目标配对都不会作为自己的支持记录。
        neighbors = torch.arange(neighbor_count, device=user.device)
        target = user if user_side else service
        eligible = eligible & (neighbors.unsqueeze(0) != target.unsqueeze(1))
        neighbor_time = time[:, None, None] + self.offsets[None, None, :]
        in_range = (neighbor_time >= 0) & (neighbor_time < self.cfg.time_num)
        clipped = neighbor_time.clamp(0, self.cfg.time_num - 1)
        source_user = neighbors[None, :, None] if user_side else user[:, None, None]
        source_service = service[:, None, None] if user_side else neighbors[None, :, None]
        residuals = self.residual_matrix[source_user, source_service, clipped]
        observed = self.support_mask[source_user, source_service, clipped] & in_range & eligible.unsqueeze(-1)

        # 先检查时间窗口有无观测，再按 Qwen 相似度选择近邻；并列保留避免相同描述偏向小编号。
        available = observed.any(dim=-1)
        ranking = similarities.masked_fill(~available, -torch.inf)
        k = min(self.cfg.semantic_neighbors, neighbor_count)
        cutoff = torch.topk(ranking, k, dim=1).values[:, -1:]
        selected = available & (similarities >= cutoff - 1e-6)
        observed = observed & selected.unsqueeze(-1)
        residuals = residuals.masked_fill(~observed, 0.0)

        # 当前时间片没有距离惩罚，越远的记录权重越低；全空路径经掩码得到严格零权重。
        distance = self.offsets.abs().to(scores.dtype)
        logits = scores.unsqueeze(-1) - self.cfg.semantic_time_decay * distance[None, None, :]
        logits = logits.masked_fill(~observed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(logits.flatten(start_dim=1), dim=1).reshape_as(logits)
        weights = weights.masked_fill(~observed, 0.0)
        weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        valid = observed.any(dim=(1, 2))

        # 变化量沿用秒数单位，并输出观测覆盖、相似度、时间距离和支持历史等可靠性统计。
        mean = (weights * residuals).sum(dim=(1, 2))
        variance = (weights * (residuals - mean[:, None, None]).square()).sum(dim=(1, 2))
        std = torch.where(variance > 0, variance.clamp_min(1e-8).sqrt(), torch.zeros_like(variance))
        counts = self.pair_counts[source_user, source_service]
        history = torch.log1p((counts - 1).clamp_min(0))
        statistics = torch.stack([
            mean, std, torch.log1p(selected.sum(dim=1).to(scores.dtype)),
            torch.log1p(observed.sum(dim=(1, 2)).to(scores.dtype)),
            (weights * similarities.unsqueeze(-1)).sum(dim=(1, 2)),
            (weights * distance[None, None, :]).sum(dim=(1, 2)) / max(self.cfg.semantic_time_radius, 1),
            (weights * (distance == 0)[None, None, :]).sum(dim=(1, 2)),
            (weights * history).sum(dim=(1, 2)),
        ], dim=-1)
        return statistics.masked_fill(~valid.unsqueeze(-1), 0.0), valid

    # 同时查询用户邻居与服务邻居，输出两条路径的结构化结果。
    def forward(
        self, user: torch.Tensor, service: torch.Tensor, time: torch.Tensor,
        user_scores: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        service_scores: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *, indices_prepared: bool = False,
    ) -> SemanticRetrievalResult:
        """
        功能描述:
            聚合只来自训练支持库的两侧变化量，缺失路径输出零统计与 False 掩码。
        参数:
            user、service、time：torch.Tensor，查询索引，各为 [B]。
            user_scores：tuple，用户分数、Qwen 相似度与布尔有效性，各为 [B, U]。
            service_scores：tuple，服务分数、Qwen 相似度与布尔有效性，各为 [B, S]。
        返回值:
            SemanticRetrievalResult：统计形状为 [B, 2, F]，掩码形状为 [B, 2]。
        """
        if not self.observations_ready:
            raise RuntimeError("语义训练支持库尚未就绪，请先调用 build_anchor 构建训练观测缓存。")
        # Semantic_Feature 已完成相同的转换与校验，直接调用此模块时仍完整检查。
        if not indices_prepared:
            user, service, time = self.prepare_indices(user, service, time)
        user_stats, user_valid = self._aggregate_path(user, service, time, *user_scores, user_side=True)
        service_stats, service_valid = self._aggregate_path(user, service, time, *service_scores, user_side=False)
        return SemanticRetrievalResult(
            statistics=torch.stack([user_stats, service_stats], dim=1),
            valid_mask=torch.stack([user_valid, service_valid], dim=1),
        )
