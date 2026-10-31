import torch
from torch import nn

from config import Config


# 中位数底座：维护推理/留一法中位数、观测数量及配对统计 Token。
class MedianAnchor(nn.Module):
    """
    功能描述:
        保留原有中位数统计规则，缓存每个 (user, service, time) 的训练留一法结果，
        并把推理中位数、留一法中位数和观测数量整理成配对 Token，供时间无关模块融合。

    Input Shape:
        train_info：训练集索引张量，·[num_observations, 3]，每行 [time, user, service]。
        train_rt：  训练集响应时间，·[num_observations, 1]。
        user_indices：当前 batch 的用户索引，·[batch_size]。
        service_indices：当前 batch 的服务索引，·[batch_size]。
        time_indices：当前 batch 的时间索引，·[batch_size]。

    Output Shape:
        anchor_value：中位数残差底座，·[batch_size]。
        pair_token：配对统计 Token，·[batch_size, config.e]。
    """

    # 初始化中位数、留一法和观测缓存，并建立配对统计投影。
    def __init__(self, config: Config, enable_token: bool = True) -> None:
        """
        功能描述:
            创建全量中位数、留一法中位数、观测数量及观测标记缓存，可按需关闭配对投影。

        Input Shape:
            config：配置对象，·[标量]。
            enable_token：bool，是否构建可训练的配对 Token 投影，·[标量]。

        Output Shape:
            return：无返回值，·[无]。
        """
        super().__init__()
        self.config = config
        # 中位数底座和训练样本标记在预处理时重建，不写入权重文件。
        self.register_buffer("anchor", torch.zeros(config.user_num, config.service_num), persistent=False)
        self.register_buffer(
            "anchor_loo", torch.zeros(config.user_num, config.service_num, config.time_num), persistent=False
        )
        self.register_buffer("pair_counts", torch.zeros(config.user_num, config.service_num), persistent=False)# 每个(u,s)组合在训练集中出现了多少次
        self.register_buffer("observed", torch.zeros(config.user_num, config.service_num, config.time_num, dtype=torch.bool), persistent=False)# 每个(u,s,t)三元组是否被观测过
        self.anchor_ready = False

        # 静态模块关闭时仅保留统计缓存，不创建无用的可训练投影。
        self.anchor_token = nn.Linear(2, config.e) if enable_token else None

    ## 构建 (u,s) 锚
    def build_anchor(self, train_info: torch.Tensor, train_rt: torch.Tensor) -> torch.Tensor:
        """
        功能描述:
            推理锚使用同一用户和服务组合的全部训练观测中位数；训练锚仅排除当前样本，
            使用该组合剩余观测的中位数。偶数条观测取中间两条的平均，奇数条取正中间一条。
            若该组合只有一条观测，训练锚使用排除当前样本后的全局中位数；
            若整个训练集只有一条观测，则使用固定值 0，避免兜底锚包含目标值。
            推理时未出现过的组合仍使用原有的全局中位数兜底规则。
            每个时间、用户、服务索引必须唯一，留一法锚按该三元组缓存。

        Input Shape:
            train_info：[N, 3] LongTensor，每行 [time, user, service]。
            train_rt：  [N, 1] FloatTensor，响应时间。

        Output Shape:
            anchor：[user_num, service_num] FloatTensor，每个组合的中位数（缺失组合为全局中位数）。
            同时更新 anchor_loo，形状为 [user_num, service_num, time_num]，供训练前向查询。
        """
        # 校验样本形状，并将预处理数据统一到处理器内存。
        self.anchor_ready = False
        cfg = self.config
        if train_info.ndim != 2 or train_info.shape[1] != 3:
            raise ValueError("训练索引必须为形状 [N, 3] 的张量。")
        info = train_info.detach().cpu().long()
        rt = train_rt.detach().cpu().reshape(-1).float()
        num_observations = info.shape[0]# 训练集观测数
        if num_observations == 0 or rt.numel() != num_observations:
            raise ValueError("构建锚需要非空训练集，且索引与响应时间的样本数必须一致。")

        # 校验索引范围及唯一性，保证每个训练样本能查询自己的留一法锚。
        limits = info.new_tensor([cfg.time_num, cfg.user_num, cfg.service_num])
        if ((info < 0) | (info >= limits)).any():
            raise ValueError("训练索引超出配置的时间、用户或服务范围。")
        time, user, service = info.unbind(dim=1)
        pair_id = user * cfg.service_num + service # 将用户和服务编码为组合编号。
        cell_id = pair_id * cfg.time_num + time
        if torch.unique(cell_id).numel() != num_observations:
            raise ValueError("训练集中存在重复的时间、用户、服务索引，无法按三元组缓存逐样本留一法锚。")

        # 先按响应时间排序，再稳定地按组合排序，使组内响应时间保持升序。
        rt_order = torch.argsort(rt, stable=True)
        global_sorted_rt = rt[rt_order]
        global_median = global_sorted_rt[(num_observations - 1) // 2].item() # 保留原有全局中位数规则。
        order = rt_order[torch.argsort(pair_id[rt_order], stable=True)]
        sorted_rt = rt[order]
        pair_ids, counts = torch.unique_consecutive(pair_id[order], return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts # 每个组合在排序结果中的起始位置。

        # 按组计算全部训练观测的中位数，写入推理锚。
        lower = starts + (counts - 1) // 2
        upper = starts + counts // 2
        medians = (sorted_rt[lower] + sorted_rt[upper]) / 2
        anchor = torch.full((cfg.user_num, cfg.service_num), global_median, dtype=torch.float32)
        anchor[pair_ids // cfg.service_num, pair_ids % cfg.service_num] = medians

        # 计算每条观测删除自身后，中位数在原排序结果中的两个位置。
        sample_counts = torch.repeat_interleave(counts, counts)
        sample_starts = torch.repeat_interleave(starts, counts)
        sample_positions = torch.arange(num_observations) - sample_starts # 当前样本在组内的位置。
        remaining_counts = (sample_counts - 1).clamp_min(1) # 单观测组合随后单独兜底。
        lower_rank = (remaining_counts - 1) // 2
        upper_rank = remaining_counts // 2
        lower_index = sample_starts + lower_rank + (sample_positions <= lower_rank).long()
        upper_index = sample_starts + upper_rank + (sample_positions <= upper_rank).long()
        lower_index = lower_index.clamp_max(num_observations - 1) # 防止单观测组合临时索引越界。
        upper_index = upper_index.clamp_max(num_observations - 1)
        loo_sorted = (sorted_rt[lower_index] + sorted_rt[upper_index]) / 2
        loo_values = torch.empty_like(rt)
        loo_values[order] = loo_sorted # 将留一法中位数恢复为原始样本顺序。

        # 单观测组合用全局其他观测的中位数兜底，仍然排除当前目标值。
        singleton_indices = order[sample_counts == 1]
        if singleton_indices.numel() > 0:
            if num_observations == 1:
                loo_values[singleton_indices] = 0.0
            else:
                global_ranks = torch.empty_like(rt_order)
                global_ranks[rt_order] = torch.arange(num_observations)
                target_ranks = global_ranks[singleton_indices]
                global_lower_rank = (num_observations - 2) // 2
                global_upper_rank = (num_observations - 1) // 2
                global_lower = global_lower_rank + (target_ranks <= global_lower_rank).long()
                global_upper = global_upper_rank + (target_ranks <= global_upper_rank).long()
                loo_values[singleton_indices] = (
                    global_sorted_rt[global_lower] + global_sorted_rt[global_upper]
                ) / 2

        # 按样本三元组写入训练锚，并同步训练和推理缓存到模型设备。
        anchor_loo = torch.full(
            (cfg.user_num, cfg.service_num, cfg.time_num), global_median, dtype=torch.float32
        )
        anchor_loo[user, service, time] = loo_values
        self.anchor.copy_(anchor)
        self.anchor_loo.copy_(anchor_loo)
        # 保存原始观测数量及三元组标记，供训练时排除目标观测。
        pair_counts = torch.zeros_like(anchor)
        pair_counts[pair_ids // cfg.service_num, pair_ids % cfg.service_num] = counts.float()
        self.pair_counts.copy_(pair_counts)
        self.observed.zero_()
        self.observed[user.to(self.observed.device), service.to(self.observed.device), time.to(self.observed.device)] = True
        self.anchor_ready = True
        return self.anchor

    ## 按训练或推理状态查询同一套中位数底座。
    def get_anchor(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        功能描述:
            训练使用排除目标观测的缓存，推理使用全部训练观测的配对中位数。
        参数:
            user_indices：torch.Tensor，设备已对齐的用户索引，形状为 [B]。
            service_indices：torch.Tensor，设备已对齐的服务索引，形状为 [B]。
            time_indices：torch.Tensor，设备已对齐的时间索引，形状为 [B]。
        返回值:
            torch.Tensor：本次查询的中位数底座，形状为 [B]。
        """
        # 判断底座是否已经初始化
        if not self.anchor_ready:
            raise RuntimeError("中位数底座尚未构建，请先调用 build_anchor。")
        if self.training:
            return self.anchor_loo[user_indices, service_indices, time_indices]
        return self.anchor[user_indices, service_indices]

    # 查询与中位数一致的配对可靠性统计。
    def get_pair_statistics(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        功能描述:
            组合中位数和对数观测数量，训练时仅对实际训练观测减去自身。
        参数:
            user_indices：torch.Tensor，用户索引，形状为 [B]。
            service_indices：torch.Tensor，服务索引，形状为 [B]。
            time_indices：torch.Tensor，时间索引，形状为 [B]。
        返回值:
            torch.Tensor：顺序为中位数、log(观测数的统计+1)，形状为 [B, 2]。
        """
        anchor = self.get_anchor(user_indices, service_indices, time_indices)# 中位数值
        count = self.pair_counts[user_indices, service_indices]# 观测数量
        if self.training:
            count = count - self.observed[user_indices, service_indices, time_indices].to(count.dtype)
        return torch.stack([anchor, torch.log1p(count.clamp_min(0))], dim=-1)

    # 查询中位数底座并生成配对统计 Token。
    def forward(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        功能描述:
            使用与训练/推理一致的留一规则取出中位数和观测数量，投影成配对 Token。

        Input Shape:
            user_indices：当前 batch 的用户索引，·[batch_size]。
            service_indices：当前 batch 的服务索引，·[batch_size]。
            time_indices：当前 batch 的时间索引，·[batch_size]。

        Output Shape:
            anchor_value：中位数残差底座，·[batch_size]。
            pair_token：配对统计 Token，·[batch_size, config.e]。
        """
        if self.anchor_token is None:
            raise RuntimeError("配对 Token 已关闭，请使用 get_anchor 查询中位数底座。")
        pair_statistics = self.get_pair_statistics(user_indices, service_indices, time_indices)# [中位数、log(观测数的统计+1)]
        return pair_statistics[:, 0], self.anchor_token(pair_statistics)# 复用统计中的中位数；Linear<2 → e>
