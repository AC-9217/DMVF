import torch
from torch import nn

from config import Config
from util.tensor_utils import as_index


# 窗口摘要统计的字段顺序（全部在"全局 z-score 后的空间"里，0 表示全局均值）
SUMMARY_NAMES = (
    "count_ratio",   # 窗口内观测数 / 2k
    "has_any",       # 窗口内是否有观测（0/1）
    "mean",          # 观测值的均值
    "std",           # 观测值的标准差（不足 2 个观测时为 0）
    "median",        # 观测值的中位数
    "nearest_value", # 按 |Δt| 最近的观测值
    "nearest_dist",  # 最近观测的距离 / k（没有观测时为 1）
    "dist_weighted_mean",# 按 1/|Δt| 加权的均值
)


class Local_Feature(nn.Module):# 局部时序特征（Local_Feature）
    """
    局部时序特征模块。

    功能描述:
        对目标调用 (u, s, t)，取目标时间片前后各 k 个时间片的归一化 QoS 值作为"证据"，
        但不把 2k 个位置逐个变成 Token（逐片值 91% 缺失、对 MAE 的增量实测为 0），
        而是先压成一组窗口摘要统计（观测数、均值、标准差、中位数、最近邻值/距离、距离加权均值），
        再用一个 Linear 投影成 1 个局部 Token。
        摘要把"要不要信窗口、信多少"交给主干自己学（观测数就是收缩强度的依据）。

    约定（可微调）:
        1.窗口半径 k 取 config.local_windows，输出 Token 数固定为 1（config.local_tokens 必须为 1）。
        2.目标时间片 t 自身不进窗口，窗口长度固定为 2k。
        3.归一化用 dataset.py 摊平后那些非零格子的全局 z-score（本项目里每个 (user, service) 组合平均只有
          个位数的观测，逐组合统计不可靠，故不做逐组合归一化）。
        4.未被观测到的 (user, service, time) 以及越界的邻居时间片都不参与统计：
          它们不进 count、不进均值/中位数/最近邻；全空窗口时 count=0、has_any=0、其余统计量为 0、
          nearest_dist=1，模型据此知道"这个窗口什么都没有"。
        5.offsets 按 |Δt| 升序存放，最近邻就是"第一个有观测的位置"，无需再排序。

    Input Shape:
        train_matrix：训练集 QoS 张量，·[user_num, service_num, time_num]，缺失位置为 0。
        user_indices：当前 batch 的用户索引，·[batch_size]。
        service_indices：当前 batch 的服务索引，·[batch_size]。
        time_indices：当前 batch 的时间片索引，·[batch_size]。

    Output Shape:
        local_matrix：归一化后的 QoS 张量，·[user_num, service_num, time_num]。
        local_mask：  观测指示张量，·[user_num, service_num, time_num]，1 表示该位置在训练集有观测。
        local_tokens：局部 Token 序列，·[batch_size, 1, config.e]。
        valid_mask：局部 Token 的布尔有效性掩码，·[batch_size, 1]。
    """

    def __init__(self, cfg: Config) -> None:
        """
        功能描述:
            初始化窗口半径、按 |Δt| 排序的相对偏移、归一化 QoS 张量、观测指示与摘要投影层。

        Input Shape:
            cfg：配置对象，·[标量]。

        Output Shape:
            return：无返回值，·[无]。
        """
        super().__init__()
        self.cfg = cfg

        # 1.窗口半径 k 与输出 Token 数
        self.window = int(cfg.local_windows)
        if self.window < 1:
            raise ValueError(f"窗口半径必须是正整数，当前 window={self.window}。")
        if int(cfg.local_tokens) != 1:
            raise ValueError(f"局部支路固定输出 1 个摘要 Token，请把 local_tokens 设为 1（当前 {cfg.local_tokens}）。")

        # 2.相对位置偏移 [-k, ..., -1, 1, ..., k]，不含目标时间片 0；按 |Δt| 升序排列，便于取最近邻
        offsets = torch.arange(-self.window, self.window + 1)
        offsets = offsets[offsets != 0]
        self.register_buffer("offsets", offsets[torch.argsort(offsets.abs(), stable=True)], persistent=False)# [2k]
        # 距离权重 1/|Δt|（距离加权均值用）与归一化距离（摘要特征用）
        self.register_buffer("distance", self.offsets.abs().float(), persistent=False)# [2k]
        self.register_buffer("inverse_distance", 1.0 / self.distance, persistent=False)# [2k]

        # 3.归一化统计量（只在 build_local_matrix 里更新）
        self.rt_mean = 0.0
        self.rt_std = 1.0

        # 4.归一化后的 QoS 张量与观测指示：forward 直接查表用
        #   persistent=False：重建成本很低，且归一化统计量不在 state_dict 里，单独存也没法用
        self.register_buffer(
            "local_matrix",
            torch.zeros(cfg.user_num, cfg.service_num, cfg.time_num, dtype=torch.float32),
            persistent=False,
        )# [user_num, service_num, time_num]
        self.register_buffer(
            "local_mask",
            torch.zeros(cfg.user_num, cfg.service_num, cfg.time_num, dtype=torch.bool),
            persistent=False,
        )# [user_num, service_num, time_num]

        # 5.窗口摘要 → 1 个局部 Token
        self.summary_projection = nn.Linear(len(SUMMARY_NAMES), cfg.e)

    ## 构建局部 QoS 张量（归一化）
    def build_local_matrix(self, train_matrix: torch.Tensor) -> torch.Tensor:
        """
        功能描述:
            训练前调用一次：直接复用 data/dataset.py 摊平好的训练集 QoS 张量，
            用其中非零格子的全局 z-score 归一化，结果写入 self.local_matrix；
            同时把"哪些位置有观测"写进 self.local_mask。
            未被观测到的位置在 local_matrix 里保持 0（归一化空间里的全局均值）。
            dataset.py 已过滤 rt <= 0，所以非零即"有观测"，不必再传 train_info 摊平一遍。

        Input Shape:
            train_matrix：[user_num, service_num, time_num] FloatTensor，训练集 rt，缺失位置为 0。

        Output Shape:
            local_matrix：[user_num, service_num, time_num] FloatTensor，归一化后的 QoS 张量（同时写入 self.local_mask）。
        """
        # ---------- 1. 搬到当前 device，并挑出真实观测 ----------
        device = self.local_matrix.device
        qos = train_matrix.to(device=device, dtype=torch.float32)# [user_num, service_num, time_num]
        observed = qos > 0# 非零即有观测 [user_num, service_num, time_num]
        rt = qos[observed]# [num_observed]
        if rt.numel() == 0:
            raise ValueError("train_matrix 里没有任何正观测值，无法构建局部 QoS 张量。")

        # ---------- 2. 全局 z-score 归一化 ----------
        self.rt_mean = rt.mean().item()
        self.rt_std = max(rt.std(unbiased=False).item(), 1e-8)# 所有 rt 相同时防止除零
        normalized = (qos - self.rt_mean) / self.rt_std# [user_num, service_num, time_num]

        # ---------- 3. 未观测的位置归零 ----------
        normalized = torch.where(observed, normalized, torch.zeros_like(normalized))# [user_num, service_num, time_num]

        # ---------- 4. 写回 buffer ----------
        self.local_matrix.copy_(normalized)
        self.local_mask.copy_(observed)# 观测指示张量
        return self.local_matrix

    ## 窗口内取值与观测掩码
    def gather_window(
        self, user_idx: torch.Tensor, service_idx: torch.Tensor, time_idx: torch.Tensor,
        *, indices_validated: bool = False,
    ):
        """
        功能描述:
            取每个目标调用前后各 k 个时间片的归一化 QoS 值与"该位置是否有观测"的掩码。
            越界的邻居时间片按缺失处理（掩码为 0），非法的时间片索引直接报错，避免负索引静默取错行。

        Input Shape:
            user_idx：   用户索引，·[batch_size]。
            service_idx：服务索引，·[batch_size]。
            time_idx：   时间片索引，·[batch_size]。

        Output Shape:
            values：  窗口内的归一化 QoS 值（缺失/越界为 0），·[batch_size, 2k]。
            observed：窗口内的观测掩码，·[batch_size, 2k]。
        """
        # 越界（尤其是负索引）会静默取到错误的时间片，直接报错
        if not indices_validated and time_idx.numel() > 0 and ((time_idx < 0) | (time_idx >= self.cfg.time_num)).any():
            raise ValueError(
                f"时间片索引超出 [0, {self.cfg.time_num - 1}] 范围："
                f"min={int(time_idx.min())}，max={int(time_idx.max())}。"
            )
        neighbor_time = time_idx.unsqueeze(1) + self.offsets.unsqueeze(0)# [batch_size, 2k]
        in_range = (neighbor_time >= 0) & (neighbor_time < self.cfg.time_num)# [batch_size, 2k]
        clipped = neighbor_time.clamp(0, self.cfg.time_num - 1)# [batch_size, 2k]
        values = self.local_matrix[user_idx.unsqueeze(1), service_idx.unsqueeze(1), clipped]# [batch_size, 2k]
        observed = self.local_mask[user_idx.unsqueeze(1), service_idx.unsqueeze(1), clipped] & in_range
        values = torch.where(observed, values, torch.zeros_like(values))# 缺失或越界 → 0 [batch_size, 2k]
        return values, observed

    ## 窗口 → 摘要统计
    def summarize(self, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """
        功能描述:
            把窗口压成 SUMMARY_NAMES 顺序的摘要统计。全空窗口时除 nearest_dist=1 外全部为 0。

        Input Shape:
            values：  窗口内的归一化 QoS 值，·[batch_size, 2k]。
            observed：窗口内的观测掩码，·[batch_size, 2k]。

        Output Shape:
            summary：窗口摘要，·[batch_size, len(SUMMARY_NAMES)]。
        """
        mask = observed.to(values.dtype)# [batch_size, 2k]
        count = mask.sum(dim=1)# [batch_size]
        safe_count = count.clamp_min(1.0)
        has_any = (count > 0).to(values.dtype)# [batch_size]
        mean = (values * mask).sum(dim=1) / safe_count# 均值
        variance = (values.pow(2) * mask).sum(dim=1) / safe_count - mean.pow(2)
        std = variance.clamp_min(0).sqrt() * has_any# 标准差（不足两个观测时方差为负/零，会被夹到 0）
        # 中位数：未观测位置填 NaN 后用 nanmedian 忽略掉。
        # 这里不能用"填 +inf 再 median"：观测数不足一半时中位数会取到 inf，进而把摘要污染成 NaN。
        masked = torch.where(observed, values, torch.full_like(values, float("nan")))
        median = torch.nanmedian(masked, dim=1).values# 全空窗口为 NaN
        median = torch.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0)
        # 最近邻：offsets 已按 |Δt| 升序，第一个观测到的位置即最近邻
        nearest_index = observed.to(torch.int32).argmax(dim=1)# 首个 True；全空窗口同样返回 0
        nearest_value = values.gather(1, nearest_index.unsqueeze(1)).squeeze(1) * has_any
        nearest_dist = self.distance[nearest_index] / self.window * has_any + (1.0 - has_any)
        # 距离加权均值：权重 1/|Δt|
        weight = self.inverse_distance.unsqueeze(0) * mask# [batch_size, 2k]
        weighted = (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-8) * has_any
        summary = torch.stack([
            count / (2.0 * self.window),# count_ratio
            has_any,
            mean,
            std,
            median,
            nearest_value,
            nearest_dist,
            weighted,
        ], dim=1)# [batch_size, len(SUMMARY_NAMES)]
        return torch.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)# 兜住浮点边界，别把 NaN 灌进主干

    # 统一生成局部 Token 及窗口有效性，供时间相关模块直接组合。
    def forward(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor,
        *, indices_validated: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        功能描述:
            取每个目标调用前后各 k 个时间片的归一化 QoS 值，压成窗口摘要后投影成 1 个局部 Token，
            同时返回窗口内是否存在有效观测的布尔掩码。

        Input Shape:
            user_indices：当前 batch 的用户索引，·[batch_size]。
            service_indices：当前 batch 的服务索引，·[batch_size]。
            time_indices：当前 batch 的时间片索引，·[batch_size]。

        Output Shape:
            local_tokens：torch.Tensor，局部 Token 序列，·[batch_size, 1, config.e]。
            valid_mask：torch.Tensor，布尔有效性掩码，·[batch_size, 1]。
        """
        # 1.对齐当前模块所在 device
        device = self.local_matrix.device
        user_idx = as_index(user_indices, device)# [batch_size]
        service_idx = as_index(service_indices, device)# [batch_size]
        time_idx = as_index(time_indices, device)# [batch_size]

        # 2.取窗口 → 摘要 → 1 个 Token
        values, observed = self.gather_window(
            user_idx, service_idx, time_idx, indices_validated=indices_validated,
        )# [batch_size, 2k]
        summary = self.summarize(values, observed)# [batch_size, len(SUMMARY_NAMES)]
        local_tokens = self.summary_projection(summary).unsqueeze(1)# [batch_size, 1, config.e]

        # 仅包含至少一条有效邻居观测的窗口可参与后续融合。
        valid_mask = observed.any(dim=1, keepdim=True)# [batch_size, 1]
        return local_tokens, valid_mask
