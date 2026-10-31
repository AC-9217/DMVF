import math

import torch
from torch import nn

from config import Config
from util.tensor_utils import as_index


class AbsoluteTimeFeature(nn.Module):
    """
    功能描述:
        将时间片编号映射为可学习时间向量，再投影为一个绝对时间 Token。

    Input Shape:
        cfg：时间片数量、向量维度和 Token 维度等配置，·[标量]。
        time_indices：时间片编号，·[batch_size]。

    Output Shape:
        time_token：绝对时间 Token，·[batch_size, 1, cfg.e]。
    """

    def __init__(self, cfg: Config) -> None:
        """
        功能描述:
            初始化时间片 embedding、Token 投影层及时间向量的初始值。

        Input Shape:
            cfg：时间片数量、向量维度和初始化标准差等配置，·[标量]。

        Output Shape:
            return：无返回值。
        """
        super().__init__()
        # 检测参数合法性
        if cfg.time_num < 1 or cfg.time_dim < 1:
            raise ValueError("time_num 和 time_dim 必须为正整数。")
        if not math.isfinite(cfg.temporal_init_std) or cfg.temporal_init_std <= 0:
            raise ValueError("temporal_init_std 必须为有限正数。")
        # embedding
        self.time_embedding = nn.Embedding(cfg.time_num, cfg.time_dim)
        # linear
        self.projection = nn.Linear(cfg.time_dim, cfg.e)
        nn.init.normal_(self.time_embedding.weight, std=cfg.temporal_init_std)

    ## 获取time_embedding的参数
    def regularized_parameters(self) -> tuple[nn.Parameter, ...]:
        """
        功能描述:
            返回需要独立权重衰减的时间查表向量，不包含 Token 投影层。

        Input Shape:
            无显式参数；读取当前模块的时间 embedding。

        Output Shape:
            parameters：时间 embedding 的权重，·[1]。
        """
        return (self.time_embedding.weight,)

    def forward(self, time_indices: torch.Tensor) -> torch.Tensor:
        """
        功能描述:
            查表并投影时间片编号，生成一个绝对时间 Token。

        Input Shape:
            time_indices：[batch_size]，时间片索引。
        Output Shape:
            time_token：[batch_size, 1, cfg.e]。
        """
        time_idx = as_index(time_indices, self.time_embedding.weight.device)
        return self.projection(self.time_embedding(time_idx)).unsqueeze(1)

