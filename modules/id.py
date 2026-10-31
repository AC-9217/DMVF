import math

import torch
from torch import nn

from config import Config


# ID 支路：提供用户/服务查表向量，并以门控残差补充 MF 主通路。
class IDFeature(nn.Module):
    """
    功能描述:
        保留原有 ID embedding 维度与默认初始化，用户和服务各自学习逐维门控，
        把 ID 表示以残差方式补充到 MF 表示上，再归一化为两个静态 Token。

    Input Shape:
        user_indices：当前 batch 的用户索引，·[batch_size]。
        service_indices：当前 batch 的服务索引，·[batch_size]。
        user_mf：用户 MF 主通路表示，·[batch_size, config.e]。
        service_mf：服务 MF 主通路表示，·[batch_size, config.e]。

    Output Shape:
        user_token：融合 ID 后的用户 Token，·[batch_size, config.e]。
        service_token：融合 ID 后的服务 Token，·[batch_size, config.e]。
    """

    # 初始化两张 ID 表、门控残差投影和归一化层。
    def __init__(self, config: Config) -> None:
        """
        功能描述:
            创建用户/服务 ID 查表、残差投影、逐维门控和 LayerNorm。

        Input Shape:
            config：配置对象，·[标量]。

        Output Shape:
            return：无返回值，·[无]。
        """
        super().__init__()
        self.config = config
        # ID 查表沿用原有维度及默认初始化。
        self.field_dim = int(config.semantic_dim)
        if self.field_dim < 1:
            raise ValueError("ID embedding 维度 semantic_dim 必须为正整数。")
        self.user_id_embedding = nn.Embedding(config.user_num, self.field_dim)
        self.service_id_embedding = nn.Embedding(config.service_num, self.field_dim)

        # MF 保留主通路，用户和服务各自学习逐维 ID 补充门控。
        if not math.isfinite(config.id_gate_init) or not 0 < config.id_gate_init < 1:
            raise ValueError("id_gate_init 必须为零至一之间的有限数值，且不能等于端点。")
        self.user_id_projection = nn.Linear(self.field_dim, config.e)
        self.service_id_projection = nn.Linear(self.field_dim, config.e)
        self.user_id_gate = nn.Linear(2 * config.e, config.e)
        self.service_id_gate = nn.Linear(2 * config.e, config.e)
        gate_logit = math.log(config.id_gate_init / (1.0 - config.id_gate_init))
        for gate in (self.user_id_gate, self.service_id_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, gate_logit)
        self.user_norm = nn.LayerNorm(config.e)
        self.service_norm = nn.LayerNorm(config.e)

    # 暴露原有两张 ID 表，供优化器保持专属权重衰减。
    def regularized_parameters(self) -> tuple[nn.Parameter, ...]:
        """
        功能描述:
            返回当前模块持有的两张 ID embedding 权重。
        参数:
            无显式参数；读取当前模块参数。
        返回值:
            tuple[nn.Parameter, ...]：两张 ID 表，形状分别为 [U, d] 和 [S, d]。
        """
        return self.user_id_embedding.weight, self.service_id_embedding.weight

    # 用逐维门控把 ID 残差补充到 MF 主通路上。
    def forward(
        self,
        user_indices: torch.Tensor,
        service_indices: torch.Tensor,
        user_mf: torch.Tensor,
        service_mf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        功能描述:
            查询 ID 表示，与 MF 表示一起计算逐维门控，输出归一化后的两侧 Token。

        Input Shape:
            user_indices：当前 batch 的用户索引，·[batch_size]。
            service_indices：当前 batch 的服务索引，·[batch_size]。
            user_mf：用户 MF 主通路表示，·[batch_size, config.e]。
            service_mf：服务 MF 主通路表示，·[batch_size, config.e]。

        Output Shape:
            user_token：融合 ID 后的用户 Token，·[batch_size, config.e]。
            service_token：融合 ID 后的服务 Token，·[batch_size, config.e]。
        """
        # ID feature
        user_id = self.user_id_projection(self.user_id_embedding(user_indices))
        service_id = self.service_id_projection(self.service_id_embedding(service_indices))
        # 门控
        user_gate = torch.sigmoid(self.user_id_gate(torch.cat([user_mf, user_id], dim=-1)))
        service_gate = torch.sigmoid(self.service_id_gate(torch.cat([service_mf, service_id], dim=-1)))
        # 封装token
        user_token = self.user_norm(user_mf + user_gate * user_id)
        service_token = self.service_norm(service_mf + service_gate * service_id)
        return user_token, service_token
