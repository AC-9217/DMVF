import torch
from torch import nn

from config import Config
from modules.fusion import TokenFusion
from modules.local import Local_Feature
from modules.temporal import AbsoluteTimeFeature


# Absolute + local
class Time_Dependent_Feature(nn.Module):
    """
    功能描述:
        保留原有绝对时间、Local 计算，先进行内部融合，再输出时间相关 Token。
    参数:
        config：Config，时间、窗口和融合配置，标量。
    返回值:
        Time_Dependent_Feature：固定输出两个 Token 及其有效性掩码的模块。
    """

    # 同时构建原有时间路径和 Local 路径。
    def __init__(self, config: Config) -> None:
        """
        功能描述:
            同时初始化绝对时间和局部路径，并建立单层内部融合。
        参数:
            config：Config，两条路径的维度及内部融合配置，标量。
        返回值:
            None：完成时间相关模块初始化。
        """
        super().__init__()
        self.config = config
        self.absolute_time_feature = AbsoluteTimeFeature(config)
        self.local = Local_Feature(config)
        self.fusion = TokenFusion(config, config.time_dependent_tokens)

    ## 暴露绝对时间查表参数，保持原有专属权重衰减。
    def regularized_parameters(self) -> tuple[nn.Parameter, ...]:
        """
        功能描述:
            返回绝对时间 embedding 权重，供优化器设置专属正则。
        参数:
            无显式参数；读取当前模块配置。
        返回值:
            tuple[nn.Parameter, ...]：一个权重，形状为 [T, time_dim]。
        """
        return self.absolute_time_feature.regularized_parameters()# time_embedding.weight

    # 将训练观测写入原有 Local 窗口缓存。
    def build_local_matrix(self, train_matrix: torch.Tensor) -> torch.Tensor:
        """
        功能描述:
            调用现有 Local 预处理，仅使用训练集进行归一化及窗口查询准备。
        参数:
            train_matrix：torch.Tensor，训练 QoS 张量，形状为 [U, S, T]。
        返回值:
            torch.Tensor：归一化后的局部缓存，形状为 [U, S, T]。
        """
        return self.local.build_local_matrix(train_matrix)

    # 先融合时间和局部摘要，再返回 Token 与有效性掩码。
    def forward(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor,
        *, indices_validated: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        功能描述:
            保留时间 Token 直接输入，并屏蔽完全没有训练观测的局部窗口。
        参数:
            user_indices：torch.Tensor，用户索引，形状为 [B]。
            service_indices：torch.Tensor，服务索引，形状为 [B]。
            time_indices：torch.Tensor，时间索引，形状为 [B]。
        返回值:
            tuple[torch.Tensor, torch.Tensor]：融合表示 [B, K, E] 和布尔掩码 [B, K]。
        """
        # 两个叶子模块负责独立调用时的设备对齐，避免在父模块重复转换。
        # 绝对时间token
        absolute_token = self.absolute_time_feature(time_indices)
        absolute_mask = torch.ones(absolute_token.shape[:2], dtype=torch.bool, device=absolute_token.device)# 全1观测掩码

        # 由 Local 统一生成窗口摘要 Token 和有效性掩码。
        local_token, local_mask = self.local(
            user_indices, service_indices, time_indices, indices_validated=indices_validated,
        )

        # 两条路径先进行模块内部组合，再交给跨模块预测器。
        tokens = torch.cat([absolute_token, local_token], dim=1)
        valid_mask = torch.cat([absolute_mask, local_mask], dim=1)
        return self.fusion(tokens, valid_mask), valid_mask
