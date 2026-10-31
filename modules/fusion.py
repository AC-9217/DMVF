import torch
from torch import nn

from config import Config
from modules.predictor import SelfAttentionBlock


# 使用一层残差注意力完成单个模块内部的 Token 组合。
class TokenFusion(nn.Module):
    """
    功能描述:
        为内部 Token 添加角色标识，并屏蔽缺失位置；保持 Token 数量和维度不变。
    参数:
        config：Config，注意力维度、头数、前馈层和丢弃率配置，标量。
        token_count：int，当前模块内部的固定 Token 数量，标量。
    返回值:
        TokenFusion：前向输入、输出形状均为 [B, token_count, config.e] 的融合层。
    """

    # 初始化模块内部的角色标识和单层注意力。
    def __init__(self, config: Config, token_count: int) -> None:
        """
        功能描述:
            初始化固定数量的角色向量和带残差的注意力块。
        参数:
            config：Config，注意力配置，标量。
            token_count：int，正整数 Token 数量，标量。
        返回值:
            None：完成融合层初始化。
        """
        super().__init__()
        if token_count < 1:
            raise ValueError("模块内部融合的 Token 数量必须为正整数。")
        self.role_embedding = nn.Parameter(torch.empty(token_count, config.e))#[token_count,e]
        nn.init.normal_(self.role_embedding, std=0.02)
        self.block = SelfAttentionBlock(config)

    # 融合有效 Token，并让全部缺失的样本安全输出零向量。
    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        功能描述:
            添加角色标识，执行带键掩码的注意力，并清零无效位置。
        参数:
            tokens：torch.Tensor，待融合表示，形状为 [B, K, E]。
            valid_mask：torch.Tensor | None，有效位置，形状为 [B, K]；None 表示全部有效。
        返回值:
            torch.Tensor：内部融合结果，形状为 [B, K, E]，无效位置为零。
        """
        # 校验内部 Token 契约，并准备默认有效性掩码。
        if tokens.ndim != 3 or tokens.shape[1:] != self.role_embedding.shape:
            raise ValueError("模块内部 Token 数量或维度与融合层不一致。")
        if valid_mask is None:
            valid_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if valid_mask.shape != tokens.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("模块内部有效性掩码必须为形状 [B, K] 的布尔张量。")

        # forward
        padding_mask = ~valid_mask# true代表空
        padding_mask[:, 0] &= valid_mask.any(dim=1)# 全缺失时开放一个零占位键，无布尔高级索引
        tokens = (tokens + self.role_embedding.unsqueeze(0)).masked_fill(~valid_mask.unsqueeze(-1), 0.0)# +role_embedding，无效位置置0
        tokens = self.block(tokens, padding_mask)
        return tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
