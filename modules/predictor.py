import torch
from torch import nn

from config import Config



## 单层编码器
class SelfAttentionBlock(nn.Module):
    """
    功能描述:
        对输入的 Token 序列做多头自注意力，再经过逐 Token 前馈网络更新，
        两处子层都是"dropout → 残差 → LayerNorm"的结构。

    Input Shape:
        tokens：待编码的 Token 序列，·[batch_size, num_tokens, config.e]。

    Output Shape:
        tokens：更新后的 Token 序列，·[batch_size, num_tokens, config.e]。
    """

    def __init__(self, config: Config) -> None:
        """
        功能描述:
            初始化多头自注意力、逐 Token 前馈网络与各自的 LayerNorm。

        Input Shape:
            config：预测器配置，·[标量]。

        Output Shape:
            return：无返回值，·[无]。
        """
        super().__init__()
        # 1.多头自注意力：在全部 Token 之间建模依赖
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.e,
            num_heads=config.predictor_heads,
            dropout=config.predictor_dropout,
            batch_first=True,
        )
        # 2.逐 Token 前馈网络（MLP）
        self.mlp = nn.Sequential(
            nn.Linear(config.e, config.predictor_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.predictor_dropout),
            nn.Linear(config.predictor_ffn_dim, config.e),
        )
        self.norm1 = nn.LayerNorm(config.e)
        self.norm2 = nn.LayerNorm(config.e)
        self.dropout = nn.Dropout(config.predictor_dropout)

    # 使用键掩码执行注意力，保留原有残差和归一化结构。
    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        功能描述:
            执行多头自注意力与前馈更新。

        Input Shape:
            tokens：待编码的 Token 序列，·[batch_size, num_tokens, config.e]。
            key_padding_mask：不参与注意力的键位置，·[batch_size, num_tokens]；None 表示全部有效。

        Output Shape:
            tokens：更新后的 Token 序列，·[batch_size, num_tokens, config.e]。
        """
        # 1.多头自注意力
        self_attended = self.self_attention(# [batch_size, num_tokens, config.e]
            tokens,
            tokens,
            tokens,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )[0]
        tokens = self.norm1(tokens + self.dropout(self_attended))# dropout→残差→LayerNorm [batch_size, num_tokens, config.e]
        # 2.逐 Token 前馈网络（MLP）
        feedforward = self.mlp(tokens)# <Linear(e → ffn_dim)> → GELU → Dropout → <Linear(ffn_dim → e)> [batch_size, num_tokens, config.e]
        return self.norm2(tokens + self.dropout(feedforward))# dropout→残差→LayerNorm [batch_size, num_tokens, config.e]


# 融合三个模块的全部有效 Token，并回归相对于中位数的修正量。
class Predictor(nn.Module):# 预测器（Predictor）
    """
    预测器。

    功能描述:
        添加模块和角色标识，对全部有效 Token 堆叠多层多头自注意力，
        展平后送入 MLP，预测相对于中位数底座的修正量。

    Input Shape:
        tokens：待预测的 Token 序列，·[batch_size, num_tokens, config.e]。

    Output Shape:
        prediction：预测结果，·[batch_size, config.predictor_out_dim]。
    """

    # 按三个模块的实际输出数量构建标识、注意力层和回归头。
    def __init__(self, config: Config) -> None:
        """
        功能描述:
            初始化模块及角色标识、多层注意力和覆盖全部 Token 的回归 MLP。

        Input Shape:
            config：预测器配置，·[标量]。

        Output Shape:
            return：无返回值，·[无]。
        """
        super().__init__()
        self.config = config
        self.token_count = config.predictor_total_tokens
        # 模块编号区分静态、动态、语义；角色编号区分用户、服务、配对、时间和局部。
        self.module_embedding = nn.Embedding(3, config.e)
        self.role_embedding = nn.Embedding(5, config.e)
        nn.init.normal_(self.module_embedding.weight, std=0.02)
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        self.register_buffer("module_ids", torch.tensor(config.predictor_module_ids), persistent=False)
        self.register_buffer("role_ids", torch.tensor(config.predictor_role_ids), persistent=False)
        # 1.多层多头自注意力，层数由 config.predictor_layers 决定
        self.layers = nn.ModuleList([SelfAttentionBlock(config) for _ in range(config.predictor_layers)])
        # 2.三个模块输出的全部 Token 展平后回归。
        self.mlp = nn.Sequential(
            nn.Linear(self.token_count * config.e, config.predictor_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.predictor_dropout),
            nn.Linear(config.predictor_mlp_hidden, config.predictor_out_dim),
        )

    # 屏蔽无效 Token 并使用全部有效表示预测修正量。
    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        功能描述:
            校验 Token 数量，添加来源标识并执行带掩码的跨模块注意力，最后展平回归。

        Input Shape:
            tokens：待预测的 Token 序列，·[batch_size, num_tokens, config.e]。
            valid_mask：有效位置，·[batch_size, num_tokens]；None 表示全部有效。

        Output Shape:
            prediction：预测结果，·[batch_size, config.predictor_out_dim]。
        """
        # 校验模块输出与回归头的契约，禁止静默丢弃多余 Token。
        if tokens.ndim != 3 or tokens.shape[1:] != (self.token_count, self.config.e):
            raise ValueError(
                f"预测器输入必须为 [B, {self.token_count}, {self.config.e}]，实际为 {tuple(tokens.shape)}。"
            )
        if valid_mask is None:
            valid_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if valid_mask.shape != tokens.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("预测器有效性掩码必须为形状 [B, num_tokens] 的布尔张量。")

        # 全缺失样本开放一个零占位键，避免注意力对全部屏蔽的位置执行 softmax。
        padding_mask = ~valid_mask# true表示无效值
        padding_mask[:, 0] &= valid_mask.any(dim=1)# 全空行开放首个键，避免布尔高级索引引发同步
        # module和role的embedding
        identity = self.module_embedding(self.module_ids) + self.role_embedding(self.role_ids)
        # 删除所有无观测的值
        tokens = (tokens + identity.unsqueeze(0)).masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        # 每层后再次清零无效查询，避免缺失位置携带偏置或其他 Token 的信息。
        for layer in self.layers:
            tokens = layer(tokens, padding_mask)
            tokens = tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        # mlp
        return self.mlp(tokens.flatten(start_dim=1))# [batch_size, config.predictor_out_dim]
