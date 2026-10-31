import torch
from torch import nn
from util.tensor_utils import as_index

from config import Config
from modules.anchor import MedianAnchor
from modules.fusion import TokenFusion
from modules.id import IDFeature
from modules.MF import MF


# 组织中位数底座、冻结 MF 和 ID embedding，并融合三个静态 Token。
class Time_Independent_Feature(nn.Module):# 与时间无关的特征（Time_Independent_Feature）
    """
    功能描述:
        与 time_dependent.py 保持一致的组织方式：本文件只负责组合子模块，
        中位数底座由 anchor.py 实现，矩阵分解由 MF.py 实现，ID 支路由 id.py 实现。
        三条静态路径分别输出用户、服务和配对 Token，内部融合后交给跨模块预测器。

    Input Shape:
        user_indices：当前 batch 的用户索引，·[batch_size]。
        service_indices：当前 batch 的服务索引，·[batch_size]。
        time_indices：当前 batch 的时间索引，·[batch_size]。

    Output Shape:
        time_independent_tokens：静态 Token，·[batch_size, 3, config.e]。
        valid_mask：静态 Token 有效性，·[batch_size, 3]。
        anchor_value：中位数残差底座，·[batch_size]。
    """

    # 初始化中位数底座、MF、ID 三条静态路径和内部融合。
    def __init__(self, config: Config) -> None:
        """
        功能描述:
            按原有顺序组织 ID、MF、中位数三条静态路径，并建立三个 Token 的内部融合。

        Input Shape:
            config：配置对象，·[标量]。

        Output Shape:
            return：无返回值，·[无]。
        """
        super().__init__()
        self.config = config
        # 1.ID 支路：先构建两张 ID 表，保持原有 ID embedding 的随机初始化顺序。
        self.id_feature = IDFeature(config)
        # 2.MF 支路：冻结的矩阵分解隐向量及其投影。
        self.mf = MF(config)
        # 3.中位数底座：全量/留一法中位数、观测数量和配对统计 Token。
        self.anchor_layer = MedianAnchor(config)
        # 4.三个静态 Token 先内部融合。
        self.fusion = TokenFusion(config, 3)

    # --- 以下只读入口保持原有外部访问方式，实现均在各自的子模块中。 ---

    # anchor_layer
    @property
    def anchor(self) -> torch.Tensor:
        return self.anchor_layer.anchor
    @property
    def anchor_loo(self) -> torch.Tensor:
        return self.anchor_layer.anchor_loo
    @property
    def pair_counts(self) -> torch.Tensor:
        return self.anchor_layer.pair_counts
    @property
    def observed(self) -> torch.Tensor:
        return self.anchor_layer.observed
    @property
    def anchor_ready(self) -> bool:
        return self.anchor_layer.anchor_ready
    @property
    def anchor_token(self) -> nn.Linear:
        return self.anchor_layer.anchor_token
    
    # MF 
    @property
    def user_factors(self) -> torch.Tensor:
        return self.mf.user_factors
    @property
    def service_factors(self) -> torch.Tensor:
        return self.mf.service_factors
    @property
    def user_projection(self) -> nn.Module:
        return self.mf.user_projection
    @property
    def service_projection(self) -> nn.Module:
        return self.mf.service_projection

    # id_feature
    @property
    def field_dim(self) -> int:
        return self.id_feature.field_dim
    @property
    def user_id_embedding(self) -> nn.Embedding:
        return self.id_feature.user_id_embedding
    @property
    def service_id_embedding(self) -> nn.Embedding:
        return self.id_feature.service_id_embedding
    @property
    def user_id_projection(self) -> nn.Linear:
        return self.id_feature.user_id_projection
    @property
    def service_id_projection(self) -> nn.Linear:
        return self.id_feature.service_id_projection
    @property
    def user_id_gate(self) -> nn.Linear:
        return self.id_feature.user_id_gate
    @property
    def service_id_gate(self) -> nn.Linear:
        return self.id_feature.service_id_gate
    @property
    def user_norm(self) -> nn.LayerNorm:
        return self.id_feature.user_norm
    @property
    def service_norm(self) -> nn.LayerNorm:
        return self.id_feature.service_norm

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
        return self.id_feature.regularized_parameters()

# -------------------------------初始化---------------------------------------------
    ## 压缩时间维度（中位数）
    def compress_time_dimension(self, train_info: torch.Tensor, train_rt: torch.Tensor):
        """
        功能描述:
            转交给 MF 子模块，把同一 (user, service) 组合的多个时间片观测汇总为中位数。
        参数:
            train_info：torch.Tensor，训练索引，形状为 [N, 3]。
            train_rt：torch.Tensor，训练响应时间，形状为 [N, 1]。
        返回值:
            tuple[torch.Tensor, torch.Tensor]：压缩后的组合索引 [P, 2] 及中位数 [P, 1]。
        """
        return self.mf.compress_time_dimension(train_info, train_rt)

    ## 构建参考矩阵（矩阵分解）
    def build_reference_matrix(
        self,
        train_info: torch.Tensor,
        train_rt: torch.Tensor,
        test_info: torch.Tensor | None = None,
        test_rt: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        功能描述:
            转交给 MF 子模块完成矩阵分解，并写回用户/服务隐向量。
        参数:
            train_info：torch.Tensor，训练索引，形状为 [N, 3]。
            train_rt：torch.Tensor，训练响应时间，形状为 [N, 1]。
            test_info：torch.Tensor | None，可选测试索引，形状为 [M, 3]。
            test_rt：torch.Tensor | None，可选测试响应时间，形状为 [M, 1]。
        返回值:
            torch.Tensor：矩阵分解补全后的参考矩阵，形状为 [user_num, service_num]。
        """
        return self.mf.build_reference_matrix(train_info, train_rt, test_info, test_rt)

    ## 构建 (u,s) 锚
    def build_anchor(self, train_info: torch.Tensor, train_rt: torch.Tensor) -> torch.Tensor:
        """
        功能描述:
            转交给中位数底座构建推理中位数及训练留一法中位数。
        参数:
            train_info：torch.Tensor，训练索引，形状为 [N, 3]。
            train_rt：torch.Tensor，训练响应时间，形状为 [N, 1]。
        返回值:
            torch.Tensor：推理中位数，形状为 [user_num, service_num]。
        """
        return self.anchor_layer.build_anchor(train_info, train_rt)
# -------------------------------初始化---------------------------------------------
    ## 按训练或推理状态查询同一套中位数底座。
    def get_anchor(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        功能描述:
            转交中位数底座查询：训练使用留一法缓存，推理使用全部训练观测的中位数。
        参数:
            user_indices：torch.Tensor，用户索引，形状为 [B]。
            service_indices：torch.Tensor，服务索引，形状为 [B]。
            time_indices：torch.Tensor，时间索引，形状为 [B]。
        返回值:
            torch.Tensor：本次查询的中位数底座，形状为 [B]。
        """
        return self.anchor_layer.get_anchor(user_indices, service_indices, time_indices)

    # 查询与中位数一致的配对可靠性统计。
    def get_pair_statistics(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        功能描述:
            转交中位数底座查询组合中位数和对数观测数量。
        参数:
            user_indices：torch.Tensor，用户索引，形状为 [B]。
            service_indices：torch.Tensor，服务索引，形状为 [B]。
            time_indices：torch.Tensor，时间索引，形状为 [B]。
        返回值:
            torch.Tensor：顺序为中位数、log(观测数的统计+1)，形状为 [B, 2]。
        """
        return self.anchor_layer.get_pair_statistics(user_indices, service_indices, time_indices)

    # 融合 ID、MF 与配对统计，并返回同一次查询使用的中位数底座。
    def forward(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        功能描述:
            查询中位数底座与配对统计，将 ID 以门控残差方式补充至 MF 表示，
            三个 Token 内部融合后返回中位数底座。

        Input Shape:
            user_indices：当前 batch 的用户索引，·[batch_size]。
            service_indices：当前 batch 的服务索引，·[batch_size]。
            time_indices：当前 batch 的时间索引，·[batch_size]。

        Output Shape:
            tokens：静态 Token，·[batch_size, 3, config.e]。
            valid_mask：有效性掩码，·[batch_size, 3]。
            anchor_value：中位数残差底座，·[batch_size]。
        """
        # 对齐当前模块所在 device
        device = self.anchor_layer.anchor.device
        user_idx = as_index(user_indices, device)
        service_idx = as_index(service_indices, device)
        time_idx = as_index(time_indices, device)
        # 数据合法性检查
        if user_idx.ndim != 1 or service_idx.shape != user_idx.shape or time_idx.shape != user_idx.shape:
            raise ValueError("用户、服务和时间索引必须是相同长度的一维张量。")

        # 中位数底座：取对应的中位数锚和配对统计 Token。
        anchor_value, pair_token = self.anchor_layer(user_idx, service_idx, time_idx)
        # MF feature
        user_mf, service_mf = self.mf(user_idx, service_idx)
        # ID feature + 门控残差
        user_token, service_token = self.id_feature(user_idx, service_idx, user_mf, service_mf)

        tokens = torch.stack([user_token, service_token, pair_token], dim=1)
        valid_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
        return self.fusion(tokens, valid_mask), valid_mask, anchor_value# 融合后的三个token,掩码,锚
