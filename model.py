import torch
from torch import nn

from config import Config
from modules.anchor import MedianAnchor
from modules.time_independent import Time_Independent_Feature
from modules.time_dependent import Time_Dependent_Feature
from modules.semantic import Semantic_Feature
from modules.predictor import Predictor
from util.tensor_utils import as_index




class Model(nn.Module):# 完整模型（Model）
    """
    完整模型。

    功能描述:
        将已启用模块内部融合后的 Token 和有效性掩码送入预测器，
        最终预测 = (u, s) 中位数锚 + 主干修正量 Δ；全部关闭时仅返回中位数锚。

    Input Shape:
        user_indices：当前 batch 的用户索引，·[batch_size]。
        service_indices：当前 batch 的服务索引，·[batch_size]。
        time_indices：当前 batch 的时间片索引，·[batch_size]。

    Output Shape:
        prediction：预测结果，·[batch_size]。
    """

    # 按开关初始化三个并列模块，所有组合共用原有中位数底座。
    def __init__(self, config: Config) -> None:
        """
        功能描述:
            只创建启用的模块；静态模块关闭时单独维护不含可训练参数的中位数底座。

        Input Shape:
            config：配置对象，·[标量]。

        Output Shape:
            return：无返回值，·[无]。
        """
        super().__init__()
        self.config = config
        # 1.静态模块关闭时，只保留所有实验共享的中位数底座。
        self.time_independent = Time_Independent_Feature(config) if config.use_time_independent else None
        self.baseline_anchor = MedianAnchor(config, enable_token=False) if self.time_independent is None else None
        # 2.时间相关模块先融合绝对时间和 Local，再输出其有效 Token。
        self.time_dependent = Time_Dependent_Feature(config) if config.use_time_dependent else None
        # 3.语义模块用 Qwen 选择跨实体观测，聚合残差后组合为两个语义 Token。
        self.semantic = Semantic_Feature(config) if config.use_semantic else None
        # 4.无特征 Token 时直接使用中位数，不创建空序列预测器。
        self.predictor = Predictor(config) if config.predictor_total_tokens > 0 else None

    # 统一访问当前启用的中位数缓存，不重复注册同一个子模块。
    @property
    def anchor_layer(self) -> MedianAnchor:
        """
        功能描述:
            静态模块启用时使用其内部底座，否则使用独立的纯统计底座。
        参数:
            无显式参数；读取模型的静态模块和独立底座。
        返回值:
            MedianAnchor：保存全量与留一法中位数缓存的子模块。
        """
        if self.time_independent is not None:
            return self.time_independent.anchor_layer
        return self.baseline_anchor

    ## 构建参考矩阵
    def build_reference_matrix(
        self,
        train_info: torch.Tensor,
        train_rt: torch.Tensor,
        test_info: torch.Tensor | None = None,
        test_rt: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        功能描述:
            训练前调用一次：构建与时间无关的参考矩阵，并训练矩阵分解、写回用户/服务隐向量。
            直接转交给 Time_Independent_Feature，方便外部只持有 Model 一个入口。

        Input Shape:
            train_info：[N, 3] LongTensor，每行 [time, user, service]。
            train_rt：  [N, 1] FloatTensor，响应时间。
            test_info： 可选，[M, 3] LongTensor；只在训练矩阵分解时用于打印测试指标。
            test_rt：   可选，[M, 1] FloatTensor。

        Output Shape:
            reference_matrix：[user_num, service_num] FloatTensor，矩阵分解补全后的参考矩阵。
        """
        if self.time_independent is None:
            raise RuntimeError("time-independent 已关闭，无需构建 MF 参考矩阵。")
        return self.time_independent.build_reference_matrix(train_info, train_rt, test_info, test_rt)

    ## 构建局部窗口矩阵
    def build_local_matrix(self, train_matrix: torch.Tensor) -> torch.Tensor:
        """
        功能描述:
            训练前调用一次：把 data/dataset.py 摊平好的训练集 QoS 张量归一化为局部窗口矩阵。
            通过时间相关模块转交给原有 Local 路径，保持外部预处理入口。

        Input Shape:
            train_matrix：[user_num, service_num, time_num] FloatTensor，训练集 rt，缺失位置为 0。

        Output Shape:
            local_matrix：[user_num, service_num, time_num] FloatTensor，归一化后的 QoS 张量。
        """
        if self.time_dependent is None:
            raise RuntimeError("time-dependent 已关闭，无需构建局部窗口矩阵。")
        return self.time_dependent.build_local_matrix(train_matrix)

    # 加载离线文本缓存，恢复 Value 和 Feature 两条语义路径。
    def build_metadata(
        self, user_text: list[list[str]], service_text: list[list[str]], cache_path: str | None = None
    ) -> None:
        """
        功能描述:
            训练前调用一次：校验并加载离线文本缓存，构建原有经纬度特征。
            直接转交给 Semantic_Feature，方便外部只持有 Model 一个入口。

        参数:
            user_text：list[list[str]]，用户九列元数据，形状为 [user_num, 9]。
            service_text：list[list[str]]，服务九列元数据，形状为 [service_num, 9]。
            cache_path：str | None，可选语义缓存路径，标量；默认使用配置中的路径。
        返回值:
            None：语义模块的运行时元数据加载完成。
        """
        if self.semantic is None:
            raise RuntimeError("semantic 已关闭，无需加载语义元数据或缓存。")
        self.semantic.build_metadata(user_text, service_text, cache_path)

    # 构建中位数锚
    def build_anchor(self, train_info: torch.Tensor, train_rt: torch.Tensor) -> torch.Tensor:
        """
        功能描述:
            使用训练观测构建中位数及计数，并独立准备语义检索支持库。
        参数:
            train_info：torch.Tensor，训练索引，形状为 [N, 3]，每行 [time, user, service]。
            train_rt：torch.Tensor，训练响应时间，形状为 [N, 1]。
        返回值:
            torch.Tensor：所有消融组合共用的推理中位数，形状为 [U, S]。
        """
        if self.semantic is not None:
            self.semantic.retrieval.observations_ready = False
        anchor = self.anchor_layer.build_anchor(train_info, train_rt)
        if self.semantic is not None:
            self.semantic.build_observations(
                train_info, train_rt, self.anchor_layer.anchor_loo, self.anchor_layer.pair_counts,
            )
        return anchor

    # 提供与消融组合无关的中位数查询，供前向和训练残差统计共用。
    def get_anchor(
        self, user_indices: torch.Tensor, service_indices: torch.Tensor, time_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        功能描述:
            对齐索引设备后查询中位数底座，训练使用留一法，推理使用全部训练观测。
        参数:
            user_indices：torch.Tensor，用户索引，形状为 [B]。
            service_indices：torch.Tensor，服务索引，形状为 [B]。
            time_indices：torch.Tensor，时间索引，形状为 [B]。
        返回值:
            torch.Tensor：与当前训练或推理状态一致的中位数，形状为 [B]。
        """
        device = self.anchor_layer.anchor.device
        return self.anchor_layer.get_anchor(
            as_index(user_indices, device), as_index(service_indices, device), as_index(time_indices, device),
        )

    # 拼接已启用模块内部融合后的结果，并叠加中位数残差底座。
    def forward(
        self,
        user_indices: torch.Tensor,
        service_indices: torch.Tensor,
        time_indices: torch.Tensor,
        *,
        indices_validated: bool = False,
        return_anchor: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        功能描述:
            并列计算已启用模块的内部融合结果，将全部 Token 和掩码交给预测器。
            最终预测 = 共享中位数底座 + 跨模块修正量 Δ；全部关闭时修正量为零。
        Input Shape:
            user_indices：当前 batch 的用户索引，·[batch_size]。
            service_indices：当前 batch 的服务索引，·[batch_size]。
            time_indices：当前 batch 的时间片索引，·[batch_size]。

        Output Shape:
            prediction：默认返回预测结果，·[batch_size]。
            return_anchor=True 时同时返回本次前向的中位数，供训练指标复用。
            indices_validated=True 仅供已校验索引范围的数据入口使用，跳过重复范围检查。
        """
        device = self.anchor_layer.anchor.device
        user_idx = as_index(user_indices, device)# [batch_size]
        service_idx = as_index(service_indices, device)# [batch_size]
        time_idx = as_index(time_indices, device)# [batch_size]

        # 时间无关模块：关闭时移除三个静态 Token，仅查询共享中位数底座。
        token_parts, mask_parts = [], []
        if self.time_independent is not None:
            static_tokens, static_mask, anchor_value = self.time_independent(user_idx, service_idx, time_idx)
            token_parts.append(static_tokens)# [B, 3, E]，依次为用户、服务、配对
            mask_parts.append(static_mask)
        else:
            anchor_value = self.anchor_layer.get_anchor(user_idx, service_idx, time_idx)

        # 全部关闭时直接返回底座，不向注意力传入空序列。
        if self.predictor is None:
            return (anchor_value, anchor_value) if return_anchor else anchor_value

        # 时间相关模块：同时保留或移除绝对时间、局部窗口两个 Token。
        if self.time_dependent is not None:
            dynamic_tokens, dynamic_mask = self.time_dependent(
                user_idx, service_idx, time_idx, indices_validated=indices_validated,
            )
            token_parts.append(dynamic_tokens)# [B, 2, E]
            mask_parts.append(dynamic_mask)

        # 语义模块：用户邻居、服务邻居两条检索路径各输出一个时间相关 Token。
        if self.semantic is not None:
            semantic_tokens, semantic_mask = self.semantic(
                user_idx, service_idx, time_idx, indices_validated=indices_validated,
            )
            token_parts.append(semantic_tokens)# [B, 2, E]，依次为用户邻居、服务邻居
            mask_parts.append(semantic_mask)

        # 顺序固定为静态、动态、语义，Token 数量由各模块实际输出相加。
        tokens = torch.cat(token_parts, dim=1)# [batch_size, cfg.predictor_total_tokens, config.e]
        valid_mask = torch.cat(mask_parts, dim=1)# [batch_size, cfg.predictor_total_tokens]
        delta = self.predictor(tokens, valid_mask).squeeze(-1)# [batch_size]
        prediction = anchor_value + delta
        return (prediction, anchor_value) if return_anchor else prediction
