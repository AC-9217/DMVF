# config.py
from dataclasses import dataclass, field
import os
import math
import torch
import argparse

from util.log_utils import console_output
from data.semantic_cache import CACHE_FILENAME



@dataclass
class Config:
    # ---------- DataLoader ----------
    # 位置
    data_path: str = "./dataset/Supplement_WSDREAM_dataset2/rtdata_cache2.csv"
    # 训练集比例
    train_ratio: float = 0.1
    # 数据集维度
    user_num: int = 140
    service_num: int = 1000
    time_num: int = 96
    # 训练参数
    batch_size: int = 8192# 1024
    num_workers: int = 0
    epochs: int = 50
    learning_rate: float = 1e-3

    # ---------- 日志 ----------
    description: str = "run"
    log_dir: str = "./log"
    @property
    def log_path(self) -> str:
        """日志文件路径。"""
        return f"{self.log_dir}/{self.description}_{self.train_ratio}_{self.batch_size}_{self.learning_rate}.log"

    # ---------- 设备 ----------
    device:str = "cuda:0"

    # --------- 随机 ----------
    seed: int = 42

    # ---------- 模型 ----------
    # Token 维度
    e: int = 64

    # ---------- time_independent ----------
    use_time_independent: bool = True  # 控制 MF、ID 和静态 Token，保留共享中位数底座
    id_gate_init: float = 0.1  # ID 残差逐维门控的初始值
    mf_rank: int = 16   # 隐向量维度
    mf_hidden: int = 128 # 隐向量映射到 Token 维度的隐藏层维度
    mf_epochs: int = 200  # 矩阵分解迭代轮数；实际值由下面的 MF_EPOCHS_BY_DENSITY 按密度覆盖
    mf_lr: float = 0.01    # 矩阵分解学习率
    mf_reg: float = 0.01    # 隐向量与偏置的 L2 正则系数
    mf_init_std: float = 0.01    # 隐向量初始化标准差
    # 密度 → 矩阵分解轮数（train_ratio 上界, 轮数）：密度越低 MF 越容易过拟合。
    # 实测 V0.3.1：0.05 密度下 test_mse 在 100 轮触底（0.5410），200 轮反而 0.6477（差 19.6%）；
    # 0.10 在 140 轮触底（0.4438）；0.15 / 0.20 到 200 轮仍在下降，保持 200。
    MF_EPOCHS_BY_DENSITY = ((0.05, 100), (0.10, 140), (1.00, 200))

    # --------- time_dependent ----------
    use_time_dependent: bool = True  # 同时控制绝对时间和 Local 两条路径
    # local
    local_windows: int = 16  # 局部窗口半径 k，取目标时间片前后各 k 个时间片做摘要统计
    local_tokens: int = 1    # 局部支路输出的 Token 数：P2.1 之后固定为 1 个窗口摘要 Token
    # temporal
    time_dim: int = 16                    # 时间查表向量维度，投影到 e 后作为一个 Token
    temporal_init_std: float = 0.1        # 绝对时间查表向量的初始化标准差
    temporal_weight_decay: float = 1e-4   # 仅作用于绝对时间查表向量的权重衰减

    # --------- semantic ----------
    use_semantic: bool = True  # 控制离线 Qwen 邻居检索及经纬度表示
    semantic_dim: int = 16          # ID 与原有经纬度编码的输出维度
    semantic_ip_octets: int = 2     # 英文描述中的 IP 前缀段数，必须与离线缓存配置一致
    semantic_geo_freqs: int = 8     # 经纬度正弦编码的频率个数 R：输入 4R+1 维（含缺失标志），再 Linear 到 semantic_dim
    semantic_cache_path: str = ""  # 空串时从 data_path 所在目录读取默认语义缓存
    semantic_text_hidden: int = 128  # 邻居评分 Adapter 的隐藏维度，输出为 e 维
    semantic_text_dropout: float = 0.1  # 文本 Adapter 隐藏层的丢弃概率
    semantic_weight_decay: float = 1e-4  # 仅作用于文本评分 Adapter 和检索统计投影的线性权重
    semantic_neighbors: int = 16  # 每侧选取有观测的前 K 个文本近邻，边界并列全部保留
    semantic_time_radius: int = 0  # 检索目标时间片及其前后各若干时间片
    semantic_temperature: float = 0.1  # 邻居分数温度，必须大于零
    semantic_time_decay: float = 1.0  # 每相隔一个时间片扣除的聚合分数，零表示不衰减
    # 经纬度网格编号已不再使用（改用上面的正弦编码），dataset/preprocess_semantic.py 的最后一列保留不影响运行

    # --------- 训练目标与正则 ----------
    loss_type: str = "huber"        # 训练损失：mse / huber / l1（huber 与 l1 对齐 MAE，mse 对齐 RMSE）
    huber_delta: float = 1.0        # huber 的切换点 delta
    id_embedding_weight_decay: float = 0.01  # 只加在 user/service 的 id embedding 上（它们最容易记训练集噪声）

    # ---------- predictor ----------
    predictor_layers: int = 2         # 多头自注意力的层数
    predictor_heads: int = 4          # 多头自注意力的头数（需能整除 e）
    predictor_ffn_dim: int = 128      # 每层前馈网络的隐藏维度
    predictor_dropout: float = 0.25   # 注意力与前馈网络的 dropout 概率（抗过拟合，从 0.1 提到 0.25）
    predictor_mlp_hidden: int = 128   # 回归 MLP 的隐藏维度
    predictor_out_dim: int = 1        # 预测结果个数（暂定 1 个)

    @property
    def time_independent_tokens(self) -> int: # 计算time-independent模块token数量
        return 3 if self.use_time_independent else 0# 用户、服务和配对三个静态 Token

    @property
    def time_dependent_tokens(self) -> int: # 计算time-dependent模块token数量
        return 1 + self.local_tokens if self.use_time_dependent else 0

    @property
    def semantic_tokens(self) -> int: # 计算semantic模块token数量
        return 2 if self.use_semantic else 0

    @property
    def predictor_total_tokens(self) -> int: # 计算三大模块总token数量
        return self.time_independent_tokens + self.time_dependent_tokens + self.semantic_tokens

    @property
    def predictor_module_ids(self) -> tuple[int, ...]:# 按照所属的模块，给token编号
        """
        功能描述:
            标记每个输出 Token 所属的大模块。
        返回值:
            tuple[int, ...]：长度为 predictor_total_tokens 的模块编号。
            0:time_independent  1:time_dependent    2:semantic
        """
        return (0,) * self.time_independent_tokens + (1,) * self.time_dependent_tokens + (2,) * self.semantic_tokens

    @property
    def predictor_role_ids(self) -> tuple[int, ...]:# 按三个模块的 Token 顺序生成实体和时间角色编号。
        """
        功能描述:
            用零至四分别标记用户、服务、配对、绝对时间和局部窗口。
        参数:
            无显式参数；读取三个模块的开关、角色顺序及局部 Token 数量。
        返回值:
            tuple[int, ...]：长度为 predictor_total_tokens 的角色编号。
        """
        # 保留角色编号含义，只移除已关闭模块对应的位置。
        static_roles = (0, 1, 2) if self.use_time_independent else ()
        dynamic_roles = (3,) + (4,) * self.local_tokens if self.use_time_dependent else ()
        semantic_roles = (0, 1) if self.use_semantic else ()
        return static_roles + dynamic_roles + semantic_roles


    # 将缓存路径与当前数据集绑定，支持显式覆盖路径。
    @property
    def resolved_semantic_cache_path(self) -> str:
        return self.semantic_cache_path or os.path.join(os.path.dirname(self.data_path), CACHE_FILENAME)

    # 打印当前配置，供训练日志记录。
    def printf(self):
        """打印所有配置项（走 console_output，同步写入日志文件）。"""
        console_output("配置项:", show_time=False)
        for key, value in self.__dict__.items():
            console_output(f"  {key}: {value}", show_time=False)

    # 解析命令行参数
    def parse_args(self):
        """解析命令行参数。"""
        # ---------- 接收参数 ----------#
        parser = argparse.ArgumentParser(description="训练模型的配置参数。")
        parser.add_argument(
            "--description",
            type=str,
            default="",
            help="实验描述，默认为空。",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="设备，默认为 cuda:0。",
        )
        parser.add_argument("--data-path", default=self.data_path, help="响应时间数据文件路径。")
        parser.add_argument(
            "--semantic-cache", "--semantic-cache-path", dest="semantic_cache_path",
            default=self.semantic_cache_path, help="独立预处理脚本生成的语义缓存路径。",
        )
        parser.add_argument("--semantic-text-hidden", type=int, default=self.semantic_text_hidden, help="文本投影隐藏维度。")
        parser.add_argument("--semantic-text-dropout", type=float, default=self.semantic_text_dropout, help="联合文本 Adapter 的丢弃概率。")
        parser.add_argument("--semantic-weight-decay", type=float, default=self.semantic_weight_decay, help="文本评分和检索统计投影线性权重的衰减。")
        parser.add_argument("--semantic-neighbors", type=int, default=self.semantic_neighbors, help="每侧有观测文本近邻数量，边界并列全部保留。")
        parser.add_argument("--semantic-time-radius", type=int, default=self.semantic_time_radius, help="语义检索时间窗口半径，包含目标时间片。")
        parser.add_argument("--semantic-temperature", type=float, default=self.semantic_temperature, help="语义邻居聚合的分数温度。")
        parser.add_argument("--semantic-time-decay", type=float, default=self.semantic_time_decay, help="支持观测每相隔一个时间片的分数惩罚。")
        parser.add_argument("--semantic-ip-octets", type=int, default=self.semantic_ip_octets, help="缓存使用的 IP 前缀段数。")
        parser.add_argument(
            "--train-ratio",
            type=float,
            default=0.1,
            help="训练集比例，默认为 0.1。",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=self.learning_rate,
            help="训练梯度",
        )
        # 补充损失、注意力丢弃率及局部窗口的调参入口。
        parser.add_argument("--huber-delta", type=float, default=self.huber_delta, help="Huber 损失的切换阈值，必须大于 0。")
        parser.add_argument("--predictor-dropout", type=float, default=self.predictor_dropout, help="模块内部及预测头注意力的丢弃概率。")
        parser.add_argument("--local-windows", type=int, default=self.local_windows, help="Local 窗口半径，取目标时间片前后各若干时间片。")
        parser.add_argument("--id-gate-init", type=float, default=self.id_gate_init, help="ID 残差门控初始值。")

        # 模型容量与正则的调参入口：Token 维度、预测器结构、MF 隐向量与 ID embedding 衰减。
        parser.add_argument("--e", type=int, default=self.e, help="Token 维度，需被 predictor_heads 整除。")
        parser.add_argument("--predictor-layers", type=int, default=self.predictor_layers, help="多头自注意力层数。")
        parser.add_argument("--predictor-heads", type=int, default=self.predictor_heads, help="多头自注意力头数，需整除 e。")
        parser.add_argument("--predictor-ffn-dim", type=int, default=self.predictor_ffn_dim, help="每层前馈网络的隐藏维度。")
        parser.add_argument("--predictor-mlp-hidden", type=int, default=self.predictor_mlp_hidden, help="回归 MLP 的隐藏维度。")
        parser.add_argument("--mf-rank", type=int, default=self.mf_rank, help="矩阵分解隐向量维度。")
        parser.add_argument(
            "--id-embedding-weight-decay", type=float, default=self.id_embedding_weight_decay,
            help="只加在 user/service 的 id embedding 上的权重衰减。",
        )
        parser.add_argument("--time-dim", type=int, default=self.time_dim, help="时间向量维度。")
        parser.add_argument(
            "--temporal-init-std", type=float, default=self.temporal_init_std,
            help="绝对时间查表向量的初始化标准差，必须大于 0。",
        )
        parser.add_argument(
            "--temporal-weight-decay", type=float, default=self.temporal_weight_decay,
            help="绝对时间查表向量的权重衰减。",
        )

        # 三个模块默认开启，消融选项可以组合使用。
        parser.add_argument(
            "--no-time-independent", dest="use_time_independent", action="store_false",
            default=self.use_time_independent, help="关闭 MF、ID 和静态 Token，保留中位数残差底座。",
        )
        parser.add_argument(
            "--no-time-dependent", dest="use_time_dependent", action="store_false",
            default=self.use_time_dependent, help="关闭绝对时间和 Local，不构建局部训练矩阵。",
        )
        parser.add_argument(
            "--no-semantic", dest="use_semantic", action="store_false",
            default=self.use_semantic, help="关闭经纬度和文本语义，不读取元数据或 Qwen 缓存。",
        )

        # ---------- 应用参数 ----------#
        new_args = parser.parse_args()
        # 替换默认配置项
        if new_args.train_ratio is not None:
            self.train_ratio = new_args.train_ratio
        if new_args.description is not None:
            self.description = new_args.description
        if new_args.device is not None:
            self.device = new_args.device
        self.learning_rate=new_args.learning_rate
        self.huber_delta = new_args.huber_delta
        self.predictor_dropout = new_args.predictor_dropout
        self.local_windows = new_args.local_windows
        self.data_path = new_args.data_path
        self.semantic_cache_path = new_args.semantic_cache_path
        self.semantic_text_hidden = new_args.semantic_text_hidden
        self.semantic_text_dropout = new_args.semantic_text_dropout
        self.semantic_weight_decay = new_args.semantic_weight_decay
        self.semantic_neighbors = new_args.semantic_neighbors
        self.semantic_time_radius = new_args.semantic_time_radius
        self.semantic_temperature = new_args.semantic_temperature
        self.semantic_time_decay = new_args.semantic_time_decay
        self.semantic_ip_octets = new_args.semantic_ip_octets
        self.id_gate_init = new_args.id_gate_init
        self.e = new_args.e
        self.predictor_layers = new_args.predictor_layers
        self.predictor_heads = new_args.predictor_heads
        self.predictor_ffn_dim = new_args.predictor_ffn_dim
        self.predictor_mlp_hidden = new_args.predictor_mlp_hidden
        self.mf_rank = new_args.mf_rank
        self.id_embedding_weight_decay = new_args.id_embedding_weight_decay
        self.time_dim = new_args.time_dim
        self.temporal_init_std = new_args.temporal_init_std
        self.temporal_weight_decay = new_args.temporal_weight_decay
        self.use_time_independent = new_args.use_time_independent
        self.use_time_dependent = new_args.use_time_dependent
        self.use_semantic = new_args.use_semantic

        # ---------- 按密度覆盖默认值 ----------#
        # MF 轮数与密度强相关，放在 train_ratio 确定之后再解析，这样 printf 打到的是实际生效的值
        for ratio_upper, epochs in self.MF_EPOCHS_BY_DENSITY:
            if self.train_ratio <= ratio_upper:
                self.mf_epochs = epochs
                break


        # ---------- 校验合法性 ----------#
        # 预测器结构：维度必须为正，且 Token 维度要能被注意力头数整除。
        if self.e < 1:
            raise ValueError("e 必须为正整数。")
        if self.predictor_heads < 1:
            raise ValueError("predictor_heads 必须为正整数。")
        if self.e % self.predictor_heads != 0:
            raise ValueError(f"e={self.e} 必须被 predictor_heads={self.predictor_heads} 整除。")
        if self.predictor_layers < 1:
            raise ValueError("predictor_layers 必须为正整数。")
        if self.predictor_ffn_dim < 1:
            raise ValueError("predictor_ffn_dim 必须为正整数。")
        if self.predictor_mlp_hidden < 1:
            raise ValueError("predictor_mlp_hidden 必须为正整数。")
        # 拒绝无效损失阈值和丢弃概率，避免调参任务启动后才报错。
        if self.loss_type == "huber" and (not math.isfinite(self.huber_delta) or self.huber_delta <= 0):
            raise ValueError("huber_delta 必须为正的有限数值。")
        if not math.isfinite(self.predictor_dropout) or not 0 <= self.predictor_dropout < 1:
            raise ValueError("predictor_dropout 必须为零至一之间的有限数值，且不能等于一。")
        # 只校验启用模块的专属超参数。
        if self.use_time_independent:
            if not math.isfinite(self.id_gate_init) or not 0 < self.id_gate_init < 1:
                raise ValueError("id_gate_init 必须为零至一之间的有限数值，且不能等于端点。")
            if self.mf_rank < 1:
                raise ValueError("mf_rank 必须为正整数。")
            if not math.isfinite(self.id_embedding_weight_decay) or self.id_embedding_weight_decay < 0:
                raise ValueError("id_embedding_weight_decay 必须为非负有限数值。")
        if self.use_semantic:
            if self.semantic_text_hidden < 1:
                raise ValueError("semantic_text_hidden 必须为正整数。")
            if not math.isfinite(self.semantic_text_dropout) or not 0 <= self.semantic_text_dropout < 1:
                raise ValueError("semantic_text_dropout 必须为零至一之间的有限数值，且不能等于一。")
            if not math.isfinite(self.semantic_weight_decay) or self.semantic_weight_decay < 0:
                raise ValueError("semantic_weight_decay 必须为非负有限数值。")
            if self.semantic_neighbors < 1:
                raise ValueError("semantic_neighbors 必须为正整数。")
            if self.semantic_time_radius < 0:
                raise ValueError("semantic_time_radius 必须为非负整数。")
            if not math.isfinite(self.semantic_temperature) or self.semantic_temperature <= 0:
                raise ValueError("semantic_temperature 必须为正的有限数值。")
            if not math.isfinite(self.semantic_time_decay) or self.semantic_time_decay < 0:
                raise ValueError("semantic_time_decay 必须为非负有限数值。")
            if not 1 <= self.semantic_ip_octets <= 4:
                raise ValueError("semantic_ip_octets 必须在 1 至 4 之间。")
        if self.use_time_dependent:
            if self.local_windows < 1:
                raise ValueError("local_windows 必须为正整数。")
            if self.time_dim < 1:
                raise ValueError("time_dim 必须为正整数。")
            if not math.isfinite(self.temporal_init_std) or self.temporal_init_std <= 0:
                raise ValueError("temporal_init_std 必须为有限正数。")
            if not math.isfinite(self.temporal_weight_decay) or self.temporal_weight_decay < 0:
                raise ValueError("temporal_weight_decay 必须为有限非负数。")

        # 日志文件：已存在则报错，不存在则创建（含所需目录）
        if os.path.exists(self.log_path):# 存在则报错
            raise FileExistsError(f"日志文件已存在：{self.log_path}")
        log_dir = os.path.dirname(self.log_path)
        if log_dir:# 创建文件夹（存在不报错）
            os.makedirs(log_dir, exist_ok=True)
        # 创建空日志文件，供 console_output 追加写入
        with open(self.log_path, "w", encoding="utf-8"):
            pass
