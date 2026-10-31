# train.py
from time import monotonic

import torch
from torch import nn
from tqdm import tqdm

from config import Config
from data.dataset import load_data_split, get_dataloader, load_semantic_text
from model import Model
from util.log_utils import console_output, set_log_path


class _BatchMetrics:
    """延迟批量读回指标，保留原先逐 batch 的 Python 双精度累加顺序。"""

    def __init__(self, count: int):
        self._totals = [0.0] * count
        self._pending = []

    def add(self, *values: torch.Tensor) -> None:
        self._pending.append(torch.stack(values).detach())

    def totals(self) -> tuple[float, ...]:
        if self._pending:
            for batch in torch.stack(self._pending).cpu().tolist():
                for index, value in enumerate(batch):
                    self._totals[index] += value
            self._pending.clear()
        return tuple(self._totals)


def build_criterion(cfg: Config) -> nn.Module:
    """
    功能描述:
        按 cfg.loss_type 构造训练损失。MSE 对齐均方误差（RMSE 最优），
        Huber / L1 对齐中位数（MAE 最优，本数据 rt 重尾，中位数比均值好约 16.6%）。

    Input Shape:
        cfg：配置对象，·[标量]。

    Output Shape:
        criterion：损失函数，·[标量]。
    """
    if cfg.loss_type == "mse":
        return nn.MSELoss()
    if cfg.loss_type == "huber":
        return nn.HuberLoss(delta=cfg.huber_delta)
    if cfg.loss_type == "l1":
        return nn.L1Loss()
    raise ValueError(f"不支持的 loss_type={cfg.loss_type!r}，可选 mse / huber / l1。")


def build_optimizer(model: Model, cfg: Config) -> torch.optim.Optimizer | None:
    """
    功能描述:
        将 ID、时间向量、语义投影和其余参数分组，分别设置权重衰减并构造 Adam 优化器。

    Input Shape:
        model：待训练的完整模型，·[标量]。
        cfg：学习率及各组权重衰减配置，·[标量]。

    Output Shape:
        optimizer：包含已启用参数组的 Adam 优化器，·[标量]；纯中位数基线返回 None。
    """
    # 各模块显式指定需要正则的参数；用参数 ID 标识，避免依赖参数名称。
    time_independent_decay = (
        {id(param) for param in model.time_independent.regularized_parameters()}
        if model.time_independent is not None else set()
    )
    time_dependent_decay = (
        {id(param) for param in model.time_dependent.regularized_parameters()}
        if model.time_dependent is not None else set()
    )
    semantic_decay = (
        {id(param) for param in model.semantic.regularized_parameters()}
        if model.semantic is not None else set()
    )

    # 语义投影只衰减线性权重，偏置、LayerNorm 和其他路径沿用原有分组。
    id_params, time_params, semantic_params, no_decay_params = [], [], [], []
    for param in model.parameters():
        if id(param) in time_independent_decay:
            id_params.append(param)
        elif id(param) in time_dependent_decay:
            time_params.append(param)
        elif id(param) in semantic_decay:
            semantic_params.append(param)
        else:
            no_decay_params.append(param)

    # 只注册实际存在的参数组；全部关闭时没有可训练参数。
    groups = []
    if no_decay_params:
        groups.append({"params": no_decay_params})
    if id_params:
        groups.append({"params": id_params, "weight_decay": cfg.id_embedding_weight_decay})
    if time_params:
        groups.append({"params": time_params, "weight_decay": cfg.temporal_weight_decay})
    if semantic_params:
        groups.append({"params": semantic_params, "weight_decay": cfg.semantic_weight_decay})
    if not groups:
        return None
    # 所有参数组共享主训练学习率，正则强度分别由对应配置控制。
    return torch.optim.Adam(groups, lr=cfg.learning_rate)


def evaluate(model: nn.Module, test_loader, device: torch.device, *, indices_validated: bool = False):
    """
    功能描述:
        在测试集上做一次评估：逐 batch 预测，累加平方误差与绝对误差。

    Input Shape:
        model：待评估的模型，·[标量]。
        test_loader：测试集 DataLoader，每个 batch 为 (info, rt)。
            info：[batch_size, 3] LongTensor，每行 [time, user, service]。
            rt：  [batch_size, 1] FloatTensor。
        device：评估所用设备，·[标量]。

    Output Shape:
        mse：均方误差，·[标量]。
        rmse：均方根误差，·[标量]。
        mae：平均绝对误差，·[标量]。
    """
    model.eval()# 关闭 dropout
    metrics = _BatchMetrics(2)
    num = 0
    model_kwargs = {"indices_validated": True} if indices_validated else {}
    total_batches = len(test_loader)
    # 测试进度条：跑完即消失（leave=False），避免每个 epoch 堆一条
    progress = tqdm(test_loader, desc="[Test]", unit="batch", dynamic_ncols=True, leave=False)
    last_postfix = monotonic()
    with torch.no_grad():
        for batch_index, (info, rt) in enumerate(progress, 1):
            # 转移到device
            info = info.to(device, non_blocking=True)
            rt = rt.to(device, non_blocking=True).squeeze(-1)# [batch_size]
            # 前向传播
            prediction = model(info[:, 1], info[:, 2], info[:, 0], **model_kwargs)# 用户索引、服务索引、时间片索引 → [batch_size]
            error = prediction - rt# [batch_size]
            # 累加平方误差与绝对误差
            metrics.add(error.pow(2).sum(), error.abs().sum())
            num += rt.numel()
            # 仅在进度条刷新时读回指标，最后一个 batch 始终更新最终值。
            now = monotonic()
            if batch_index == total_batches or now - last_postfix >= progress.mininterval:
                squared_sum, absolute_sum = metrics.totals()
                progress.set_postfix(mse=f"{squared_sum / num:.6f}", mae=f"{absolute_sum / num:.6f}", refresh=False)
                last_postfix = now
    squared_sum, absolute_sum = metrics.totals()
    mse = squared_sum / num
    return mse, mse ** 0.5, absolute_sum / num


# 加载离线特征和训练数据，再按现有损失与优化器训练主模型。
def main():
    """
    功能描述:
        加载配置与数据，构建特征和模型，完成训练并逐轮评估。

    Input Shape:
        无函数参数；训练选项由命令行和 Config 提供。

    Output Shape:
        return：无返回值；训练过程和评估结果写入日志。
    """
    # ---------- 1. 加载配置 ----------
    cfg = Config()# 实例化参数
    cfg.parse_args()# 解析命令行参数
    set_log_path(cfg.log_path)# 绑定日志文件地址，整个进程只调用一次
    console_output("[config]加载配置...")
    cfg.printf()# 打印配置项

    # ---------- 2. 固定随机种子与设备 ----------
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    console_output(f"[device]使用设备 {device}。")

    # ---------- 3. 加载数据 ----------
    console_output("[data]加载并划分数据集...")
    train_info, train_rt, test_info, test_rt, train_matrix = load_data_split(cfg)
    console_output(f"   [data/DataSet]训练集 {tuple(train_info.shape)}，测试集 {tuple(test_info.shape)}。")
    train_loader, test_loader = get_dataloader(train_info, train_rt, test_info, test_rt, cfg)
    console_output(f"   [data/DataLoader]训练集 {len(train_loader)} 个 batch，测试集 {len(test_loader)} 个 batch。")

    # ---------- 4. 构建模型 ----------
    console_output("[model]构建模型...")
    model = Model(cfg).to(device)
    console_output(f"[model]模型构建完毕，模型所在设备 {model.anchor_layer.anchor.device}。")
    console_output(
        f"[model/tokens]time-independent={cfg.time_independent_tokens}，"
        f"time-dependent={cfg.time_dependent_tokens}，semantic={cfg.semantic_tokens}，"
        f"回归头输入 {cfg.predictor_total_tokens} 个 Token。"
    )
    console_output(
        f"[model/features]time-independent={'开启' if cfg.use_time_independent else '关闭'}，"
        f"time-dependent={'开启' if cfg.use_time_dependent else '关闭'}，"
        f"semantic={'开启' if cfg.use_semantic else '关闭'}；中位数残差底座始终保留。"
    )

    # ---------- 5. 预处理 ----------
    # 优先校验语义缓存，避免缺失缓存时先执行耗时的矩阵分解训练。
    if model.semantic is not None:
        console_output(f"[model/semantic]读取离线语义缓存：{cfg.resolved_semantic_cache_path}")
        user_text, service_text = load_semantic_text(cfg)
        model.build_metadata(user_text, service_text)
        console_output(
            f"[model/semantic]缓存加载完毕，用户 {len(user_text)} 行、服务 {len(service_text)} 行，"
            f"编码器 {model.semantic.cache_metadata['encoder_id']}，"
            f"文本向量 {model.semantic.cache_metadata['embedding_dim']} 维。"
        )
    else:
        console_output("[model/semantic]已关闭，跳过元数据和语义缓存加载。")
    # 参考矩阵 B 与 MF 训练只服务于时间无关特征支路。
    if model.time_independent is not None:
        console_output("[model/time_independent_MF]构建与时间无关的参考矩阵...")
        reference_matrix = model.build_reference_matrix(train_info, train_rt, test_info, test_rt)
        console_output(f"[model/time_independent_MF]参考矩阵构建完毕，形状 {tuple(reference_matrix.shape)}。")
    else:
        console_output("[model/time_independent_MF]已关闭，跳过 MF 预训练。")
    # 局部窗口矩阵直接复用 dataset.py 构建的训练集 QoS 张量。
    if model.time_dependent is not None:
        console_output("[model/time_dependent/local]构建局部窗口矩阵...")
        local_matrix = model.build_local_matrix(train_matrix)
        console_output(f"[model/time_dependent/local]局部窗口矩阵构建完毕，形状 {tuple(local_matrix.shape)}。")
    else:
        console_output("[model/time_dependent/local]已关闭，跳过局部训练矩阵构建。")
    # 构建推理使用的全量中位数锚和训练使用的留一法锚。
    console_output("[model/anchor]构建共享的全量中位数、留一法中位数及观测数量...")
    anchor = model.build_anchor(train_info, train_rt)
    console_output(f"[model/anchor]锚构建完毕，形状 {tuple(anchor.shape)}，训练使用留一法锚，预测 = 锚 + Δ。")
    # 检索支持库随共享中位数由训练集独立构建，关闭 Local 不会影响跨实体检索。
    if model.semantic is not None:
        support_count = int(model.semantic.retrieval.support_mask.sum().item())
        console_output(
            f"[model/semantic_retrieval]训练支持观测 {support_count} 条，"
            f"每侧近邻 K={cfg.semantic_neighbors}（边界并列全部保留），"
            f"时间半径={cfg.semantic_time_radius}，temperature={cfg.semantic_temperature}，"
            f"time_decay={cfg.semantic_time_decay}；排除目标配对和单观测配对。"
        )
    else:
        console_output("[model/semantic_retrieval]已关闭，跳过跨实体支持库构建。")

    # ---------- 6. 训练 ----------
    optimizer = build_optimizer(model, cfg)
    # 纯统计基线只评估一次，避免空优化器和无梯度反向传播。
    if optimizer is None:
        console_output("[train]三大模块均已关闭，仅评估中位数底座，不执行参数训练。")
        _, rmse, mae = evaluate(model, test_loader, device, indices_validated=True)
        console_output(f"[baseline/Eval]test_rmse={rmse:.6f} test_mae={mae:.6f} | delta_mean=+0.0000 delta_std=0.0000")
        console_output("[train]基线评估结束。")
        return
    criterion = build_criterion(cfg)
    console_output(
        f"[train]损失函数 {cfg.loss_type}"
        f"{f'(delta={cfg.huber_delta})' if cfg.loss_type == 'huber' else ''}，"
        f"id embedding weight_decay={cfg.id_embedding_weight_decay}，"
        f"temporal weight_decay={cfg.temporal_weight_decay}，dropout={cfg.predictor_dropout}，"
        f"semantic weight_decay={cfg.semantic_weight_decay}，text dropout={cfg.semantic_text_dropout}。"
    )
    console_output("[train]开始训练...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()# 打开 dropout
        metrics = _BatchMetrics(4)
        loss_sum = torch.zeros((), device=device)# 检索评分与主预测共同使用原有 QoS 损失
        num = 0
        # 训练进度条：每轮结束后保留最终显示
        progress = tqdm(
            train_loader,
            desc=f"[Train]{epoch}/{cfg.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True
        )
        last_postfix = monotonic()
        for batch_index, (info, rt) in enumerate(progress, 1):
            # 转移device
            info = info.to(device, non_blocking=True)# [batch_size, 3]
            rt = rt.to(device, non_blocking=True).squeeze(-1)# [batch_size]
            # 前向传播
            # load_data_split 已校验所有索引范围；复用同次前向的锚，避免再查一次。
            prediction, anchor_value = model(
                info[:, 1], info[:, 2], info[:, 0], indices_validated=True, return_anchor=True,
            )# [batch_size], [batch_size]
            # 反向传播
            loss = criterion(prediction, rt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累加误差与样本数：指标一律按“平方误差 / 绝对误差”累计，与训练损失无关
            with torch.no_grad():
                loss_sum += loss.detach() * rt.numel()
                error = prediction.detach() - rt
                # 减去本次前向使用的留一法训练锚，统计主干修正量。
                delta = prediction.detach() - anchor_value
                metrics.add(error.pow(2).sum(), error.abs().sum(), delta.sum(), delta.pow(2).sum())
            num += rt.numel()# 总样本数
            now = monotonic()
            if batch_index == len(train_loader) or now - last_postfix >= progress.mininterval:
                squared_sum, _, _, _ = metrics.totals()
                progress.set_postfix(mse=f"{squared_sum / num:.6f}", refresh=False)
                last_postfix = now

        # ---------- 7. 测试 ----------
        squared_sum, absolute_sum, delta_sum, delta_sq_sum = metrics.totals()
        mse, rmse, mae = evaluate(model, test_loader, device, indices_validated=True)
        delta_mean = delta_sum / num
        delta_std = max(delta_sq_sum / num - delta_mean ** 2, 0.0) ** 0.5# Δ 的均值与标准差
        mean_loss = (loss_sum / num).item()
        console_output(
            f"   [train/Eval]{epoch}/{cfg.epochs}: train_rmse={(squared_sum / num) ** 0.5:.6f}"
            f" train_mae={absolute_sum / num:.6f} | test_rmse={rmse:.6f} test_mae={mae:.6f}"
            f" | delta_mean={delta_mean:+.4f} delta_std={delta_std:.4f}"
            f" | main_loss={mean_loss:.6f}"
        )

    console_output("[train]训练结束。")


if __name__ == "__main__":
    main()
