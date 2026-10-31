import torch
from torch import nn

from config import Config
from util.log_utils import console_output


# 矩阵分解：压缩时间维度后补全用户-服务参考矩阵，并投影 MF 表示。
class MF(nn.Module):
    """
    功能描述:
        沿时间维度汇总训练 QoS 观测（每个 user-service 组合只保留所有时间片的中位数），
        在压缩后的无时间片数据集上做完整矩阵分解，补全用户-服务交互，生成与时间无关的参考矩阵 B。
        分解得到的用户/服务隐向量保存在 self.user_factors / self.service_factors，
        前向时再投影成 MF 表示，供 ID 残差门控组合。

    Input Shape:
        train_info：训练集索引张量，·[num_observations, 3]，每行 [time, user, service]。
        train_rt：  训练集响应时间，·[num_observations, 1]。
        user_indices：当前 batch 的用户索引，·[batch_size]。
        service_indices：当前 batch 的服务索引，·[batch_size]。

    Output Shape:
        reference_matrix：矩阵分解补全后的参考矩阵，·[user_num, service_num]。
        user_mf：用户 MF 表示，·[batch_size, config.e]。
        service_mf：服务 MF 表示，·[batch_size, config.e]。
    """

    # 初始化冻结的 MF 隐向量缓存及其到 Token 维度的投影层。
    def __init__(self, config: Config) -> None:
        """
        功能描述:
            创建用户/服务隐向量缓存和两条 MF 投影路径。

        Input Shape:
            config：配置对象，·[标量]。

        Output Shape:
            return：无返回值，·[无]。
        """
        super().__init__()
        self.config = config
        # 用户和服务的矩阵分解隐向量（矩阵分解的结果写在这里）
        self.register_buffer("user_factors", torch.zeros(config.user_num, config.mf_rank, dtype=torch.float32), persistent=True)
        self.register_buffer("service_factors", torch.zeros(config.service_num, config.mf_rank, dtype=torch.float32), persistent=True)
        # 2.因向量→Token 的线性投影层
        self.user_projection = nn.Sequential(
            nn.Linear(config.mf_rank, config.mf_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.mf_hidden, config.e),
            )
        self.service_projection = nn.Sequential(
            nn.Linear(config.mf_rank, config.mf_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.mf_hidden, config.e),
            )

    ## 压缩时间维度（中位数）
    def compress_time_dimension(self, train_info: torch.Tensor, train_rt: torch.Tensor):
        """
        功能描述:
            压缩时间维度。同一个 (user, service) 组合在多个时间片上会有多条观测，
            这里把该组合的所有时间片观测汇总成一个值，取中位数。

        Input Shape:
            train_info：[N, 3] LongTensor，每行 [time, user, service]。
            train_rt：  [N, 1] FloatTensor，响应时间。

        Output Shape:
            mf_train_info：[P, 2] LongTensor，每行 [user, service]，P 为用户-服务组合数，按 (user, service) 升序排列。
            mf_train_rt：  [P, 1] FloatTensor，每个组合所有时间片观测的中位数。
        """
        user = train_info[:, 1].long()# 用户 [N]
        service = train_info[:, 2].long()# 服务 [N]
        rt = train_rt.reshape(-1).float().to(user.device)# [N]

        # 1.把 (user, service) 两列编码成唯一组合，inverse 给出每条观测所属的组合编号
        pairs = torch.stack([user, service], dim=1)# [N, 2]
        mf_train_info, inverse = torch.unique(pairs, dim=0, return_inverse=True)# [P, 2], [N]

        # 2.把观测按 (组合编号, rt 升序) 排列，使同一组合的观测连续且按 rt 递增
        #  order = 用户-服务组合编号 拼接 rt 升序名次
        _, rt_order = torch.sort(rt)# [N]
        rt_rank = torch.empty_like(rt_order) # [N]
        rt_rank[rt_order] = torch.arange(rt.numel(), device=rt.device)# [N]
        order = torch.argsort(inverse * rt.numel() + rt_rank)# 用户-服务组合编号 * N + rt 升序名次，保证同一组合的观测连续且按 rt 递增
        rt_sorted = rt[order]# 按order排序后的rt [N]

        # 3.统计每个组合的观测条数与起始位置
        counts = torch.bincount(inverse, minlength=mf_train_info.shape[0])# 统计每个用户-服务对出现次数[P]
        starts = torch.cumsum(counts, dim=0) - counts# 累积计数减去每个组合的观测次数，得到起始位置[P]

        # 4.取中位数：奇数条取正中间那条，偶数条取中间两条的平均
        mid = starts + (counts - 1) // 2# 计算中位数位置
        upper = torch.clamp(mid + 1, max=rt_sorted.numel() - 1)# 偶数的右中位数
        median = torch.where(
            counts % 2 == 0,
            (rt_sorted[mid] + rt_sorted[upper]) / 2,# 偶数个
            rt_sorted[mid]# 奇数个
        )
        mf_train_rt = median.unsqueeze(1)# [P, 1]
        return mf_train_info, mf_train_rt

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
            先在 compress_time_dimension 里压掉时间维度，得到不带时间片的 (user, service) → rt 数据集；
            再在该数据集上做完整的带偏置矩阵分解：
                r ≈ 全局均值 + 用户偏置 + 服务偏置 + <用户隐向量, 服务隐向量>
            用 Adam 只在观测到的 (user, service) 上优化，并对观测条目做 L2 正则。
            分解出的用户/服务隐向量写回 self.user_factors / self.service_factors，供 forward 投影成时间无关 Token。
            训练过程中按固定间隔打印训练集 MSE；若传入测试集，则同时打印测试集的 MSE / RMSE / MAE。
            测试集同样先压缩时间维度，保证与训练集在同一口径下比较。

        Input Shape:
            train_info：[N, 3] LongTensor，每行 [time, user, service]。
            train_rt：  [N, 1] FloatTensor，响应时间。
            test_info： 可选，[M, 3] LongTensor，每行 [time, user, service]；不传则只打印训练集指标。
            test_rt：   可选，[M, 1] FloatTensor，响应时间。

        Output Shape:
            reference_matrix：[user_num, service_num] FloatTensor，矩阵分解补全后的参考矩阵。
        """
        cfg = self.config
        device = torch.device(cfg.device)

        # ---------- 1. 压缩时间维度，得到不带时间片的数据集 ----------
        mf_train_info, mf_train_rt = self.compress_time_dimension(train_info, train_rt)
        user_idx = mf_train_info[:, 0].to(device)# [P]
        service_idx = mf_train_info[:, 1].to(device)# [P]
        rt = mf_train_rt.reshape(-1).to(device)# [P]

        # 测试集用同一口径压缩，只用来算指标；不传测试集时三个变量都为 None
        if test_info is not None and test_rt is not None:
            mf_test_info, mf_test_rt = self.compress_time_dimension(test_info, test_rt)
            test_user_idx = mf_test_info[:, 0].to(device)# [M]
            test_service_idx = mf_test_info[:, 1].to(device)# [M]
            test_rating = mf_test_rt.reshape(-1).to(device)# [M]
        else:
            test_user_idx = None
            test_service_idx = None
            test_rating = None

        # ---------- 2. 参数初始化 ----------
        generator = torch.Generator().manual_seed(cfg.seed)
        # 隐向量
        user_factors = nn.Parameter(
            torch.empty(cfg.user_num, cfg.mf_rank).normal_(0.0, cfg.mf_init_std, generator=generator).to(device)
        )
        service_factors = nn.Parameter(
            torch.empty(cfg.service_num, cfg.mf_rank).normal_(0.0, cfg.mf_init_std, generator=generator).to(device)
        )
        # 偏置
        user_bias = nn.Parameter(torch.zeros(cfg.user_num, device=device))
        service_bias = nn.Parameter(torch.zeros(cfg.service_num, device=device))
        # 均值
        global_mean = rt.mean().detach()# 标量，不参与优化
        # 优化器
        optimizer = torch.optim.Adam([user_factors, service_factors, user_bias, service_bias], lr=cfg.mf_lr)

        # ---------- 3. 训练 ----------
        num_ratings = rt.numel()# 观测条目数
        log_interval = max(1, cfg.mf_epochs // 10)# 全程打印约 10 次
        for epoch in range(1, cfg.mf_epochs + 1):
            optimizer.zero_grad()
            # 取出本批观测条目的隐向量与偏置
            u_vec = user_factors[user_idx]# [P, mf_rank]
            s_vec = service_factors[service_idx]# [P, mf_rank]
            u_bias = user_bias[user_idx]# [P]
            s_bias = service_bias[service_idx]# [P]
            # 预测与平方误差
            prediction = (u_vec * s_vec).sum(dim=1) + u_bias + s_bias + global_mean# [P]
            # 纯均方误差，只用于打印；loss 还要加上正则项
            train_mse = ((prediction - rt) ** 2).mean()
            # L2 正则只作用在观测到的条目上（标准 MF 目标函数），按观测数归一化
            loss = train_mse + cfg.mf_reg * (
                u_vec.pow(2).sum() + s_vec.pow(2).sum() + u_bias.pow(2).sum() + s_bias.pow(2).sum()
            ) / num_ratings
            loss.backward()
            optimizer.step()
            if epoch % log_interval == 0 or epoch == cfg.mf_epochs:
                message = f"        [Model/time_independent/MF]{epoch}/{cfg.mf_epochs} train_mse={train_mse.item():.6f}"
                if test_user_idx is not None:
                    with torch.no_grad():
                        # 测试集沿用训练得到的同一套参数
                        test_prediction = (
                            (user_factors[test_user_idx] * service_factors[test_service_idx]).sum(dim=1)
                            + user_bias[test_user_idx] + service_bias[test_service_idx] + global_mean
                        )# [M]
                        test_error = test_prediction - test_rating# [M]
                        test_mse = test_error.pow(2).mean()
                        mse_value, rmse_value, mae_value = torch.stack([
                            test_mse, test_mse.sqrt(), test_error.abs().mean(),
                        ]).cpu().tolist()
                        message += (
                            f" test_mse={mse_value:.6f}"
                            f" test_rmse={rmse_value:.6f}"
                            f" test_mae={mae_value:.6f}"
                        )
                console_output(message)

        # ---------- 4. 写回隐向量并补全参考矩阵 ----------
        with torch.no_grad():
            # copy_ 支持跨 device，buffer 原本在哪个 device 就写回哪个 device
            self.user_factors.copy_(user_factors.detach())
            self.service_factors.copy_(service_factors.detach())
            reference_matrix = (
                user_factors @ service_factors.t()# [user_num, service_num]
                + user_bias.unsqueeze(1)
                + service_bias.unsqueeze(0)
                + global_mean
            )
        return reference_matrix.detach()

    # 查询冻结的 MF 隐向量并投影成两侧表示。
    def forward(self, user_indices: torch.Tensor, service_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        功能描述:
            从矩阵分解结果中查出当前 batch 的隐向量，分别投影到 Token 维度。

        Input Shape:
            user_indices：当前 batch 的用户索引，·[batch_size]。
            service_indices：当前 batch 的服务索引，·[batch_size]。

        Output Shape:
            user_mf：用户 MF 表示，·[batch_size, config.e]。
            service_mf：服务 MF 表示，·[batch_size, config.e]。
        """
        return self.user_projection(self.user_factors[user_indices]), self.service_projection(self.service_factors[service_indices])
