import os

import pandas as pd
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader

from config import Config


class _BatchedTensorDataset(TensorDataset):
    """整批索引内存张量，避免逐样本创建 Tensor 和随后再次 stack。"""

    def __getitems__(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        return tuple(tensor[indices] for tensor in self.tensors)


def _collate_tensor_batch(batch):
    """__getitems__ 已组装好 batch，直接交给 DataLoader 的 pin_memory 流程。"""
    return batch


# ============================== QoS数据（dataset/Supplement_WSDREAM_dataset2/rtdata_cache.csv） ==============================
# 由 dataset/preprocess_qos.py 生成，包含 time, user, service, rt 四列，制表符分隔，无表头。


# 加载rt
def load_data_split(cfg: Config):
    """
    功能描述:
        使用 pandas 读取数据并划分训练集/测试集。

        读取过程中丢弃列数错误、缺失或非数字、索引非整数、rt <= 0 及索引越界的行。

    Input Shape:
        cfg：数据路径、划分比例、随机种子与用户/服务/时间片维度配置，·[标量]。

    Output Shape:
        train_info：训练索引，每行 [time, user, service]，·[N_train, 3]。
        train_rt：训练响应时间，·[N_train, 1]。
        test_info：测试索引，每行 [time, user, service]，·[N_test, 3]。
        test_rt：测试响应时间，·[N_test, 1]。
        train_matrix：训练集 QoS 张量，缺失位置为 0，·[user_num, service_num, time_num]；关闭时间相关模块时为 None。
    """
    # ---------- 1. 读取数据 ----------
    # 数据文件是制表符分隔的 TSV，必须用 sep='\t'：
    # 若用 sep=r'\s+'，制表符丢失的损坏行（如 "193\t29\t78 0.067"）会被切成
    # 4 个字段而当成合法样本读入，rt 静默错位成 0.067。
    # on_bad_lines='skip' 跳过列数不等于 4 的行。
    df = pd.read_csv(
        cfg.data_path,
        sep='\t',            # 制表符分隔
        header=None,         # 文件没有表头
        names=['time', 'user', 'service', 'rt'],  # 指定列名
        encoding='utf-8',
        on_bad_lines='skip'  # 跳过列数不正确的行
    )

    # ---------- 2. 转换数据类型，无法转换的变为 NaN ----------
    # 使用 errors='coerce'，遇到空值或非数字字符串时转为 NaN，而不是抛出异常
    for col in ('time', 'user', 'service', 'rt'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---------- 3. 过滤非法行 ----------
    # 删除含有 NaN 的行：列数不足被填充的 NaN、空值、无法转换为数字的值
    df.dropna(inplace=True)

    # time / user / service 必须是整数，保留浮点值会被 astype(int) 静默截断
    for col in ('time', 'user', 'service'):
        df = df[df[col] % 1 == 0]

    # rt <= 0 全部丢弃：-1 表示调用失败/缺失，0 视为无效观测
    df = df[df['rt'] > 0]

    # 越界值直接丢弃：既能避免 train_matrix 索引越界报错，
    # 也能避免负索引被 PyTorch 解释成"倒数第几个"而静默写到错误位置
    df = df[
        df['time'].between(0, cfg.time_num - 1)
        & df['user'].between(0, cfg.user_num - 1)
        & df['service'].between(0, cfg.service_num - 1)
    ]

    if df.empty:
        raise ValueError(
            f"过滤后没有任何合法样本，请检查数据文件 {cfg.data_path} "
            f"以及维度配置 user_num={cfg.user_num}, "
            f"service_num={cfg.service_num}, time_slots={cfg.time_num}。"
        )

    # 此时已无 NaN，转换安全；rt 保持浮点
    df['time'] = df['time'].astype(int)
    df['user'] = df['user'].astype(int)
    df['service'] = df['service'].astype(int)
    df['rt'] = df['rt'].astype(float)

    # ---------- 4. 随机划分训练集 / 测试集 ----------
    # 使用显式种子的独立随机数生成器：既能复现，也不会污染全局随机状态
    rng = np.random.default_rng(cfg.seed)
    mask = rng.random(len(df)) < cfg.train_ratio

    train_df = df[mask]
    test_df = df[~mask]

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"划分后训练集/测试集为空（train={len(train_df)}, test={len(test_df)}），"
            f"请检查 train_ratio={cfg.train_ratio} 是否设置合理。"
        )

    # ---------- 5. 准备返回的张量 ----------
    # train_info / test_info 的列顺序保持为 [time, user, service]
    train_info = torch.tensor(train_df[['time', 'user', 'service']].to_numpy(), dtype=torch.long)# [num_train,3]
    train_rt = torch.tensor(train_df[['rt']].to_numpy(), dtype=torch.float32)# [num_train,1]
    test_info = torch.tensor(test_df[['time', 'user', 'service']].to_numpy(), dtype=torch.long)# [num_test,3]
    test_rt = torch.tensor(test_df[['rt']].to_numpy(), dtype=torch.float32)# [num_test,1]

    # ---------- 6. 构建时间相关模块使用的 train_matrix ----------
    # 索引顺序：train_matrix[user][service][time] = rt；重复位置由后出现的记录覆盖。
    train_matrix = None
    if cfg.use_time_dependent:
        train_matrix = torch.zeros(cfg.user_num, cfg.service_num, cfg.time_num, dtype=torch.float32)
        train_matrix[train_info[:, 1], train_info[:, 2], train_info[:, 0]] = train_rt.squeeze(1)

    return train_info, train_rt, test_info, test_rt, train_matrix


# 制作rt的DataLoader
def get_dataloader(train_info, train_rt, test_info, test_rt, cfg):
    """
    功能描述:
        将已划分的索引和响应时间封装为训练与测试 DataLoader。

    Input Shape:
        train_info：训练索引，·[N_train, 3]。
        train_rt：训练响应时间，·[N_train, 1]。
        test_info：测试索引，·[N_test, 3]。
        test_rt：测试响应时间，·[N_test, 1]。
        cfg：批量大小、进程数、设备和随机种子等配置，·[标量]。

    Output Shape:
        train_loader：训练集 DataLoader，逐批输出索引和响应时间。
        test_loader：测试集 DataLoader，逐批输出索引和响应时间。
    """

    # 设备统一取自 cfg.device：只有 CUDA 设备才需要 pin_memory
    device = torch.device(cfg.device)
    pin_memory = device.type == "cuda"

    # 训练集的 shuffle 依赖 torch 的随机数生成器，传入固定种子以保证 batch 顺序可复现
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    # 构建 DataLoader
    train_dataset = _BatchedTensorDataset(train_info, train_rt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=_collate_tensor_batch,
        persistent_workers=cfg.num_workers > 0,
    )

    # 测试集不 shuffle，没有随机操作，因此不需要 generator
    test_dataset = _BatchedTensorDataset(test_info, test_rt)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_tensor_batch,
        persistent_workers=cfg.num_workers > 0,
    )

    return train_loader, test_loader


# ============================== 语义元数据（userlist_grid.txt / wslist_grid.txt） ==============================
# 这两份文件由 dataset/preprocess_semantic.py 生成：在原 *list.txt 每行末尾追加了经纬度网格编号。
# 位置与数据缓存放在同一个数据集目录下：<cfg.data_path 所在目录>/data/
SEMANTIC_DIR_NAME = "data"
SEMANTIC_USER_FILE = "userlist_grid.txt"
SEMANTIC_SERVICE_FILE = "wslist_grid.txt"
# 每行的列数与列序（与预处理后的文件一致）：
# id, ip, 国家, 州, 城市, 纬度, 经度, 自治系统号, 经纬度网格编号
SEMANTIC_COLUMN_NUM = 9


def _read_semantic_list(path: str, num_rows: int) -> list[list[str]]:
    """
    功能描述:
        读取预处理后的语义元数据，校验列数及所需行数。

    Input Shape:
        path：制表符分隔、无表头的语义文件路径，·[标量]。
        num_rows：需要保留的前若干行数，·[标量]。

    Output Shape:
        rows：去掉首尾空白后的文本，·[num_rows, SEMANTIC_COLUMN_NUM]。
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")# 删除右侧的"\n"
            # 跳过空行
            if line.strip() == "":
                continue# 跳过空行
            # 切割字段
            fields = [x.strip() for x in line.split("\t")]
            # 检查字段数
            if len(fields) != SEMANTIC_COLUMN_NUM:
                raise ValueError(
                    f"{path} 第 {len(rows)} 行有 {len(fields)} 列，应为 {SEMANTIC_COLUMN_NUM} 列；"
                    f"请先运行 dataset/preprocess_semantic.py 生成带网格编号的文件。"
                )
            # 写入rows
            rows.append(fields)
    # 校验个数
    if len(rows) < num_rows:
        raise ValueError(f"{path} 只有 {len(rows)} 行，少于需要的 {num_rows} 行。")
    return rows[:num_rows]# 只保留 rtdata 用得到的 id（userlist 有 420 行，但 user 只用到前 140 个）


# 读取当前模型所需实体的原始元数据，用于缓存校验和原有经纬度编码。
def load_semantic_text(cfg: Config):
    """
    功能描述:
        从数据集目录读取用户与服务的元数据。

    参数:
        cfg：Config，数据缓存路径及用户、服务数量配置，标量。
    返回值:
        tuple[list[list[str]], list[list[str]]]：用户和服务元数据，形状分别为 [user_num, 9] 和 [service_num, 9]。
    """
    # 获取数据集文件夹路径
    semantic_dir = os.path.join(os.path.dirname(cfg.data_path), SEMANTIC_DIR_NAME)
    # 读取数据
    user_text = _read_semantic_list(os.path.join(semantic_dir, SEMANTIC_USER_FILE), cfg.user_num)
    service_text = _read_semantic_list(os.path.join(semantic_dir, SEMANTIC_SERVICE_FILE), cfg.service_num)
    return user_text, service_text




