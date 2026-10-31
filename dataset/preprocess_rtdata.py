#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按本文件"参数区"里的 TIME_SLOTS / USER_NUM / SERVICE_NUM 截取 WS-DREAM 的 rtdata.txt，
结合源文件的连续排列验证和修复索引，筛选重复记录，
写成一个三元组唯一的 TSV 缓存，供 data/dataset.py 快速加载。

rtdata.txt 每行格式（制表符分隔、无表头、4 列）:
    time | user | service | rt
其中 rt == -1 表示该次调用失败 / 缺失。

用法:
    1. 改下面"参数区"里的路径和三个维度上限
    2. python dataset/preprocess_rtdata.py --overwrite

说明:
    1. 不读取 config.py，所有参数都在本文件的参数区里定义，因此也不需要安装 torch。
    2. 流式读取源文件，只缓存有限邻接行和截取范围内的候选记录。
    3. 输出列顺序与 data/dataset.py 里的 names 完全一致，且不带表头、以 \\n 结尾。
    4. 索引修复需要两侧连续记录共同给出唯一答案，响应时间的数字不作猜测修复。
    5. 重复项优先保留邻接证据更充分的记录，同等可信但响应时间冲突的组不输出。
    6. 清洗明细写到缓存路径后缀为 .cleaning.tsv 的文件，源文件不作修改。
"""

import argparse
from collections import deque
from dataclasses import dataclass
import math
import os
import csv
import re
import tempfile
import time
from typing import Iterator

import numpy as np
import pandas as pd

# 根据脚本位置定位源文件和缓存，不依赖命令启动目录。
_HERE = os.path.dirname(os.path.abspath(__file__))# dataset/ 的绝对路径


# ============================== 参数区（直接在这里改） ==============================
# 原始数据文件路径
INPUT_PATH = os.path.join(_HERE, "Supplement_WSDREAM_dataset2", "data", "rtdata.txt")
# 输出 TSV 路径：要与 config.py 里的 data_path 指向同一个文件，data/dataset.py 会读它
OUTPUT_PATH = os.path.join(_HERE, "Supplement_WSDREAM_dataset2", "rtdata_cache2.csv")
# 清洗明细保留原始行号、修复索引及重复项选择依据。
REPORT_PATH = OUTPUT_PATH + ".cleaning.tsv"

# 截取范围：只保留 time < TIME_SLOTS、user < USER_NUM、service < SERVICE_NUM 的行
TIME_SLOTS = 96
USER_NUM = 140
SERVICE_NUM = 1000

# 源文件完整维度及排列顺序：先用户，再服务，最后时间；切换数据版本时需同步调整。
SOURCE_TIME_SLOTS = 480
SOURCE_USER_NUM = 420
SOURCE_SERVICE_NUM = 1000
# 源文件正常响应时间保留三位小数，用于识别截断或多出数字的损坏字段。
SOURCE_RT_DECIMALS = 3
# 每侧最多查看多少行邻接记录，连续序号不唯一时不修复。
CONTEXT_RADIUS = 8
# 最低索引可信度：2 表示至少得到一侧连续记录支持，3 表示两侧共同支持。
MIN_CONFIDENCE = 2

# 是否额外丢掉 rt<=0 的行（-1 表示调用失败 / 缺失，约占 15%）
DROP_INVALID_RT = False
# 输出文件已存在时是否直接覆盖
OVERWRITE = False
# 每隔多少源文件行汇报进度，以及每批写出的记录数。
CHUNK_ROWS = 2_000_000

# 列顺序必须与 data/dataset.py 的 names 保持一致
COLUMNS = ["time", "user", "service", "rt"]
# ==================================================================================

# 索引必须为标准非负整数，避免将损坏的 000 无条件解释为 0。
INDEX_PATTERN = re.compile(rb"(?:0|[1-9][0-9]*)\Z")
# 响应时间必须符合源文件的小数位数，调用失败标记允许原始的 -1。
RT_NUMBER_PATTERN = rb"(?:0|[1-9][0-9]*)\.[0-9]{" + str(SOURCE_RT_DECIMALS).encode("ascii") + rb"}"
RT_PATTERN = re.compile(rb"(?:-1|" + RT_NUMBER_PATTERN + rb")\Z")
# 分隔符损坏时，仅提取有明确分隔的十进制响应时间或失败标记。
RT_SUFFIX_PATTERN = re.compile(rb"[\t ](-1|" + RT_NUMBER_PATTERN + rb")\Z")


# 将源文件原始行和可解析索引保存在有限大小的邻接窗口中。
@dataclass(slots=True)
class SourceRecord:
    """
    功能描述:
        保存原始字节和行号，索引按源文件完整维度编码为连续序号。
        无法严格解析的索引或响应时间字段以 None 表示，等待邻接验证。

    参数:
        line_number（int，标量）：从 1 开始的源文件物理行号。
        raw（bytes，一维字节序列）：去掉行尾换行符后的原始记录。
        indices（tuple[int, int, int] 或 None，形状 [3]）：时间、用户和服务索引。
        ordinal（int 或 None，标量）：该三元组在完整源数据中的连续序号。
        rt_text（bytes 或 None，一维字节序列）：尚未数值化的响应时间字段。

    返回值:
        record（SourceRecord，标量）：源文件记录实例。
    """
    line_number: int
    raw: bytes
    indices: tuple[int, int, int] | None
    ordinal: int | None
    rt_text: bytes | None


# 以字节形式解析一行，保留不能直接解析的记录供邻接修复。
def parse_source_record(line_number: int, raw: bytes) -> SourceRecord:
    """
    功能描述:
        严格检查四列结构、索引文本和完整源数据的索引范围。
        列数异常时只尝试提取明确的响应时间尾字段，不猜测索引或响应时间数字。

    参数:
        line_number（int，标量）：原文件行号。
        raw（bytes，一维字节序列）：原文件的一整行。

    返回值:
        record（SourceRecord，标量）：含原始行和解析结果的记录。
    """
    # 保留原始字节，按制表符检查四列结构。
    raw = raw.rstrip(b"\r\n")
    fields = raw.split(b"\t")
    indices = None
    ordinal = None
    rt_text = fields[3] if len(fields) == 4 else None

    # 只接受标准整数索引，超出完整源维度的索引不能作为邻接锚点。
    if len(fields) == 4 and all(INDEX_PATTERN.fullmatch(field) for field in fields[:3]):
        try:
            time_idx, user_idx, service_idx = map(int, fields[:3])
        except ValueError:
            pass
        else:
            if (
                0 <= time_idx < SOURCE_TIME_SLOTS
                and 0 <= user_idx < SOURCE_USER_NUM
                and 0 <= service_idx < SOURCE_SERVICE_NUM
            ):
                indices = (time_idx, user_idx, service_idx)
                ordinal = (user_idx * SOURCE_SERVICE_NUM + service_idx) * SOURCE_TIME_SLOTS + time_idx

    # 少列时保留明确的尾字段，多列可能合并了多个观测，不提取其响应时间。
    if rt_text is None and len(fields) < 4:
        match = RT_SUFFIX_PATTERN.search(raw)
        if match:
            rt_text = match.group(1)
    return SourceRecord(line_number, raw, indices, ordinal, rt_text)


# 流式产生当前记录及有限数量的前后邻接记录。
def iter_context_records(path: str) -> Iterator[tuple[SourceRecord, tuple[SourceRecord, ...], tuple[SourceRecord, ...]]]:
    """
    功能描述:
        按源文件物理行逐行读取，保留每侧最多 CONTEXT_RADIUS 条记录。
        不跳过坏行，保证修复时使用的行距不会被过滤操作改变。

    参数:
        path（str，标量）：源文件路径。

    返回值:
        contexts（Iterator，每项形状 [3]）：当前记录、前侧记录元组和后侧记录元组。
    """
    # 预取后侧窗口，前侧窗口随读取自动淘汰旧记录。
    with open(path, "rb") as source:
        rows = enumerate(source, start=1)
        before = deque(maxlen=CONTEXT_RADIUS)
        pending = deque()
        for _ in range(CONTEXT_RADIUS + 1):
            item = next(rows, None)
            if item is None:
                break
            pending.append(parse_source_record(*item))

        # 每次只推进一条物理记录，保留真实的相对行位置。
        while pending:
            current = pending.popleft()
            yield current, tuple(before), tuple(pending)
            before.append(current)
            item = next(rows, None)
            if item is not None:
                pending.append(parse_source_record(*item))


# 将完整源数据中的连续序号还原为时间、用户和服务索引。
def decode_ordinal(ordinal: int) -> tuple[int, int, int]:
    """
    功能描述:
        按源文件的用户、服务、时间排列顺序解码连续序号。

    参数:
        ordinal（int，标量）：完整源数据中的有效连续序号。

    返回值:
        indices（tuple[int, int, int]，形状 [3]）：时间、用户和服务索引。
    """
    # 先还原时间，再拆分用户和服务组合。
    pair_id, time_idx = divmod(ordinal, SOURCE_TIME_SLOTS)
    user_idx, service_idx = divmod(pair_id, SOURCE_SERVICE_NUM)
    return time_idx, user_idx, service_idx


# 交叉验证两侧连续片段，只对具有唯一邻接答案的索引进行修复。
def resolve_indices(
    record: SourceRecord,
    before: tuple[SourceRecord, ...],
    after: tuple[SourceRecord, ...],
) -> tuple[tuple[int, int, int] | None, int, str]:
    """
    功能描述:
        每个锚点至少由两条连续索引组成，两侧锚点的序号跨度必须与物理行距一致。
        两侧推断结果唯一时修复索引；无法唯一定位时不根据单侧序号猜测新索引。
        未修复的合法索引按邻接证据分为单侧支持或证据不足。

    参数:
        record（SourceRecord，标量）：待检查记录。
        before（tuple[SourceRecord, ...]，形状 [L]）：按原顺序排列的前侧记录。
        after（tuple[SourceRecord, ...]，形状 [R]）：按原顺序排列的后侧记录。

    返回值:
        result（tuple，形状 [3]）：索引或 None、可信度 0 至 3、修复或拒绝原因。
    """
    # 优先使用紧邻的两个连续锚点，正常数据无需扩大搜索。
    left = before[-1] if len(before) >= 2 else None
    right = after[0] if len(after) >= 2 else None
    left_supported = (
        left is not None and left.ordinal is not None and before[-2].ordinal is not None
        and before[-2].ordinal + 1 == left.ordinal
    )
    right_supported = (
        right is not None and right.ordinal is not None and after[1].ordinal is not None
        and right.ordinal + 1 == after[1].ordinal
    )
    if left_supported and right_supported and right.ordinal - left.ordinal == 2:
        expected = left.ordinal + 1
        reason = "两侧连续索引共同确定修复位置" if expected != record.ordinal else ""
        return decode_ordinal(expected), 3, reason

    # 从更远的连续片段寻找锚点，相对位置必须保留中间坏行。
    left_anchors = []
    for i in range(1, len(before)):
        previous, current = before[i - 1], before[i]
        if previous.ordinal is not None and current.ordinal is not None and previous.ordinal + 1 == current.ordinal:
            left_anchors.append((i - len(before), current.ordinal))
    right_anchors = []
    for i in range(len(after) - 1):
        current, following = after[i], after[i + 1]
        if current.ordinal is not None and following.ordinal is not None and current.ordinal + 1 == following.ordinal:
            right_anchors.append((i + 1, current.ordinal))

    # 所有可用双侧组合必须给出同一个序号，避免跨越合并行或错位片段猜测。
    expected_ordinals = set()
    for left_position, left_ordinal in left_anchors:
        for right_position, right_ordinal in right_anchors:
            if right_ordinal - left_ordinal == right_position - left_position:
                expected_ordinals.add(left_ordinal - left_position)
    if len(expected_ordinals) == 1:
        expected = expected_ordinals.pop()
        reason = "两侧连续片段及物理行距共同确定修复位置" if expected != record.ordinal else ""
        return decode_ordinal(expected), 3, reason
    if len(expected_ordinals) > 1:
        return None, 0, "两侧邻接片段给出不同位置，无法确定真实索引"

    # 无双侧唯一答案时，只保留有连续邻接支持的原索引。
    if record.indices is None:
        return None, 0, "索引或分隔符损坏，缺少唯一修复证据"
    if left_anchors and right_anchors:
        left_ordinal = left_anchors[-1][1]
        right_ordinal = right_anchors[0][1]
        if left_ordinal < right_ordinal and not left_ordinal < record.ordinal < right_ordinal:
            return None, 0, "原索引与两侧连续片段顺序冲突，无法唯一修复"
    if (
        (left_supported and left.ordinal + 1 == record.ordinal)
        or (right_supported and record.ordinal + 1 == right.ordinal)
    ):
        return record.indices, 2, ""
    return record.indices, 1, "原索引缺少连续邻接支持"


# 按三元组存储最高可信候选，冲突状态跨整个源文件保持。
class CandidateCache:
    """
    功能描述:
        每个输出三元组只保留最高可信度的候选记录。
        最高可信度相同时，响应时间相同则合并，不同则标记整组不输出。

    参数:
        无显式参数；缓存大小由 TIME_SLOTS、USER_NUM 和 SERVICE_NUM 确定。

    返回值:
        cache（CandidateCache，标量）：含响应时间、可信度、行号和冲突状态的一维缓存。
    """

    # 分配固定大小数组，避免为数百万条记录创建字典对象。
    def __init__(self) -> None:
        """
        功能描述:
            初始化输出范围内的候选状态，三元组编码后的数组长度为 K。

        参数:
            无；K 等于 USER_NUM × SERVICE_NUM × TIME_SLOTS。

        返回值:
            None：各状态数组形状均为 [K]。
        """
        # 响应时间和源行号保留数值精度，可信度为 0 表示尚无候选。
        size = USER_NUM * SERVICE_NUM * TIME_SLOTS
        self.values = np.zeros(size, dtype=np.float64)
        self.confidence = np.zeros(size, dtype=np.uint8)
        self.line_numbers = np.zeros(size, dtype=np.int64)
        self.conflicts = np.zeros(size, dtype=bool)
        self.repaired = np.zeros(size, dtype=bool)

    # 比较同一三元组的候选证据，记录选择或冲突结果。
    def add(
        self, indices: tuple[int, int, int], rt: float, confidence: int,
        line_number: int, repaired: bool,
    ) -> tuple[str, int, float | None]:
        """
        功能描述:
            新候选可信度更高时替换旧候选并清除低可信冲突；更低时丢弃新候选。
            同等可信的不同响应时间保持冲突，后续相同值不会擅自解除冲突。

        参数:
            indices（tuple[int, int, int]，形状 [3]）：输出范围内的时间、用户和服务索引。
            rt（float，标量）：严格解析后的有限响应时间。
            confidence（int，标量）：索引可信度，范围 1 至 3。
            line_number（int，标量）：源文件行号。
            repaired（bool，标量）：当前索引是否经过修复。

        返回值:
            decision（tuple，形状 [3]）：事件名称、旧候选行号、旧候选响应时间或 None。
        """
        # 查找同一三元组在输出缓存中的唯一位置。
        time_idx, user_idx, service_idx = indices
        position = (user_idx * SERVICE_NUM + service_idx) * TIME_SLOTS + time_idx
        old_confidence = int(self.confidence[position])
        old_line = int(self.line_numbers[position])
        old_rt = float(self.values[position]) if old_confidence else None

        # 首次出现或证据更充分时写入新候选。
        if confidence > old_confidence:
            self.values[position] = rt
            self.confidence[position] = confidence
            self.line_numbers[position] = line_number
            self.conflicts[position] = False
            self.repaired[position] = repaired
            event = "首次候选" if old_confidence == 0 else "保留更可信记录"
            return event, old_line, old_rt

        # 低可信重复项不能覆盖高可信记录。
        if confidence < old_confidence:
            return "剔除低可信重复项", old_line, old_rt

        # 同等可信的相同值合并，不同值保持整组冲突。
        if rt == old_rt:
            return "合并相同重复项", old_line, old_rt
        self.conflicts[position] = True
        return "标记重复组冲突", old_line, old_rt


# 将修复和重复选择依据写入清洗明细，保留原始坏字节的转义形式。
def write_audit(
    writer, event: str, record: SourceRecord, indices: tuple[int, int, int] | None,
    reason: str, other_line: int = 0, rt: float | None = None, other_rt: float | None = None,
) -> None:
    """
    功能描述:
        写出清洗事件、源行号、对照行号、修复前后索引和原始字节。
        明细中的重复组冲突可能被后续更高可信候选解除，最终取舍以汇总和缓存为准。

    参数:
        writer（csv 写入器，标量）：清洗明细的制表符写入器。
        event（str，标量）：清洗事件名称。
        record（SourceRecord，标量）：当前源记录。
        indices（tuple[int, int, int] 或 None，形状 [3]）：处理后的索引。
        reason（str，标量）：修复或取舍依据。
        other_line（int，标量）：对照候选的源行号，0 表示无对照。
        rt（float 或 None，标量）：当前候选响应时间。
        other_rt（float 或 None，标量）：对照候选响应时间。

    返回值:
        None：向明细文件追加一行。
    """
    # 使用字节转义显示原行，避免控制字符损坏明细的列结构。
    writer.writerow([
        event, record.line_number, other_line or "", record.indices or "", indices or "",
        rt if rt is not None else "", other_rt if other_rt is not None else "",
        reason, repr(record.raw),
    ])


# 流式清洗源数据，筛选唯一候选并原子替换输出缓存。
def preprocess() -> None:
    """
    功能描述:
        按邻接证据恢复可确定的索引，拒绝证据不足或响应时间损坏的记录。
        全局筛选重复三元组，先写临时文件，完整成功后再替换缓存和清洗明细。

    参数:
        无；输入路径、截取维度和清洗规则取自参数区。

    返回值:
        None：生成四列缓存和带源文件行号的清洗明细。
    """
    # 初始化候选缓存和统计，只为截取范围分配存储空间。
    start_time = time.monotonic()
    cache = CandidateCache()
    stats = {
        "read": 0, "repaired": 0, "index_rejected": 0, "out_of_range": 0,
        "damaged_rt": 0, "candidates": 0, "duplicates": 0, "replaced": 0,
    }
    output_tmp = None
    report_tmp = None
    print(f"[preprocess]输入 {INPUT_PATH}", flush=True)
    print(f"[preprocess]输出 {OUTPUT_PATH}", flush=True)
    print(f"[preprocess]清洗明细 {REPORT_PATH}", flush=True)

    # 清洗明细先写临时文件，失败时不覆盖已有结果。
    try:
        report_fd, report_tmp = tempfile.mkstemp(
            prefix=os.path.basename(REPORT_PATH) + ".", suffix=".tmp",
            dir=os.path.dirname(os.path.abspath(REPORT_PATH)),
        )
        with os.fdopen(report_fd, "w", encoding="utf-8", newline="") as report:
            writer = csv.writer(report, delimiter="\t", lineterminator="\n")
            writer.writerow([
                "事件", "源文件行号", "对照源行号", "原索引", "处理后索引",
                "当前响应时间", "对照响应时间", "依据", "原始字节",
            ])

            # 用原始物理行距验证索引，所有统计包含源文件坏行。
            for record, before, after in iter_context_records(INPUT_PATH):
                stats["read"] = record.line_number
                if stats["read"] % CHUNK_ROWS == 0:
                    print(
                        f"   [preprocess]源行={stats['read']} 候选={stats['candidates']} "
                        f"修复索引={stats['repaired']} 重复候选={stats['duplicates']} "
                        f"耗时={time.monotonic() - start_time:.1f}s", flush=True,
                    )
                indices, confidence, reason = resolve_indices(record, before, after)
                if indices is None or confidence < MIN_CONFIDENCE:
                    stats["index_rejected"] += 1
                    write_audit(writer, "剔除不确定索引", record, indices, reason)
                    continue

                # 邻接证据确定的位置可能超出截取范围，不能继续沿用损坏索引。
                repaired = indices != record.indices
                if repaired:
                    stats["repaired"] += 1
                time_idx, user_idx, service_idx = indices
                if not (time_idx < TIME_SLOTS and user_idx < USER_NUM and service_idx < SERVICE_NUM):
                    stats["out_of_range"] += 1
                    if repaired:
                        write_audit(writer, "修复后超出截取范围", record, indices, reason)
                    continue

                # 响应时间严格解析，不按邻居的响应时间补值或猜测损坏数字。
                if record.rt_text is None or RT_PATTERN.fullmatch(record.rt_text) is None:
                    stats["damaged_rt"] += 1
                    write_audit(writer, "剔除损坏响应时间", record, indices, "响应时间字段损坏，真实数值无法确定")
                    continue
                rt = float(record.rt_text)
                if not math.isfinite(rt):
                    stats["damaged_rt"] += 1
                    write_audit(writer, "剔除非有限响应时间", record, indices, "响应时间不是有限数值")
                    continue
                if repaired:
                    write_audit(writer, "修复索引", record, indices, reason, rt=rt)

                # 在整个源文件范围内筛选候选，跨进度区间的重复项同样参与比较。
                stats["candidates"] += 1
                event, other_line, other_rt = cache.add(indices, rt, confidence, record.line_number, repaired)
                if other_line:
                    stats["duplicates"] += 1
                    if event == "保留更可信记录":
                        stats["replaced"] += 1
                    decision_reasons = {
                        "保留更可信记录": "当前行邻接证据更充分，替换对照行并解除较低可信度的冲突",
                        "剔除低可信重复项": "对照行邻接证据更充分，保留对照行",
                        "合并相同重复项": "同等可信且响应时间相同，合并；已有不同值冲突仍保留",
                        "标记重复组冲突": "最高可信度相同但响应时间不同，整组暂不输出",
                    }
                    write_audit(writer, event, record, indices, decision_reasons[event], other_line, rt, other_rt)

        # 完成全局去重后确定输出集合，每个三元组最多有一条记录。
        keep_mask = (cache.confidence >= MIN_CONFIDENCE) & ~cache.conflicts
        invalid_rt_count = 0
        if DROP_INVALID_RT:
            invalid_rt_count = int(np.count_nonzero(keep_mask & (cache.values <= 0)))
            keep_mask &= cache.values > 0
        selected = np.flatnonzero(keep_mask)
        if selected.size == 0:
            raise SystemExit("没有可信且无冲突的记录可写出，原缓存未替换；请检查源文件维度和清洗规则。")

        # 按用户、服务和时间顺序分批写临时缓存，保留原有四列格式。
        output_fd, output_tmp = tempfile.mkstemp(
            prefix=os.path.basename(OUTPUT_PATH) + ".", suffix=".tmp",
            dir=os.path.dirname(os.path.abspath(OUTPUT_PATH)),
        )
        with os.fdopen(output_fd, "w", encoding="utf-8", newline="") as destination:
            for offset in range(0, selected.size, CHUNK_ROWS):
                positions = selected[offset:offset + CHUNK_ROWS]
                pair_ids, time_indices = np.divmod(positions, TIME_SLOTS)
                user_indices, service_indices = np.divmod(pair_ids, SERVICE_NUM)
                output = pd.DataFrame({
                    "time": time_indices, "user": user_indices,
                    "service": service_indices, "rt": cache.values[positions],
                }, columns=COLUMNS)
                output.to_csv(destination, sep="\t", header=False, index=False, lineterminator="\n")
                print(f"   [preprocess]已写出 {min(offset + CHUNK_ROWS, selected.size)} 条", flush=True)

        # 完整成功后原子替换文件，避免中断留下半份缓存。
        os.replace(report_tmp, REPORT_PATH)
        os.replace(output_tmp, OUTPUT_PATH)
        print("[preprocess]处理完毕。", flush=True)
        print(f"   源文件物理行数    {stats['read']}", flush=True)
        print(f"   确定修复的索引    {stats['repaired']}", flush=True)
        print(f"   索引不确定剔除    {stats['index_rejected']}", flush=True)
        print(f"   超出截取范围      {stats['out_of_range']}", flush=True)
        print(f"   响应时间损坏剔除  {stats['damaged_rt']}", flush=True)
        print(f"   重复候选数        {stats['duplicates']}", flush=True)
        print(f"   更可信候选替换    {stats['replaced']}", flush=True)
        print(f"   最终冲突组剔除    {int(np.count_nonzero(cache.conflicts))}", flush=True)
        print(f"   rt<=0 剔除        {invalid_rt_count}", flush=True)
        print(f"   写出记录数        {selected.size}", flush=True)
        print(f"   写出修复记录数    {int(np.count_nonzero(cache.repaired[selected]))}", flush=True)
        print(f"   总耗时            {time.monotonic() - start_time:.1f}s", flush=True)
        print("   提示：邻接证据只能验证索引，语法合法但已损坏的响应时间数字仍可能残留。", flush=True)
    finally:
        # 失败或中断时删除临时文件，源数据始终只读。
        for temporary_path in (output_tmp, report_tmp):
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)


# 校验输入、输出和源维度，允许命令行明确选择替换旧缓存。
def main() -> None:
    """
    功能描述:
        检查路径和维度参数，确认覆盖策略后执行清洗。

    参数:
        无函数参数；命令行 --overwrite（bool，标量）允许替换已有缓存和清洗明细。

    返回值:
        None：校验通过后调用预处理，异常时退出并打印原因。
    """
    # 解析覆盖开关，保持参数区默认不覆盖已有文件的行为。
    parser = argparse.ArgumentParser(description="验证和修复源数据索引，筛选重复项并生成唯一三元组缓存。")
    parser.add_argument("--overwrite", action="store_true", help="允许替换已有缓存和清洗明细。")
    args = parser.parse_args()
    allow_overwrite = OVERWRITE or args.overwrite

    # 检查文件路径，避免覆盖源数据或把缓存与明细写到同一位置。
    if not os.path.isfile(INPUT_PATH):
        raise SystemExit(f"输入文件不存在：{INPUT_PATH}")
    paths = [os.path.realpath(path) for path in (INPUT_PATH, OUTPUT_PATH, REPORT_PATH)]
    if len(set(paths)) != 3:
        raise SystemExit("源文件、缓存和清洗明细必须使用不同路径。")
    for path in (OUTPUT_PATH, REPORT_PATH):
        if os.path.exists(path) and not allow_overwrite:
            raise SystemExit(f"输出文件已存在：{path}\n如需替换，请添加 --overwrite 或设置 OVERWRITE = True。")

    # 完整源维度用于邻接推断，截取范围不得超出完整源维度。
    dimensions = (TIME_SLOTS, USER_NUM, SERVICE_NUM, SOURCE_TIME_SLOTS, SOURCE_USER_NUM, SOURCE_SERVICE_NUM)
    if any(not isinstance(value, int) or value <= 0 for value in dimensions):
        raise SystemExit("源维度和截取维度必须为正整数。")
    if TIME_SLOTS > SOURCE_TIME_SLOTS or USER_NUM > SOURCE_USER_NUM or SERVICE_NUM > SOURCE_SERVICE_NUM:
        raise SystemExit("截取维度不能超过完整源维度，请检查参数区。")
    if CONTEXT_RADIUS < 2 or MIN_CONFIDENCE not in (2, 3) or CHUNK_ROWS < 1 or SOURCE_RT_DECIMALS < 1:
        raise SystemExit("邻接半径至少为 2，最低可信度只能为 2 或 3，分批大小和源小数位数必须大于 0。")

    # 创建输出目录，开始清洗时不提前删除已有缓存。
    for path in (OUTPUT_PATH, REPORT_PATH):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    preprocess()


# 仅在直接运行脚本时开始清洗。
if __name__ == "__main__":
    main()
