#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义元数据预处理：给 userlist.txt / wslist.txt 的每一行末尾追加"经纬度网格编号"。

原文件格式（制表符分隔、无表头、8 列）:
    id | ip | 国家 | 州 | 城市 | 纬度 | 经度 | 自治系统号
其中 "None" 表示该字段缺失（有 57/420 个用户、139/1000 个服务的整行都是 None）。

本脚本做两件事:
    1. 把每行规整成固定 8 列：列数不足补 "None"；列数过多说明城市名里混进了制表符
       （如 wslist.txt 的 id=8：Argentina / Ciudad Autonoma de Buenos Aires / Buenos Aires），
       按"前 2 列是 id/ip、最后 3 列是 纬度/经度/自治系统号、中间几列并成 国家/州/城市"还原。
    2. 用硬编码的网格规则算出编号，追加成第 9 列，写到同目录的新文件里：
           网格编号 = 0                                          （经纬度缺失或越界）
                    = 1 + 纬度格号 × LON_NUM + 经度格号           （正常）
           纬度格号 = int((纬度 - LAT_LOW) / GRID_DEGREE)
           经度格号 = int((经度 - LON_LOW) / GRID_DEGREE)
       编号 0 专门留给 unknown，所以正常格子的编号从 1 开始。

用法:
    1. 需要时改下面"参数区"里的路径与网格度数
    2. python dataset/preprocess_semantic.py
    3. data/dataset.py 的 load_semantic_text 直接读输出文件（第 9 列就是网格编号）

说明:
    1. 不读取 config.py，所有参数都在本文件的参数区里定义，因此也不需要安装 torch。
    2. 原始的 *list.txt 不会被修改，输出是同目录下的 *_grid.txt。
    3. 行号必须和 id 一一对应：任何异常行都只做补齐/合并，绝不丢行，脚本会校验第 0 列等于行号。
"""

import os
import time


_HERE = os.path.dirname(os.path.abspath(__file__))# dataset/ 的绝对路径

# ============================== 参数区（直接在这里改） ==============================
# 两份 list 文件所在目录
DATA_DIR = os.path.join(_HERE, "Supplement_WSDREAM_dataset2", "data")
# 输入 → 输出：原始文件不动，输出只是多了最后一列网格编号
JOBS = (
    ("userlist.txt", "userlist_grid.txt"),
    ("wslist.txt", "wslist_grid.txt"),
)

# 经纬度网格：硬编码的划分规则（1° × 1° → 纬度 180 格 × 经度 360 格）
GRID_DEGREE = 1.0
LAT_LOW, LAT_HIGH = -90.0, 90.0 # 纬度范围
LON_LOW, LON_HIGH = -180.0, 180.0# 经度范围

# 输出文件已存在时是否直接覆盖
OVERWRITE = False

# 输入列序（与文件一致）与输出列数（多一列网格编号）
COLUMN_ID, COLUMN_IP, COLUMN_COUNTRY, COLUMN_STATE, COLUMN_CITY, COLUMN_LAT, COLUMN_LON, COLUMN_ASN = range(8)
INPUT_COLUMN_NUM = 8
OUTPUT_COLUMN_NUM = 9
NONE_TOKEN = "None"
# ==================================================================================


# 由网格度数推出的格数（硬编码规则的一部分）
LAT_NUM = int(round((LAT_HIGH - LAT_LOW) / GRID_DEGREE))
LON_NUM = int(round((LON_HIGH - LON_LOW) / GRID_DEGREE))
GRID_NUM = LAT_NUM * LON_NUM

## 读取一行文本，规整成固定列数
def parse_row(line: str) -> list[str]:
    """
    功能描述:
        把一行原始文本规整成 INPUT_COLUMN_NUM 列：列数不足补 "None"，列数过多按规则合并回 国家/州/城市。

    Input Shape:
        line：一行原始文本（不含换行符），·[标量]。

    Output Shape:
        fields：规整后的文本列，·[INPUT_COLUMN_NUM]。
    """
    # 清洗每个列
    fields = [x.strip() for x in line.split("\t")]
    # 列数不足则用NONE_TOKEN补齐
    if len(fields) < INPUT_COLUMN_NUM:
        fields += [NONE_TOKEN] * (INPUT_COLUMN_NUM - len(fields))
    # 处理列数过多的情况
    elif len(fields) > INPUT_COLUMN_NUM:
        # 多出来的列是城市名里混进的制表符：前 2 列 id/ip、最后 3 列 纬度/经度/ASN，中间并成 国家/州/城市
        middle = fields[2:-3]# [国家, 州, 城市, ...]
        country = middle[0]# 国家
        city = middle[-1]# 城市
        state = " ".join(middle[1:-1]).strip() or NONE_TOKEN# 州
        fields = fields[:2] + [country, state, city] + fields[-3:]# 拼接
    return fields

## 计算经纬度块编号
def grid_id(fields: list[str]) -> int:
    """
    功能描述:
        算出一行的经纬度网格编号：0 表示缺失或越界，其余为 1 + 纬度格号 × LON_NUM + 经度格号。

    Input Shape:
        fields：规整后的文本列，·[INPUT_COLUMN_NUM]。

    Output Shape:
        id：网格编号，·[标量]。
    """
    # 获取纬度/经度文本
    lat_text, lon_text = fields[COLUMN_LAT], fields[COLUMN_LON]# [纬度, 经度]
    # 缺失或非法值直接返回 0
    if lat_text == NONE_TOKEN or lon_text == NONE_TOKEN:# 缺失
        return 0
    try:# 非法值
        lat, lon = float(lat_text), float(lon_text)
    except ValueError:
        return 0
    if not (LAT_LOW <= lat < LAT_HIGH) or not (LON_LOW <= lon < LON_HIGH):# 越界
        return 0# 越界（含 lat=90、lon=180 这类落不进任何格子的边界值）
    # 分块
    lat_idx = int((lat - LAT_LOW) / GRID_DEGREE)# 合法范围内偏移非负，int 即向下取整
    lon_idx = int((lon - LON_LOW) / GRID_DEGREE)
    if not (0 <= lat_idx < LAT_NUM and 0 <= lon_idx < LON_NUM):
        return 0
    return 1 + lat_idx * LON_NUM + lon_idx

## 主功能函数
def preprocess_file(input_path: str, output_path: str) -> dict:
    """
    功能描述:
        读一份 *list.txt，规整每行并追加网格编号后写到 output_path。

    Input Shape:
        input_path： 原始 list 文件路径，制表符分隔、无表头，·[标量]。
        output_path：输出文件路径，·[标量]。

    Output Shape:
        stats：统计量字典（行数、补齐/合并行数、网格缺失数、网格编号范围、去重格子数），·[标量]。
    """
    # 输出文件已存在时直接报错，避免误覆盖原始文件
    if os.path.exists(output_path) and not OVERWRITE:
        raise FileExistsError(f"输出文件已存在：{output_path}（确认要覆盖就把 OVERWRITE 改成 True）")

    # 统计
    stats = {"rows": 0, "padded": 0, "merged": 0, "grid_unknown": 0, "grid_min": 0, "grid_max": 0, "grid_cells": 0}# 行数，有填充的行数，有合并的行数，网格编号缺失的行数，网格编号最小值，网格编号最大值，去重后的格子数
    cells = set()# 集合，存放出现过的网格编号
    with open(input_path, encoding="utf-8", errors="replace") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")# 删除右侧的"\n"
            # 跳过空行
            if line.strip() == "":
                continue
            # 解析行数据
            raw_column_num = len(line.split("\t"))# 元素个数
            fields = parse_row(line)# 规整成List[str]
            # 统计补齐/合并行数
            if raw_column_num < INPUT_COLUMN_NUM:
                stats["padded"] += 1
            elif raw_column_num > INPUT_COLUMN_NUM:
                stats["merged"] += 1
            # 检查 id 列与行号是否一致
            if fields[COLUMN_ID] != str(stats["rows"]):
                raise ValueError(
                    f"{input_path} 第 {stats['rows']} 行的 id 列是 {fields[COLUMN_ID]!r}，与行号不一致；"
                    f"语义表按行号查表，不一致会静默错位，请检查 list 文件。"
                )
            number = grid_id(fields)
            # 统计位置编号的个数
            if number == 0:
                stats["grid_unknown"] += 1
            else:
                cells.add(number)
                stats["grid_min"] = number if stats["grid_min"] == 0 else min(stats["grid_min"], number)
                stats["grid_max"] = max(stats["grid_max"], number)
            # 写入output
            fout.write("\t".join(fields + [str(number)]) + "\n")
            stats["rows"] += 1
    stats["grid_cells"] = len(cells)# 统计去重后的格子数
    return stats


def main() -> None:
    """
    功能描述:
        按参数区的 JOBS 依次预处理两份 list 文件，并打印统计量。

    Input Shape:
        无参数，路径与网格度数全部取自本文件顶部的参数区。

    Output Shape:
        return：无返回值，·[无]。
    """
    print(f"[preprocess_semantic]数据目录 {DATA_DIR}")
    print(f"[preprocess_semantic]网格规则 {GRID_DEGREE}° × {GRID_DEGREE}° → {LAT_NUM} × {LON_NUM} = {GRID_NUM} 格，编号 1..{GRID_NUM}（0 = unknown）")
    start_time = time.time()
    for input_name, output_name in JOBS:
        input_path = os.path.join(DATA_DIR, input_name)
        output_path = os.path.join(DATA_DIR, output_name)
        stats = preprocess_file(input_path, output_path)
        print(f"[preprocess_semantic]{input_name} → {output_name}")
        print(f"   行数              {stats['rows']}")
        print(f"   补齐列数不足的行  {stats['padded']}")
        print(f"   合并列数过多的行  {stats['merged']}")
        print(f"   网格编号缺失(0)   {stats['grid_unknown']}")
        print(f"   网格编号范围      {stats['grid_min']}..{stats['grid_max']}")
        print(f"   去重后的格子数    {stats['grid_cells']}")
        print(f"   输出大小          {os.path.getsize(output_path) / 1024:.1f} KiB")
    print(f"[preprocess_semantic]处理完毕，总耗时 {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
