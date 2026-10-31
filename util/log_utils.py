# util/log_utils.py
import os
from datetime import datetime

_LOG_PATH: str | None = None# 全局唯一，绑定一次

def set_log_path(log_path: str) -> None:
    """
    功能描述:
        绑定进程使用的日志文件地址。

    Input Shape:
        log_path：已创建的日志文件路径，·[标量]。

    Output Shape:
        return：无返回值。
    """
    global _LOG_PATH
    _LOG_PATH = log_path

def console_output(message: object, *, flush: bool = False, show_time: bool = False) -> None:
    """
    功能描述:
        将消息打印到控制台，并追加到已绑定的日志文件。

    Input Shape:
        message：要输出的消息，·[标量]。
        flush：是否立即刷新输出，默认 False。
        show_time：是否在消息末尾追加当前本地时间，默认 False。

    Output Shape:
        return：无返回值。
    """
    # 确认地址已定义
    if not _LOG_PATH:
        raise RuntimeError("尚未绑定日志文件地址，请先调用 set_log_path(cfg.log_path)。")
    # 拼接日志内容
    line = f"{message}"
    if show_time:
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{line}({timestamp})"
    # 打印到控制台，并追加写入日志文件
    print(line, flush=flush)
    if not os.path.isfile(_LOG_PATH):
        raise FileNotFoundError(f"日志文件不存在：{_LOG_PATH}")
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")
