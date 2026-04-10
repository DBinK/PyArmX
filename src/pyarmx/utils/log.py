# utils/logging.py
from pathlib import Path
import sys

from loguru import logger

# 确保日志目录存在
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
# 定义 patcher：确保每条日志的 extra 都有 task_id
def patcher(record):
    if "task_id" not in record["extra"]:
        record["extra"]["task_id"] = ""

LOGURU_FORMAT = (
    "<green>{time:YY.MM.DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{function}</cyan> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - "
    "<magenta>{extra[task_id]}</magenta>: " 
    "<level>{message}</level>"
)

# 移除默认处理器
logger.remove()

# 配置 patcher 
logger.configure(patcher=patcher)

# 1. 控制台输出：INFO 及以上，格式清晰，带颜色（开发环境友好）
logger.add(
    sys.stderr,
    # level="INFO" if not __debug__ else "DEBUG",
    level="INFO",
    enqueue=True,  # 多线程安全
    colorize=True, # 终端彩色输出
    format=LOGURU_FORMAT
)


# 2. 全量日志文件：DEBUG 及以上，用于问题排查
# logger.add(
#     LOG_DIR / "app.log",
#     level="DEBUG",
#     rotation="10 MB",     # 文件超过10MB自动轮转
#     retention="7 days",   # 保留7天旧文件
#     compression="zip",    # 压缩归档，节省空间
#     enqueue=True,         # 多进程/线程安全
#     encoding="utf-8",
# )

# 3. 错误日志：仅 ERROR 及以上，便于监控告警
# logger.add(
#     LOG_DIR / "errors.log",
#     level="ERROR",
#     rotation="10 MB",
#     retention="30 days",
#     compression="zip",
#     enqueue=True,
#     encoding="utf-8",
# )

# 4. （可选）警告日志单独分离
# logger.add(
#     LOG_DIR / "warnings.log",
#     level="WARNING",
#     rotation="10 MB",
#     retention="14 days",
#     enqueue=True,
#     encoding="utf-8",
# )

# 导出 logger 实例，供其他模块导入使用
# 这是关键：所有模块都从这里导入，确保全局唯一、配置一致

def fmt_arr(arr, precision=3):
    """将数组格式化为固定小数位的字符串，如 [0.000, 1.234]"""
    return "[" + ", ".join(f"{x:.{precision}f}" for x in arr) + "]"


__all__ = ["logger", "fmt_arr"]