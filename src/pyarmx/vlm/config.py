# config.py
import os
from dataclasses import dataclass
from enum import Enum, auto
from dotenv import load_dotenv

# ===== 加载环境变量 =====
load_dotenv()


# ===== 环境变量 =====
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")


# ===== 常量 =====
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LMS_BASE_URL = "http://127.0.0.1:1234/v1"

# ===== 类型定义 =====
@dataclass
class ModelCfg:
    model: str
    base_url: str
    api_key: str
    system_prompt: str = ""

class ModelID(Enum):
    ALIYUN = auto()
    LMS = auto()

# ===== 配置 =====
configs: dict[ModelID, ModelCfg] = {
    ModelID.ALIYUN: ModelCfg(
        model="qwen3.5-flash",
        system_prompt=" ",
        api_key=DASHSCOPE_API_KEY,
        base_url=ALIYUN_BASE_URL,
    ),
    ModelID.LMS: ModelCfg(
        model="qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive",
        system_prompt=" ",
        api_key="any_string",
        base_url=LMS_BASE_URL,
    ),
}
 

# ===== 工具函数 =====
def get_config(provider: ModelID) -> ModelCfg:
    """统一访问入口，IDE自动补全 provider"""
    return configs[provider]


def from_str(name: str) -> ModelID:
    """字符串转 Enum（处理外部输入）"""
    return ModelID(name)