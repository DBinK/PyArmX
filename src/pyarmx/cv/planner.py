import cv2
import numpy as np

from pyarmx.vlm.chatbot import ChatBot
from pyarmx.vlm.config import ModelID, get_config

from pyarmx.vlm.prompts import DM_PROMPT
from pyarmx.vlm.visualizer import draw_bbox

from pyarmx.utils.log import logger


class VLMPlanner:
    """视觉语言模型检测器"""
    
    def __init__(
        self, 
        model_id: ModelID = ModelID.LMS
    ):
        """
        初始化 VLM 检测器
        
        Args:
            model_id: 模型ID
            auto_init: 是否自动初始化资源
        """
        config = get_config(model_id)
        config.system_prompt = DM_PROMPT

        self._bot = ChatBot(config)

    def analyze(self, img_cv2: np.ndarray, cmd: str) -> dict | None:
        """
        使用 VLM 根据指令规划动作或分析图像
        
        Args:
            img_cv2: OpenCV 图像矩阵
            cmd: 用户指令
            
        Returns:
            结果字典，失败时返回 None
        """

        img_b64 = self._bot.encode_img_cv2(img_cv2)

        ret_str = self._bot.chat(cmd, img_b64)
        if not ret_str:
            logger.error("模型响应为空")
            return None

        ret_dict = self._bot.json_loads(ret_str)
        if not ret_dict:
            logger.error("JSON解析失败")
            return None

        return ret_dict

    def draw(self, img_cv2: np.ndarray, ret_dict: dict):
        return draw_bbox(img_cv2, ret_dict)

    def show(self, img_draw: np.ndarray):
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    # img_path = r"img\pick\WIN_20260420_23_49_29_Pro.jpg"   # 
    # cmd = "把奶龙捡起来"

    # img_path = r"img\transport_nl\WIN_20260419_22_28_59_Pro.jpg"   # 
    # cmd = "把奶龙放到胶带上"
    # cmd = "把方块放到胶带上"

    # img_path = r"img\transport\WIN_20260419_22_23_32_Pro.jpg"   # 
    # cmd = "把奶龙放到胶带上"
    # cmd = "把方块放到胶带上"

    img_path = r"img\classify\WIN_20260419_23_57_58_Pro.jpg"   # 
    cmd = "把所有方块放到胶带上"

    planner = VLMPlanner()

    img_cv2 = cv2.imread(img_path)
    
    if img_cv2 is None:
        print("无法读取图片")
        exit()

    ret_dict = planner.analyze(img_cv2, cmd)
    if ret_dict is None:
        print("模型返回结果为空")
        exit()

    img_draw = planner.draw(img_cv2, ret_dict)
    planner.show(img_draw)