
import cv2

from pyarmx.cv.camera import UVCamera, CamCfg

from pyarmx.vlm.chatbot import ChatBot
from pyarmx.vlm.config import ModelID, get_config

from pyarmx.vlm.prompts import DM_PROMPT
from pyarmx.vlm.visualizer import draw_bbox

config = get_config(ModelID.LMS)
config.system_prompt = DM_PROMPT

bot = ChatBot(config)

# text = bot.chat("你好, 你是谁")


# img_path = r"img\pick\WIN_20260420_23_49_29_Pro.jpg"   # 
# cmd = "把奶龙捡起来"

# img_path = r"img\transport_nl\WIN_20260419_22_28_59_Pro.jpg"   # 
# cmd = "把奶龙放到胶带上"
# cmd = "把方块放到胶带上"

# img_path = r"img\transport\WIN_20260419_22_23_32_Pro.jpg"   # 
# cmd = "把奶龙放到胶带上"
# cmd = "把方块放到胶带上"

# img_path = r"img\classify\WIN_20260419_23_57_58_Pro.jpg"   # 
cmd = "把所有方块放到胶带上"

# img_b64 = bot.encode_img_path(img_path)
# img_cv2 = cv2.imread(img_path)

# 直接拍照
cam = UVCamera(CamCfg(cam_id=1))

cmd = "把所有方块放到胶带上"
ret, img_cv2 = cam.read()

if ret:
    cv2.imwrite("img/vlm/img_cv2.jpg", img_cv2)

img_b64 = bot.encode_img_cv2(img_cv2)

ret_str = bot.chat(cmd, img_b64)

if ret_str is None:
    exit()

ret_dict = bot.json_loads(ret_str)

if ret_dict is None:
    exit()

img_ret = draw_bbox(img_cv2, ret_dict, cmd)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img_ret)

cv2.imwrite("img/vlm/img_ret.jpg", img_ret)

cv2.waitKey(0)
cv2.destroyAllWindows()
