import cv2
import random
import json
from cv2.typing import NumPyArrayNumeric
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_text_img(
    text, font_path=r"C:\Windows\Fonts\simhei.ttf", font_size=24, color=(255, 255, 255)
):
    """生成透明背景文字图（BGRA）"""
    # 创建字体对象
    font = ImageFont.truetype(font_path, font_size)

    # 创建一个空白图像以获取文本的尺寸
    temp_image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(temp_image)
    bbox = draw.textbbox((0, 0), text, font=font)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 创建足够大的图像来容纳文本
    image = Image.new("RGB", (width, height), (0, 0, 0))  # type: ignore
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill=color)

    return image


def overlay_image(background, foreground, x_offset, y_offset):
    fg = np.array(foreground)  # 将 PIL 图像转换为 NumPy 数组
    fh, fw = fg.shape[:2]
    bg = background.copy()
    bg[y_offset : y_offset + fh, x_offset : x_offset + fw] = fg
    return bg


def draw_bbox(
    cv2_img,
    data: dict,
    output_path: str | None = None,
    normalized_range: float | None = 1000.0,
) -> cv2.Mat:
    """
    根据输入字典绘制边界框 (适配objs字典格式)

    Args:
        cv2_img (cv2.Mat): 输入图片
        data (dict): 输入数据，包含objs字段
        output_path (str | None): 输出图片路径，如果为None则不保存
        normalized_range (float | None): 如果传入，则表示输入坐标是归一化的
    """
    # img = cv2.imread(input_path)

    img = cv2_img.copy()
    
    if img is None:
        raise ValueError(f"无法读取图片: {cv2_img}")


    h, w = img.shape[:2]

    objs = data.get("objs", {})

    # 给每个类别随机分配颜色
    colors = {
        name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for name in objs
    }

    # 绘制边界框和标签
    for name, box in objs.items():
        x1, y1, x2, y2 = box
        if normalized_range:
            x1 = int(x1 * w / normalized_range)
            x2 = int(x2 * w / normalized_range)
            y1 = int(y1 * h / normalized_range)
            y2 = int(y2 * h / normalized_range)
        else:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # 保证边界框在图片范围内
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        color = colors[name]
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 绘制文字标签
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1
        )
        cv2.putText(
            img, name, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"结果已保存到: {output_path}")

    return img


if __name__ == "__main__":
    image_path = "tmp/test1.png"

    # 紧凑数组格式
    result_json_fix = """
{
  "say": "目标物体为桌面上的香蕉、柠檬、苹果和猕猴桃，我需要将它们依次放到键盘上。",
  "task": "依次抓取香蕉、柠檬、苹果和猕猴桃，移动到键盘位置，然后放下。",
  "acts": [
    ["moveTo", "banana"],
    ["grip", "banana"],
    ["moveTo", "keyboard"],
    ["release"],
    ["moveTo", "lemon"],
    ["grip", "lemon"],
    ["moveTo", "keyboard"],
    ["release"],
    ["moveTo", "apple"],
    ["grip", "apple"],
    ["moveTo", "keyboard"],
    ["release"],
    ["moveTo", "kiwi"],
    ["grip", "kiwi"],
    ["moveTo", "keyboard"],
    ["release"]
  ],
  "objs": {
    "banana": [160, 230, 215, 310],
    "lemon": [210, 302, 255, 360],
    "apple": [320, 245, 355, 308],
    "kiwi": [300, 290, 340, 338],
    "keyboard": [200, 135, 400, 195]
  }
}
    """

    prompt = "把奶龙放到白色托盘里"

    result_list = json.loads(result_json_fix)
    
    cv2_img = cv2.imread(image_path)

    img: cv2.Mat = draw_bbox(cv2_img, result_list)
    cv2.imshow("YOLO", img)

    img = draw_bbox(cv2_img, result_list,  None, None)
    cv2.imshow("YOLO_NORM", img)
    cv2.waitKey(0)
