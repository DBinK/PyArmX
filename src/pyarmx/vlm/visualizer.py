import cv2
import random
import json
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
    fg = np.array(foreground)
    fh, fw = fg.shape[:2]
    bg = background.copy()
    
    h, w = bg.shape[:2]
    
    # 计算背景上的有效区域
    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    x_end = min(w, x_offset + fw)
    y_end = min(h, y_offset + fh)
    
    # 计算前景图像对应的裁剪区域
    fg_x_start = max(0, -x_offset)
    fg_y_start = max(0, -y_offset)
    fg_x_end = fg_x_start + (x_end - x_start)
    fg_y_end = fg_y_start + (y_end - y_start)
    
    # 只有在有有效重叠区域时才进行赋值
    if x_end > x_start and y_end > y_start:
        bg[y_start:y_end, x_start:x_end] = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
    
    return bg


def draw_bbox(
    img: np.ndarray,
    data: dict,
    prompt: str | None = None,
    output_path: str | None = None,
    normalized_range: float | None = 1000,
):
    """
    根据输入字典绘制边界框 (适配objs字典格式)

    Args:
        image_path (str): 图片路径
        data (dict): 输入数据，包含objs字段
        output_path (str | None): 输出图片路径，如果为None则不保存
        normalized_range (float | None): 如果传入，则表示输入坐标是归一化的
    """
    h, w = img.shape[:2]

    objs = data.get("objs", {})
    acts = data.get("acts")
    say = data.get("say")
    task = data.get("task")

    # 中文任务和 say 用贴图方式
    render_font_size = 28
    if prompt is not None:
        text_img = render_text_img(
            f"prompt: {prompt}", font_size=render_font_size, color=(255, 255, 0)
        )
        img = overlay_image(img, text_img, 10, h - render_font_size*3-30)
    if say:
        text_img = render_text_img(f"say: {say}", font_size=render_font_size, color=(255, 255, 255))
        img = overlay_image(img, text_img, 10, h - render_font_size*2-20)
    if task:
        text_img = render_text_img(f"task: {task}", font_size=render_font_size, color=(255, 255, 255))
        img = overlay_image(img, text_img, 10, h - render_font_size*1-10)

    # 显示任务描述（如果存在）
    if acts:    
        # 添加半透明遮罩
        mask_percentage = 25
        overlay = img.copy()
        mask_width = int(w * mask_percentage / 100)  # 计算遮罩宽度
        cv2.rectangle(overlay, (0, 0), (mask_width, h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        # 设置字体和大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.80
        thickness = 2

        # 计算所有动作文本的尺寸
        act_strings = [
            f"{act[0]}({act[1]})" if len(act) > 1 else act[0] for act in acts
        ]
        line_height = 30  # 每行的高度
        start_y = 30  # 起始Y位置

        # 绘制每个动作
        for i, act_str in enumerate(act_strings):
            y_position = start_y + i * line_height
            # 确保不会绘制超出图像范围
            if y_position < h - 10:
                cv2.putText(
                    img,
                    act_str,
                    (10, y_position),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )
            else:
                # 如果超出范围，显示省略号
                cv2.putText(
                    img,
                    "...",
                    (10, h - 10),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )
                break

    # 给每个类别随机分配颜色
    colors = {
        name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for name in objs
    }

    # 绘制边界框和标签
    for name, box in objs.items():
        # print(box)
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
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(
            img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1
        )
        cv2.putText(
            img, name, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"结果已保存到: {output_path}")

    return img


if __name__ == "__main__":
    img_path = r"img\vlm\img_cv2.jpg" 
    img_cv2 = cv2.imread(img_path)

    result_json_fix = """
{
  "say": "画面中有五个方块，正在将它们逐一放到胶带上",
  "task": "依次移动到黑色、红色和黄色方块处进行抓取，然后移动到胶带上方释放",
  "acts": [
    ["move_to", "black_block"],
    ["grip", "black_block"],
    ["move_to", "tape"],
    ["release"],
    ["move_to", "red_block_1"],
    ["grip", "red_block_1"],
    ["move_to", "tape"],
    ["release"],
    ["move_to", "red_block_2"],
    ["grip", "red_block_2"],
    ["move_to", "tape"],
    ["release"],
    ["move_to", "yellow_block_1"],
    ["grip", "yellow_block_1"],
    ["move_to", "tape"],
    ["release"],
    ["move_to", "yellow_block_2"],
    ["grip", "yellow_block_2"],
    ["move_to", "tape"],
    ["release"]
  ],
  "objs": {
    "black_block": [610, 193, 652, 253],
    "red_block_1": [728, 168, 770, 240],
    "red_block_2": [675, 324, 724, 390],
    "yellow_block_1": [599, 490, 646, 561],
    "yellow_block_2": [782, 333, 837, 413],
    "tape": [715, 460, 879, 723]
  }
}

    """

    prompt = "把奶龙放到白色托盘里"

    result_list = json.loads(result_json_fix)

    # img = draw_bbox(img_path, result_list)
    # cv2.imshow("YOLO", img)

    img = draw_bbox(img_cv2, result_list, prompt, None, 1000.0) # type: ignore
    cv2.namedWindow("YOLO_NORM", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLO_NORM", img)
    cv2.waitKey(0)