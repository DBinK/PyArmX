from typing import TypeAlias

import cv2
import numpy as np
from pupil_apriltags import Detector
from pupil_apriltags.bindings import Detection

MatLike: TypeAlias = cv2.typing.MatLike

# 初始化一个Detector实例，用于检测和解码Apriltag标记
at_detector = Detector(
    families="tag16h5",  # 指定使用的Apriltag家族，这里选择的是'tag16h5'
    nthreads=4,  # 设置用于加速计算的线程数，此处设置为1
    quad_decimate=1.0,  # 设置图像简化比例，用于加快处理速度，1.0表示不简化
    quad_sigma=0.0,  # 指定在检测标记前对图像进行高斯模糊的程度，0.0表示不进行模糊处理
    refine_edges=1,  # 设置是否对检测到的标记边缘进行精细化处理，以提高定位精度
    decode_sharpening=0.25,  # 设置解码过程中的图像锐化程度，以提高解码成功率
    debug=0,  # 设置调试模式级别，0表示不启用调试模式
)

def pre_process(img: MatLike):
    # 预处理
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)  # 应用高斯滤波以平滑图像
    _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化

    return img_bin


# 鼠标回调函数
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 检测左键点击事件
        
        #p_raw = [x, y]
        #p_obj = transform_image_to_object(p_raw, H_matrix)
        #p_obj_fix = [(p_obj[0]/10)-20, (p_obj[1]/10)-20]

        print(f'点击坐标: ({x}, {y})')  # 打印点击的坐标
        # print(f'转换坐标: ({p_obj[0]}, {p_obj[1]})')
        # print(f'转换坐标fix: ({p_obj_fix[0]}, {p_obj_fix[1]})')

        # cv2.putText(img_trans, f"{int(p_obj_fix[0])},{int(p_obj_fix[1])}", (int(p_obj[0]+100), int(p_obj[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        # cv2.circle(img_trans, (int(p_obj[0]), int(p_obj[1])), 50, (0, 0, 255), -1)
        # cv2.imshow('Warped Image', img_raw)  # 重新显示图像

def draw_tags(img: MatLike, detections: list[Detection]):

    img_draw = img.copy()

    # 绘制检测结果
    for detection in detections:

        corners = detection.corners
        center = detection.center

        if corners is None or center is None:
            continue

        # 绘制边界框
        for i in range(4):
            cv2.line(
                img_draw,
                tuple(corners[i].astype(int)),
                tuple(corners[(i + 1) % 4].astype(int)),
                (0, 255, 0),
                3,
            )  # 绿色线条

        # 在中心绘制标签 ID
        center = int(center[0]-15), int(center[1]+12)
        cv2.putText(
            img_draw,
            f"{detection.tag_id}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 222),
            2,
        )  # 红色文本

    return img_draw


if __name__ == "__main__":

    from rich import print as rprint

    # 读取图像
    img_path = "img/blk_sen.jpg"
    # img_path = "img/tags.jpg"
    img = cv2.imread(img_path)

    # 预处理
    if img is None:
        print("Error: Could not read image.")
        exit()

    img_pre = pre_process(img)

    # 检测标记
    detections: list[Detection] = at_detector.detect(img_pre)  # type: ignore # 注: 这个返回的类型是列表, 库里面的注解是错的

    rprint(detections)

    img_draw = draw_tags(img, detections)
    cv2.imshow("Warped Image", img_draw)
    cv2.setMouseCallback("Warped Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()