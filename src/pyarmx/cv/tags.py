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

def get_tag_size(corners):
    # 获取角点坐标的 NumPy 数组
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    
    # 计算宽度和高度
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)

    # print(f"Tag size: width={width}, height={height}.")

    return width, height


def homo_trans(corners: list[float], width=int(1), height=int(1)):

    """ 计算 将图像中的特定四边形区域 变换为目标长方形区域 所需的矩阵 H """

    # 定义图像中的长方形四个顶点（根据你实际值设定）
    image_points = np.array(corners, dtype='float32')

    # 定义目标长方形的四个顶点
    object_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

    # 计算同伦变换矩阵
    H_matrix , _ = cv2.findHomography(image_points, object_points)
    
    if H_matrix is None:
        raise ValueError("无法计算单应性矩阵")
    
    H_matrix = np.asarray(H_matrix, dtype=np.float64)

    # 计算反向变换矩阵
    H_inv = np.linalg.inv(H_matrix)

    return H_matrix, H_inv


def filter_by_size(detections: list[Detection], min_size=(100, 100), max_size=(2000, 2000)) -> list[Detection]:
    valid_tags = []
    for detection in detections:
        corners = detection.corners  
        # tag_id = detection.tag_id  
        # x, y = detection.center.tolist()

        # 计算标签大小
        width, height = get_tag_size(corners)

        # 根据大小过滤标签
        if (min_size[0] <= width <= max_size[0]) and (min_size[1] <= height <= max_size[1]):
            # print(f"Tag {tag_id} , center={(x, y)}")
            valid_tags.append(detection)
        else:
            # print(f"Tag {tag_id} is filtered out due to size: width={width}, height={height}.")
            continue
    
    return valid_tags
    
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


def draw_integer_grid(img, homography, step=1, range_limit=100, homography_filter=None):
    """
    在图像上绘制 tag 平面整数坐标点
    :param homography_filter: 可选的单应性矩阵滤波器
    """
    img_draw = img.copy()
    
    H = homography

    h, w = img.shape[:2]

    for x in range(-range_limit, range_limit + 1, step):
        for y in range(-range_limit, range_limit + 1, step):

            p_tag = np.array([x, y, 1.0], dtype=np.float32)
            p_img = H @ p_tag

            # 齐次归一化
            if p_img[2] == 0:
                continue
            p_img = p_img / p_img[2]

            px, py = int(p_img[0]), int(p_img[1])

            # 只画在图像内的点
            if 0 <= px < w and 0 <= py < h:

                # 点
                cv2.circle(img_draw, (px, py), 2, (0, 255, 255), -1)

                # 可选：标坐标（太密会很乱）
                if x % 10 == 0 and y % 10 == 0:
                    cv2.putText(
                        img_draw,
                        f"{x},{y}",
                        (px + 3, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 200, 255),
                        1
                    )

    return img_draw


class HomographyFilter:
    """单应性矩阵滤波器 - 通过对角点滤波实现"""
    
    def __init__(self, alpha=0.7):
        """
        初始化滤波器
        :param alpha: 滤波系数，越大响应越快但噪声越多，越小越平滑但延迟越高
        """
        self.alpha = alpha
        self.previous_corners = None
    
    def update(self, detection: Detection) -> np.ndarray:
        """
        更新并返回滤波后的单应性矩阵
        :param detection: AprilTag检测结果
        :return: 滤波后的3x3单应性矩阵
        """
        corners = detection.corners
        
        if corners is None:
            return detection.homography
        
        if self.previous_corners is None:
            self.previous_corners = corners.copy()
            return detection.homography
        
        # 对角点进行指数移动平均滤波
        filtered_corners = self.alpha * corners + (1 - self.alpha) * self.previous_corners
        self.previous_corners = filtered_corners
        
        # 使用滤波后的角点重新计算单应性矩阵
        tag_size = 2.0  # tag的标准尺寸（可根据实际情况调整）
        object_points = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [tag_size/2, -tag_size/2, 0],
            [tag_size/2, tag_size/2, 0],
            [-tag_size/2, tag_size/2, 0]
        ], dtype=np.float32)
        
        image_points = filtered_corners.astype(np.float32)
        
        # 计算新的单应性矩阵
        H, _ = cv2.findHomography(object_points[:, :2], image_points)
        
        return H
    
    def reset(self):
        """重置滤波器状态"""
        self.previous_corners = None



if __name__ == "__main__":

    from rich import print as rprint

    # 读取图像
    img_path = r"img\transport_nl\WIN_20260419_22_28_59_Pro.jpg"
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

    img_draw = draw_integer_grid(img, detections[0])
    cv2.imshow("Warped Image", img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(detections[0].homography)
    print(detections[0].tag_id)

