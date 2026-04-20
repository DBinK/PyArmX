import cv2
import numpy as np
from typing import NamedTuple
from pupil_apriltags import Detection, Detector


class Point2D(NamedTuple):
    """二维点坐标"""
    x: float
    y: float


class HomographyFilter:
    """单应性矩阵滤波器：通过对角点指数移动平均滤波实现平滑"""
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.previous_corners = None
    
    def update(self, detection: Detection) -> np.ndarray:
        corners = detection.corners
        if corners is None:
            return detection.homography  # type: ignore
        
        if self.previous_corners is None:
            self.previous_corners = corners.copy()
            return detection.homography  # type: ignore
        
        filtered_corners = self.alpha * corners + (1 - self.alpha) * self.previous_corners
        self.previous_corners = filtered_corners
        
        tag_size = 2.0
        object_points = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [tag_size/2, -tag_size/2, 0],
            [tag_size/2, tag_size/2, 0],
            [-tag_size/2, tag_size/2, 0]
        ], dtype=np.float32)
        
        image_points = filtered_corners.astype(np.float32)
        H, _ = cv2.findHomography(object_points[:, :2], image_points)
        return H
    
    def reset(self):
        self.previous_corners = None


class TagLocator:
    """封装 AprilTag 检测、过滤与物理坐标系转换的完整逻辑"""
    def __init__(self, filter_alpha: float = 0.1):
        self.detector = Detector(
            families="tag16h5",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.hmf = HomographyFilter(alpha=filter_alpha)

        self.scale_factor = 1/40  # 意为将40mm一格转为1mm一格
        self.tx = 0.0
        self.ty = 140  # 把桌面坐标原点向 Y 轴移动 140 mm

        self.H_desk2pix: np.ndarray | None = None
        self.H_pix2desk: np.ndarray | None = None

    def locate_target(self, img: np.ndarray, target_id: int) -> tuple[np.ndarray | None, np.ndarray | None, list[Detection]]:
        """
        输入原始图像，寻找指定 ID 并计算双向转换矩阵
        返回: (H_desk2pix, H_pix2desk, 过滤后的所有 tags)
        """
        img_pre = self._pre_process(img)
        detections = self.detector.detect(img_pre)
        valid_detections = self._filter_by_size(detections)
        
        target = next((d for d in valid_detections if d.tag_id == target_id), None)
        
        if target:
            H_raw = self.hmf.update(target)
            
            # 内部调用齐次矩阵运算进行缩放和平移 (40mm -> 1mm)
            self.H_desk2pix = self._scale_homo(H_raw, self.scale_factor)
            self.H_desk2pix = self._translate_homo(self.H_desk2pix, self.tx, self.ty)
            
            try:
                self.H_pix2desk = np.linalg.inv(self.H_desk2pix)
            except np.linalg.LinAlgError:
                self.H_pix2desk = None
                return self.H_desk2pix, None, valid_detections
                
            return self.H_desk2pix, self.H_pix2desk, valid_detections
            
        return self.H_desk2pix, self.H_pix2desk, valid_detections
    
    
    def desk_to_pixel(self, point: Point2D) -> Point2D | None:
        """将桌面坐标转换为像素坐标(int)"""
        if self.H_desk2pix is not None:
            px, py = self.apply_transform(point, self.H_desk2pix)
            return Point2D(int(px), int(py))
        return None
        
    def pixel_to_desk(self, point: Point2D) -> Point2D | None:
        """将像素坐标转换为桌面坐标(float)"""
        if self.H_pix2desk is not None:
            return self.apply_transform(point, self.H_pix2desk)
        return None
    

    @staticmethod
    def apply_transform(point: Point2D, H: np.ndarray) -> Point2D:
        """执行齐次坐标系转换 (像素与物理坐标互转)"""
        p = np.array([point.x, point.y, 1.0], dtype=np.float32)
        p_trans = H @ p
        if p_trans[2] != 0:
            p_trans /= p_trans[2]
        return Point2D(float(p_trans[0]), float(p_trans[1]))


    @staticmethod
    def _pre_process(img: np.ndarray) -> np.ndarray:
        """内部图像预处理"""
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return img_bin

    @staticmethod
    def _filter_by_size(detections: list[Detection], min_size=(100, 100), max_size=(2000, 2000)) -> list[Detection]:
        """内部大小过滤"""
        valid_tags = []
        for detection in detections:
            x_coords = detection.corners[:, 0]  # type: ignore
            y_coords = detection.corners[:, 1]  # type: ignore
            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            
            if (min_size[0] <= width <= max_size[0]) and (min_size[1] <= height <= max_size[1]):
                valid_tags.append(detection)
        return valid_tags

    @staticmethod
    def _scale_homo(H: np.ndarray, scale_factor: float) -> np.ndarray:
        """内部矩阵缩放"""
        scale_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]], dtype=np.float64)
        return H @ scale_matrix

    @staticmethod
    def _translate_homo(H: np.ndarray, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
        """内部矩阵平移"""
        translate_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        return H @ translate_matrix


class TagVisualizer:
    """封装所有可视化与调试绘图功能"""
    @staticmethod
    def draw_point(img: np.ndarray, point: Point2D, color: list=[0, 0, 255], text: str = ""):
        """在指定位置绘制点与文字"""
        img_draw = img.copy()
        cv2.circle(img_draw, (int(point.x), int(point.y)), 3, color, -1)
        
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # 获取文本大小以绘制背景
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 计算背景矩形坐标
            x = int(point.x) + 5
            y = int(point.y) - 5
            
            # 绘制背景矩形 (黑色半透明或实心黑底，这里用实心黑底保证对比度)
            cv2.rectangle(img_draw, 
                          (x, y - text_height - baseline), 
                          (x + text_width, y + baseline), 
                          (0, 0, 0), 
                          -1)
            
            # 绘制文字
            cv2.putText(img_draw, text, (x, y), font, font_scale, (255, 255, 255), thickness)
            
        return img_draw

    @staticmethod
    def draw_tags(img: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """绘制所有 AprilTag"""
        img_draw = img.copy()
        for detection in detections:
            corners = detection.corners
            center = detection.center
            if corners is None or center is None:
                continue
                
            for i in range(4):
                cv2.line(img_draw, tuple(corners[i].astype(int)), tuple(corners[(i + 1) % 4].astype(int)), (0, 255, 0), 3)
                         
            center_text = int(center[0] - 15), int(center[1] + 12)
            cv2.putText(img_draw, f"{detection.tag_id}", center_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 222), 2)
        return img_draw

    @staticmethod
    def draw_grid(img: np.ndarray, homography: np.ndarray, step: int = 1, range_limit: int = 500) -> np.ndarray:
        """绘制网格
        Args:
            img (np.ndarray): 输入图像
            homography (np.ndarray): 投影矩阵
            step (int, optional): 网格步长. Defaults to 1.
            range_limit (int, optional): 网格范围. Defaults to 500.
        """
        img_draw = img.copy()
        h, w = img.shape[:2]
        for x in range(-range_limit, range_limit + 1, step):
            for y in range(-range_limit, range_limit + 1, step):
                p_tag = np.array([x, y, 1.0], dtype=np.float32)
                p_img = homography @ p_tag
                if p_img[2] == 0:
                    continue
                p_img = p_img / p_img[2]
                px, py = int(p_img[0]), int(p_img[1])

                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(img_draw, (px, py), 2, (0, 255, 255), -1)
                    if x % 100 == 0 and y % 100 == 0:
                        cv2.putText(img_draw, f"{x},{y}", (px + 3, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
        return img_draw
