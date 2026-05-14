import cv2
import numpy as np
from typing import NamedTuple
from pupil_apriltags import Detection, Detector


class Point2D(NamedTuple):
    """二维点坐标"""
    x: float
    y: float

class TagResult(NamedTuple):
    """Tag检测结果"""
    success: bool
    H_desk2pix: np.ndarray | None
    H_pix2desk: np.ndarray | None
    detections: list[Detection]

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
        self.ty = 136  # 把桌面坐标原点向 Y 轴移动 140 mm

        self.H_desk2pix: np.ndarray | None = None
        self.H_pix2desk: np.ndarray | None = None

    def locate_target(self, img: np.ndarray, target_id: int) -> TagResult:
        """
        输入原始图像，寻找指定 ID 并计算双向转换矩阵
        返回: TagResult 包含成功标志、变换矩阵和检测结果
        """
        img_pre = self._pre_process(img)
        detections = self.detector.detect(img_pre)
        valid_detections = self._filter_by_size(detections)
        
        target = next((d for d in valid_detections if d.tag_id == target_id), None)
        
        if not target:
            return TagResult(False, None, None, valid_detections)
            
        H_raw = self.hmf.update(target)
        
        # 内部调用齐次矩阵运算进行缩放和平移 (40mm -> 1mm)
        H_desk2pix = self._scale_homo(H_raw, self.scale_factor)
        H_desk2pix = self._translate_homo(H_desk2pix, self.tx, self.ty)
        
        try:
            H_pix2desk = np.linalg.inv(H_desk2pix)
        except np.linalg.LinAlgError:
            return TagResult(False, None, None, valid_detections)
            
        # 更新实例变量供其他方法使用
        self.H_desk2pix = H_desk2pix
        self.H_pix2desk = H_pix2desk
        
        return TagResult(True, H_desk2pix, H_pix2desk, valid_detections)
    
    
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
    def draw_grid(img: np.ndarray, homography: np.ndarray, step: int = 20, range_limit: int = 500) -> np.ndarray:
        """基于纯透视变换的双端点极致优化版"""
        img_draw = img.copy()
        h, w = img.shape[:2]

        # 生成刻度序列
        ticks = np.arange(-range_limit, range_limit + 1, step)
        
        # 构建所有网格线的起点和终点坐标 (N, 2)
        # 垂直线: x 不变, y 从 -limit 到 +limit
        v_starts = np.column_stack((ticks, np.full_like(ticks, -range_limit)))
        v_ends = np.column_stack((ticks, np.full_like(ticks, range_limit)))
        
        # 水平线: y 不变, x 从 -limit 到 +limit
        h_starts = np.column_stack((np.full_like(ticks, -range_limit), ticks))
        h_ends = np.column_stack((np.full_like(ticks, range_limit), ticks))

        # 合并所有起点和终点
        all_starts = np.vstack([v_starts, h_starts])
        all_ends = np.vstack([v_ends, h_ends])
        
        # 为了只做一次矩阵乘法，将起点和终点拼接到一起 (2N, 2)
        all_points = np.vstack([all_starts, all_ends])

        # 转换为齐次坐标并执行一次性批量变换 (3, 2N)
        pts_h = np.vstack([all_points.T, np.ones(len(all_points))])
        projected = homography @ pts_h

        # 透视除法，并加上微小 epsilon 防止除以 0
        z = projected[2, :]
        z[z == 0] = 1e-6
        pts_img = (projected[:2, :] / z).T.astype(np.int32)

        # 从大矩阵中重新分离出 起点 和 终点
        num_lines = len(all_starts)
        proj_starts = pts_img[:num_lines]
        proj_ends = pts_img[num_lines:]

        # OpenCV 的 cv2.line 底层有高效的裁剪算法，直接遍历绘制即可
        for pt1, pt2 in zip(proj_starts, proj_ends):
            cv2.line(img_draw, tuple(pt1), tuple(pt2), (100, 100, 100), 1)

        # 绘制文字标签 (仅限关键点)
        label_ticks = np.arange(-range_limit, range_limit + 1, 100)
        if len(label_ticks) > 0:
            grid_x, grid_y = np.meshgrid(label_ticks, label_ticks)
            labels_pts = np.vstack([grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)])
            
            l_proj = homography @ labels_pts
            lz = l_proj[2, :]
            lz[lz == 0] = 1e-6
            l_img = (l_proj[:2, :] / lz).T.astype(np.int32)

            for i, (px, py) in enumerate(l_img):
                if 0 <= px < w and 0 <= py < h:
                    lx, ly = grid_x.ravel()[i], grid_y.ravel()[i]
                    cv2.putText(img_draw, f"{lx},{ly}", (px + 2, py - 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)


        return img_draw
    

    @staticmethod
    def draw_axes_2d(img: np.ndarray, homography: np.ndarray, length: float = 200.0) -> np.ndarray:
        """绘制桌面坐标系 XY 轴 (X红 Y绿)"""

        img_draw = img.copy()

        def proj(pt: tuple[float, float]) -> np.ndarray:
            """桌面坐标 -> 像素坐标"""
            p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
            p = homography @ p
            p /= (p[2] + 1e-6)
            return p[:2]

        # 原点 + 两个轴端点
        origin = proj((0, 0))
        x_end = proj((length, 0))
        y_end = proj((0, length))

        o = tuple(origin.astype(int))
        x = tuple(x_end.astype(int))
        y = tuple(y_end.astype(int))

        # 画轴
        cv2.line(img_draw, o, x, (0, 0, 255), 3)  # X
        cv2.line(img_draw, o, y, (0, 255, 0), 3)  # Y

        # 标注
        cv2.putText(img_draw, "X", (x[0]+5, x[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(img_draw, "Y", (y[0]+5, y[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        return img_draw

    @classmethod
    def draw_tag_result(cls, img: np.ndarray, tag_ret: TagResult) -> np.ndarray:
        """绘制识别结果"""
        
        if not tag_ret.success:
            return img
        
        assert tag_ret.H_desk2pix is not None, "Success is True but H_desk2pix is None"
        
        img_draw = img

        img_draw = cls.draw_grid(img_draw, tag_ret.H_desk2pix)
        img_draw = cls.draw_axes_2d(img_draw, tag_ret.H_desk2pix)
        img_draw = cls.draw_tags(img_draw, tag_ret.detections)

        return img_draw
        