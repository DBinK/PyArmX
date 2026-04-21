from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True)
class Pose7D:
    """7D 位姿: [x, y, z, qx, qy, qz, qw]"""
    # 默认使用工厂函数提供内存分配，允许直接 Pose7D() 初始化
    _data: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=float))

    def __post_init__(self):
        # 确保数据格式和连续性
        self._data = np.asarray(self._data, dtype=float)
        if self._data.shape != (7,):
            raise ValueError(f"位姿数据需要 shape (7,), 传入的 shape 为 {self._data.shape}")

    # ===== 属性与视图访问 =====
    @property
    def array(self) -> np.ndarray:
        """返回底层 (7,) 数组的视图"""
        return self._data

    @property
    def pos(self) -> np.ndarray:
        """返回位置视图 (3,)"""
        return self._data[:3]

    @property
    def quat(self) -> np.ndarray:
        """返回四元数视图 (4,)"""
        return self._data[3:]
    
    # ===== 原地更新机制 (宽进) =====@array.setter
    @array.setter
    def array(self, value: np.ndarray | list[float] | tuple[float, ...]) -> None:
        """
        支持 pose.array = new_data 的直观语法。
        一口气更新完整的 7D 数据，底层使用 [:] 切片进行原地内存替换。
        """
        self._data[:] = value
        
    @pos.setter
    def pos(self, value: np.ndarray | list[float] | tuple[float, ...]) -> None:
        """
        支持 pose.pos = new_pos 的直观语法，底层为原地修改。
        允许传入 numpy 数组、列表或元组。
        """
        self._data[:3] = value

    @quat.setter
    def quat(self, value: np.ndarray | list[float] | tuple[float, ...]) -> None:
        """
        支持 pose.quat = new_quat 的直观语法，底层为原地修改。
        """
        self._data[3:] = value

    def update(
        self, 
        pos: np.ndarray | list[float] | tuple[float, ...], 
        quat: np.ndarray | list[float] | tuple[float, ...]
    ) -> None:
        """主循环高频调用的统一更新入口"""
        self._data[:3] = pos
        self._data[3:] = quat

    # ===== 领域运算 =====
    def normalize_quat(self):
        """原地归一化四元数，消除浮点数累积误差"""
        q = self._data[3:]
        norm = float(np.linalg.norm(q))
        if norm > 1e-8:
            self._data[3:] = q / norm
        else:
            # 应对奇异状态的兜底
            self._data[3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)