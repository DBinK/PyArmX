from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Pose7D:
    """7D 位姿: [x, y, z, qx, qy, qz, qw]"""
    data: np.ndarray  # shape (7,)

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.shape != (7,):
            raise ValueError("需要 shape (7,)")

    # ===== 视图访问（零拷贝）=====
    @property
    def pos(self) -> np.ndarray:
        return self.data[:3]  # view

    @property
    def quat(self) -> np.ndarray:
        return self.data[3:]  # view
    
    def to_array(self) -> np.ndarray:
        return self.data
    
    # ===== 原地更新 =====
    def set_pos(self, value):
        self.data[:3] = value

    def set_quat(self, value):
        self.data[3:] = value

    def normalize_quat(self):
        q = self.data[3:]
        self.data[3:] = q / np.linalg.norm(q)