from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(slots=True)
class Pose7D:
    """7D 位姿: [x, y, z, qx, qy, qz, qw]"""
    # 默认使用工厂函数提供内存分配，允许直接 Pose7D() 初始化
    _data: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float
        )  # 默认初始化为合法的单位位姿: 位置 [0,0,0], 单位四元数 [0,0,0,1]
    )

    def __post_init__(self):
        # 确保数据格式和连续性
        self._data = np.asarray(self._data, dtype=float)
        if self._data.shape != (7,):
            raise ValueError(
                f"位姿数据需要 shape (7,), 传入的 shape 为 {self._data.shape}"
            )

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
        quat: np.ndarray | list[float] | tuple[float, ...],
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

    # ===== 数据转换 =====
    @property
    def rot_mat(self) -> np.ndarray:
        """返回 3x3 旋转矩阵 (shape: 3, 3)"""
        # scipy 的 from_quat 默认接收 (x, y, z, w) 格式，与我们一致
        return R.from_quat(self._data[3:]).as_matrix()

    @property
    def homo_mat(self) -> np.ndarray:
        """返回 4x4 齐次变换矩阵 (shape: 4, 4)"""
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = self.rot_mat
        mat[:3, 3] = self._data[:3]
        return mat

    @property
    def euler(self) -> np.ndarray:
        """
        返回欧拉角 [rx, ry, rz] (弧度制)。
        默认使用 'xyz' (固定轴外旋/Roll-Pitch-Yaw) 顺序。
        """
        return R.from_quat(self._data[3:]).as_euler("xyz", degrees=False)

    def get_euler(self, seq: str = "xyz", degrees: bool = False) -> np.ndarray:
        """
        提供更灵活的欧拉角获取方式。
        可自定义旋转顺序 (如 'zyx', 'ZYZ' 等) 和是否返回角度制。
        """
        return R.from_quat(self._data[3:]).as_euler(seq, degrees=degrees)  # type: ignore

    # ===== 逆向工厂方法 =====
    @classmethod
    def from_pos_quat(
        cls,
        pos: np.ndarray | list[float] | tuple[float, ...],
        quat: np.ndarray | list[float] | tuple[float, ...],
    ) -> "Pose7D":
        """
        从分离的位置和四元数构造。
        最基础的组装方法，省去先实例化再调用 update 的繁琐。
        """
        instance = cls()
        instance.update(pos, quat)
        return instance

    @classmethod
    def from_pos_euler(
        cls,
        pos: np.ndarray | list[float] | tuple[float, ...],
        euler: np.ndarray | list[float] | tuple[float, ...],
        seq: str = "xyz",
        degrees: bool = False,
    ) -> "Pose7D":
        """
        从位置和欧拉角构造。
        非常适合解析人类手写输入的配置 (UI 参数、示教器端点微调等)。
        """
        # 利用 scipy 的强大能力，将欧拉角安全转换为四元数
        quat = R.from_euler(seq, euler, degrees=degrees).as_quat()
        instance = cls()
        instance.update(pos, quat)
        return instance

    @classmethod
    def from_pos_rot_mat(
        cls,
        pos: np.ndarray | list[float] | tuple[float, ...],
        rot_mat: np.ndarray | list[list[float]],
    ) -> "Pose7D":
        """
        从位置和 3x3 旋转矩阵构造。
        通常用于对接 OpenCV 视觉相机标定或手眼标定算法输出的 R 和 t。
        """
        quat = R.from_matrix(rot_mat).as_quat()
        instance = cls()
        instance.update(pos, quat)
        return instance

    @classmethod
    def from_homo_mat(cls, mat: np.ndarray | list[list[float]]) -> "Pose7D":
        """从 4x4 齐次变换矩阵构造 Pose7D 对象"""
        mat_arr = np.asarray(mat, dtype=float)
        if mat_arr.shape != (4, 4):
            raise ValueError(
                f"齐次矩阵需要 shape (4, 4), 传入的 shape 为 {mat_arr.shape}"
            )

        pos = mat_arr[:3, 3]
        quat = R.from_matrix(mat_arr[:3, :3]).as_quat()

        instance = cls()
        instance.update(pos, quat)
        return instance
    
    def __matmul__(self, other: "Pose7D") -> "Pose7D":
        """
        重载 @ 运算符，实现位姿的级联变换 (Pose Composition)。
        用法: pose_C = pose_A @ pose_B
        """
        if not isinstance(other, Pose7D):
            return NotImplemented  # 允许 Python 尝试其他类型的反向运算
        
        # 1. 旋转部分的级联: R_new = R_A * R_B
        r_a = R.from_quat(self.quat)
        r_b = R.from_quat(other.quat)
        r_new = r_a * r_b  # scipy 中 Rotation 的乘法就是连续旋转的组合
        
        # 2. 平移部分的级联: P_new = P_A + R_A * P_B
        # r_a.apply() 会将 other.pos 这个向量，用 r_a 进行旋转变换
        p_new = self.pos + r_a.apply(other.pos)
        
        # 3. 返回全新的位姿对象
        return Pose7D.from_pos_quat(p_new, r_new.as_quat())