from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass(slots=True)
class Pose:
    """
    严谨且工程友好的 SE(3) 位姿类。
    底层使用 t + Rotation 确保数学安全性，上层提供丰富的兼容与组装接口。
    """
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    R: Rotation = field(default_factory=Rotation.identity)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        if self.t.shape != (3,):
            raise ValueError(f"平移向量 t 必须是 (3,)，当前为 {self.t.shape}")
        
        # 兜底保护：防止直接传入四元数数组初始化 R 导致报错
        if not isinstance(self.R, Rotation):
            self.R = Rotation.from_quat(self.R)

    # ==========================================
    # 1. 属性视图与原地更新机制 (完美兼容老接口)
    # ==========================================
    @property
    def pos(self) -> np.ndarray:
        """返回平移向量的视图"""
        return self.t

    @pos.setter
    def pos(self, value: np.ndarray | list[float] | tuple[float, ...]) -> None:
        """位置支持内存连续的原地切片更新"""
        self.t[:] = value

    @property
    def quat(self) -> np.ndarray:
        """按需生成并返回四元数数组"""
        return self.R.as_quat()

    @quat.setter
    def quat(self, value: np.ndarray | list[float] | tuple[float, ...]) -> None:
        """更新四元数时，自动生成新的安全 Rotation 对象"""
        self.R = Rotation.from_quat(value)

    @property
    def array(self) -> np.ndarray:
        """当底层 C++ 接口需要 7D 数组时，直接拼接返回"""
        return np.concatenate([self.t, self.R.as_quat()])

    @array.setter
    def array(self, value: np.ndarray | list[float] | tuple[float, ...]) -> None:
        """支持 pose.array = new_data 语法，一口气更新 7D 状态"""
        arr = np.asarray(value, dtype=float)
        self.t[:] = arr[:3]
        self.R = Rotation.from_quat(arr[3:])

    def update(
        self, 
        pos: np.ndarray | list[float] | tuple[float, ...], 
        quat: np.ndarray | list[float] | tuple[float, ...]
    ) -> None:
        """主循环高频调用的统一更新入口"""
        self.t[:] = pos
        self.R = Rotation.from_quat(quat)

    # ==========================================
    # 2. 数据转换接口
    # ==========================================
    @property
    def rot_mat(self) -> np.ndarray:
        """返回 3x3 旋转矩阵"""
        return self.R.as_matrix()

    @property
    def homo_mat(self) -> np.ndarray:
        """返回 4x4 齐次变换矩阵"""
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = self.R.as_matrix()
        mat[:3, 3] = self.t
        return mat

    @property
    def euler(self) -> np.ndarray:
        """默认使用 'xyz' (固定轴外旋/RPY) 顺序返回弧度制欧拉角"""
        return self.R.as_euler('xyz', degrees=False)

    def get_euler(self, seq: str = 'xyz', degrees: bool = False) -> np.ndarray:
        """提供更灵活的欧拉角获取方式"""
        return self.R.as_euler(seq, degrees=degrees)

    # ==========================================
    # 3. 核心数学运算 (级联与求逆)
    # ==========================================
    def __matmul__(self, other: "Pose") -> "Pose":
        """重载 @ 运算符，实现级联变换 pose_C = pose_A @ pose_B"""
        if not isinstance(other, Pose):
            return NotImplemented
        return Pose(
            t=self.t + self.R.apply(other.t),
            R=self.R * other.R
        )

    def inv(self) -> "Pose":
        """求逆位姿 (求 target 在 current 局部坐标系下的偏差时极常用)"""
        r_inv = self.R.inv()
        return Pose(
            t=-r_inv.apply(self.t),
            R=r_inv
        )

    # ==========================================
    # 4. 逆向工厂方法 (全场景覆盖)
    # ==========================================
    @classmethod
    def from_array(cls, data: np.ndarray | list[float] | tuple[float, ...]) -> "Pose":
        """从 7D 数组 [x, y, z, qx, qy, qz, qw] 构造"""
        arr = np.asarray(data, dtype=float)
        return cls(t=arr[:3], R=Rotation.from_quat(arr[3:]))

    @classmethod
    def from_pos_quat(
        cls, 
        pos: np.ndarray | list[float] | tuple[float, ...], 
        quat: np.ndarray | list[float] | tuple[float, ...]
    ) -> "Pose":
        """从分离的位置和四元数构造"""
        return cls(t=np.asarray(pos, dtype=float), R=Rotation.from_quat(quat))

    @classmethod
    def from_pos_euler(
        cls, 
        pos: np.ndarray | list[float] | tuple[float, ...], 
        euler: np.ndarray | list[float] | tuple[float, ...], 
        seq: str = 'xyz', 
        degrees: bool = False
    ) -> "Pose":
        """从位置和欧拉角构造 (方便对接 UI 或人工输入)"""
        return cls(
            t=np.asarray(pos, dtype=float), 
            R=Rotation.from_euler(seq, euler, degrees=degrees)
        )

    @classmethod
    def from_pos_rot_mat(
        cls, 
        pos: np.ndarray | list[float] | tuple[float, ...], 
        rot_mat: np.ndarray | list[list[float]]
    ) -> "Pose":
        """从位置和 3x3 旋转矩阵构造 (方便对接视觉算法输出)"""
        return cls(t=np.asarray(pos, dtype=float), R=Rotation.from_matrix(rot_mat))

    @classmethod
    def from_homo_mat(cls, mat: np.ndarray | list[list[float]]) -> "Pose":
        """从 4x4 齐次变换矩阵构造"""
        arr = np.asarray(mat, dtype=float)
        return cls(t=arr[:3, 3], R=Rotation.from_matrix(arr[:3, :3]))