import time
import keyboard
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R


class IKSolver:
    """基于阻尼最小二乘的位姿级逆运动学求解器"""

    def __init__(
        self,
        model,
        data,
        site_id,
        arm_dof,
        q_min,
        q_max,
        max_iters=8,
        pos_weight=1.0,
        rot_weight=0.3,
        step_max=0.08,
        pos_tol=1e-4,
        rot_tol=2e-3,
    ):
        self.model = model
        self.data = data
        self.site_id = site_id
        self.arm_dof = arm_dof
        self.q_min = q_min
        self.q_max = q_max
        self.max_iters = max_iters
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.step_max = step_max
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol

        # 预分配雅可比矩阵缓冲区，避免循环内重复创建
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    @staticmethod
    def _rotation_error(R_current, R_target):
        """计算当前姿态与目标姿态的旋转误差向量"""
        R_err = R_target @ R_current.T
        return 0.5 * np.array(
            [
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ]
        )

    @staticmethod
    def _clamp_norm(x, max_norm):
        """限制向量模长（用于限制单步更新量）"""
        n = np.linalg.norm(x)
        if n > max_norm and n > 1e-12:
            return x / n * max_norm
        return x

    @staticmethod
    def _adaptive_damping_from_svd(J, lam_min=1e-4, lam_max=5e-2, sigma_ref=0.05):
        """基于 SVD 最小奇异值的自适应阻尼参数计算"""
        s = np.linalg.svd(J, compute_uv=False)
        sigma_min = s[-1] if len(s) > 0 else 0.0
        ratio = np.clip(sigma_min / sigma_ref, 0.0, 1.0)
        lam = lam_max * (1.0 - ratio) ** 2 + lam_min
        return lam, sigma_min

    def solve(self, q_init, target_pos, target_quat):
        """
        执行 IK 迭代
        target_quat: 目标姿态四元数 [w, x, y, z]
        """
        # 内部统一使用旋转矩阵计算误差
        target_rot = R.from_quat(
            [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
        ).as_matrix()

        q = q_init.copy()
        for _ in range(self.max_iters):
            # 临时更新关节状态并执行前向动力学
            self.data.qpos[: self.arm_dof] = q
            mujoco.mj_forward(self.model, self.data)

            # 获取末端当前位姿
            current_pos = self.data.site_xpos[self.site_id].copy()
            current_rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()

            # 计算位置与姿态误差
            pos_err = target_pos - current_pos
            rot_err = self._rotation_error(current_rot, target_rot)

            # 收敛判断
            if (
                np.linalg.norm(pos_err) < self.pos_tol
                and np.linalg.norm(rot_err) < self.rot_tol
            ):
                break

            # 计算加权雅可比矩阵
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
            J = np.vstack(
                [
                    self.pos_weight * self.jacp[:, : self.arm_dof],
                    self.rot_weight * self.jacr[:, : self.arm_dof],
                ]
            )

            # 构建阻尼最小二乘方程
            lam, _ = self._adaptive_damping_from_svd(J)
            H = J.T @ J + lam * np.eye(self.arm_dof)
            err = np.concatenate([self.pos_weight * pos_err, self.rot_weight * rot_err])
            g = J.T @ err

            # 求解关节增量
            try:
                dq = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(H) @ g

            # 限制步长与关节限位
            dq = self._clamp_norm(dq, self.step_max)
            q = np.clip(q + dq, self.q_min, self.q_max)

        return q


def fmt_arr(arr, precision=3):
    """将 numpy 数组格式化为固定小数位的字符串，如 [0.000, 1.234]"""
    return "[" + ", ".join(f"{x:.{precision}f}" for x in arr) + "]"


# =========================
# 🔧 场景初始化
# =========================
model = mujoco.MjModel.from_xml_path("xml/mjcf/scene.xml")
data = mujoco.MjData(model)
site_id = model.site("ee").id
target_id = model.body("target").id

ARM_DOF = 6
dt = model.opt.timestep
mujoco.mj_forward(model, data)

# 初始化 IK 求解器
ik_solver = IKSolver(
    model=model,
    data=data,
    site_id=site_id,
    arm_dof=ARM_DOF,
    q_min=model.jnt_range[:ARM_DOF, 0].copy(),
    q_max=model.jnt_range[:ARM_DOF, 1].copy(),
    max_iters=8,
    pos_weight=1.0,
    rot_weight=0.1115,
    step_max=0.08,
)

#  Q: [-0.052,-1.424,-0.21 , 0.056,-0.399, 3.022] | P: [0.008,0.137,0.013] | Quat: [ 0.006,-0.005,-0.022, 1.   ]  ]

# [x, y, z] [w, x, y, z]

target_pos = np.array([0.008, 0.072, 0.086])
target_quat = np.array([1.000, 0.006, -0.005, -0.022])

# target_pos = model.body_pos[target_id].copy()
# target_quat = np.roll(R.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_quat(), 1)

q_sol = data.qpos[:ARM_DOF].copy()

target_speed = 0.15
rot_speed = 1.0  # rad/s
last_print_time = 0.0


# =========================
# 🚀 仿真主循环
# =========================
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.perf_counter()

        # --- 键盘位置控制 ---
        move_dir = np.zeros(3)
        speed_scale = 4.0 if keyboard.is_pressed("space") else 1.0

        if keyboard.is_pressed("up"):
            move_dir[1] += 1.0
        if keyboard.is_pressed("down"):
            move_dir[1] -= 1.0
        if keyboard.is_pressed("left"):
            move_dir[0] -= 1.0
        if keyboard.is_pressed("right"):
            move_dir[0] += 1.0
        if keyboard.is_pressed("alt"):
            move_dir[2] += 1.0
        if keyboard.is_pressed("ctrl"):
            move_dir[2] -= 1.0

        if np.linalg.norm(move_dir) > 1e-12:
            target_pos += (
                (move_dir / np.linalg.norm(move_dir)) * target_speed * speed_scale * dt
            )

        # --- 键盘姿态控制 (世界坐标系增量) ---
        rot_vec = np.zeros(3)
        if keyboard.is_pressed("-"):
            rot_vec[0] -= 1.0
        if keyboard.is_pressed("="):
            rot_vec[0] += 1.0
        if keyboard.is_pressed("["):
            rot_vec[1] -= 1.0
        if keyboard.is_pressed("]"):
            rot_vec[1] += 1.0
        if keyboard.is_pressed(";"):
            rot_vec[2] -= 1.0
        if keyboard.is_pressed("'"):
            rot_vec[2] += 1.0

        if np.linalg.norm(rot_vec) > 1e-12:
            # 计算轴角增量
            delta_ang = rot_vec * rot_speed * speed_scale * dt
            delta_R = R.from_rotvec(delta_ang)
            # 转换为 scipy 格式 [x,y,z,w] 的四元数对象
            current_R = R.from_quat(
                [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
            )
            # 左乘：世界坐标系下叠加旋转
            new_R = delta_R * current_R
            # 更新 target_quat [w, x, y, z] 并归一化防漂移
            target_quat = np.roll(new_R.as_quat(), 1)
            target_quat /= np.linalg.norm(target_quat)

        # --- 重置位姿 ---
        if keyboard.is_pressed("r"):
            mujoco.mj_forward(model, data)
            target_quat = np.roll(
                R.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_quat(), 1
            )
            target_pos = data.site_xpos[site_id].copy()

        # 更新可视化目标
        model.body_pos[target_id] = target_pos

        # --- 执行 IK ---
        q_sol = ik_solver.solve(q_sol, target_pos, target_quat)

        # --- 驱动与仿真 ---
        data.ctrl[:ARM_DOF] = q_sol
        mujoco.mj_step(model, data)
        viewer.sync()

        # --- 打印状态与误差 ---
        now = time.perf_counter()
        if now - last_print_time > 0.1:
            current_rot = data.site_xmat[site_id].reshape(3, 3)
            target_rot = R.from_quat(
                [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
            ).as_matrix()
            r_err = np.linalg.norm(IKSolver._rotation_error(current_rot, target_rot))
            p_err = np.linalg.norm(target_pos - data.site_xpos[site_id])

            # 格式化数组字符串，suppress_small 隐藏极小值，precision 控制小数位
            q_str = fmt_arr(q_sol)
            p_str = fmt_arr(target_pos)
            quat_str = fmt_arr(target_quat)

            print(
                f"\rPos Err: {p_err:.4f} | Rot Err: {r_err:.4f} | Q: {q_str} | P: {p_str} | Quat: {quat_str} {8 * ' '}",
                end="",
            )
            last_print_time = now

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, dt - elapsed))
