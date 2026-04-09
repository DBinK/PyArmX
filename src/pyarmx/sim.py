import time

import keyboard
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R


class ArmSimulator:
    def __init__(self, model_path: str, arm_dof: int = 6, site_name: str = "ee"):
        """ 创建一个仿真器 """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.arm_dof = arm_dof
        self.dt = self.model.opt.timestep

        self.site_id = self.model.site(site_name).id

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

        self.viewer: mujoco.viewer.Handle | None = None

        mujoco.mj_forward(self.model, self.data)  # 获取初始状态

    def get_q_current(self):
        """获取当前所有关节角"""
        return self.data.qpos[:self.arm_dof].copy()

    def get_fk_mat(self, q: np.ndarray):
        """获取FK结果, 返回位置和旋转矩阵"""
        self.data.qpos[:self.arm_dof] = q
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, rot

    def get_fk_quat(self, q: np.ndarray):
        """获取FK结果, 返回位置和四元数"""
        pos, rot = self.get_fk_mat(q)
        quat = R.from_matrix(rot).as_quat()
        return pos, quat

    def get_jacobian(self, q: np.ndarray):
        """获取Jacobian"""
        self.data.qpos[:self.arm_dof] = q
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
        return self.jacp, self.jacr

    def update_target_dot(self, target_pos):
        """更新目标绿点的可视化位置"""
        target_id = self.model.body("target").id
        self.model.body_pos[target_id] = target_pos

    def step(self, q_target: np.ndarray):
        """ 更新仿真器 """
        self.data.ctrl[:self.arm_dof] = q_target
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def launch(self):
        """启动可视化界面"""
        return mujoco.viewer.launch_passive(self.model, self.data)


# =========================
# 3. KeyboardController ⭐
# =========================

class KeyboardController:
    def __init__(self, target_speed=0.15, rot_speed=1.0):
        self.target_speed = target_speed
        self.rot_speed = rot_speed

    def update(self, target_pos, target_quat, dt):
        """键盘输入 -> 新的 target"""

        move_dir = np.zeros(3)
        speed_scale = 4.0 if keyboard.is_pressed("space") else 1.0

        # --- 平移 ---
        if keyboard.is_pressed("up"): move_dir[1] += 1.0
        if keyboard.is_pressed("down"): move_dir[1] -= 1.0
        if keyboard.is_pressed("left"): move_dir[0] -= 1.0
        if keyboard.is_pressed("right"): move_dir[0] += 1.0
        if keyboard.is_pressed("alt"): move_dir[2] += 1.0
        if keyboard.is_pressed("ctrl"): move_dir[2] -= 1.0

        if np.linalg.norm(move_dir) > 1e-12:
            target_pos = target_pos + (move_dir / np.linalg.norm(move_dir)) * self.target_speed * speed_scale * dt

        # --- 旋转 ---
        rot_vec = np.zeros(3)
        if keyboard.is_pressed("-"): rot_vec[0] -= 1.0
        if keyboard.is_pressed("="): rot_vec[0] += 1.0
        if keyboard.is_pressed("["): rot_vec[1] -= 1.0
        if keyboard.is_pressed("]"): rot_vec[1] += 1.0
        if keyboard.is_pressed(";"): rot_vec[2] -= 1.0
        if keyboard.is_pressed("'"): rot_vec[2] += 1.0

        if np.linalg.norm(rot_vec) > 1e-12:
            delta_ang = rot_vec * self.rot_speed * speed_scale * dt
            delta_R = R.from_rotvec(delta_ang)
            current_R = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
            new_R = delta_R * current_R
            target_quat = np.roll(new_R.as_quat(), 1)
            target_quat /= np.linalg.norm(target_quat)

        return target_pos, target_quat
