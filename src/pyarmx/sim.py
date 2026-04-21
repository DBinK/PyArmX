import time
import threading

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
        # self.dt = 20.2001

        self.site_id = self.model.site(site_name).id

        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

        self.viewer: mujoco.viewer.Handle | None = None

        self.q_target = np.zeros(self.arm_dof)

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
        """获取FK结果, 返回位置和四元数 [x, y, z, w]"""
        pos, rot = self.get_fk_mat(q)
        quat = R.from_matrix(rot).as_quat()
        return pos, quat

    def get_ee_pose(self):
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
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

    # 异步方法启动仿真
    def set_q_target(self, q_target: np.ndarray):
        """设置目标关节角"""
        self.q_target = q_target.copy()
        # print("q_target:", q_target)
    
    def start(self):
        self.start_thread = threading.Thread(target=self._loop, daemon=True)
        self.start_thread.start()
        time.sleep(1.0)

    def _loop(self):
        self.viewer = self.launch()
        while self.viewer.is_running():
            self.step(self.q_target)
            # time.sleep(self.dt)


if __name__ == "__main__":
    MODEL_PATH = "xml/mjcf/scene.xml"

    sim = ArmSimulator(MODEL_PATH)

    sim.viewer = sim.launch()  # 启动仿真 
    
    q_command = np.asanyarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    while True:
        # sim.step(sim.get_q_current())
        sim.set_q_target(q_command)
        time.sleep(0.01)