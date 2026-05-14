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
        
        # 异步控制相关
        self._q_target = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # 用于 IK 计算的独立数据副本（避免与仿真线程竞争）
        self._ik_model = mujoco.MjModel.from_xml_path(model_path)
        self._ik_data = mujoco.MjData(self._ik_model)
        self._ik_jacp = np.zeros((3, self._ik_model.nv))
        self._ik_jacr = np.zeros((3, self._ik_model.nv))

        self.q_target = np.zeros(self.arm_dof)

        mujoco.mj_forward(self.model, self.data)  # 获取初始状态

    def get_q_current(self):
        """获取当前所有关节角"""
        with self._lock:
            return self.data.qpos[:self.arm_dof].copy()

    def get_fk_mat(self, q: np.ndarray):
        """获取FK结果, 返回位置和旋转矩阵（使用独立副本，无锁）"""
        self._ik_data.qpos[:self.arm_dof] = q
        mujoco.mj_forward(self._ik_model, self._ik_data)
        pos = self._ik_data.site_xpos[self.site_id].copy()
        rot = self._ik_data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, rot

    def get_fk_quat(self, q: np.ndarray):
        """获取FK结果, 返回位置和四元数 [x, y, z, w]（使用独立副本，无锁）"""
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
        """获取Jacobian（使用独立副本，无锁）"""
        self._ik_data.qpos[:self.arm_dof] = q
        mujoco.mj_forward(self._ik_model, self._ik_data)
        mujoco.mj_jacSite(self._ik_model, self._ik_data, self._ik_jacp, self._ik_jacr, self.site_id)
        return self._ik_jacp.copy(), self._ik_jacr.copy()

    def update_target_dot(self, target_pos):
        """更新目标绿点的可视化位置"""
        with self._lock:
            target_id = self.model.body("target").id
            self.model.body_pos[target_id] = target_pos.copy()

    def set_q_target(self, q_target: np.ndarray):
        """设置目标关节角(非阻塞)"""
        with self._lock:
            self._q_target = q_target.copy()

    def _simulation_loop(self):
        """内部仿真循环"""
        while self._running:
            with self._lock:
                if self._q_target is not None:
                    self.data.ctrl[:self.arm_dof] = self._q_target
            
            mujoco.mj_step(self.model, self.data)
            
            if self.viewer is not None:
                with self._lock:
                    self.viewer.sync()
            
            # time.sleep(self.dt)

    def start(self):
        """启动仿真器(后台运行)"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止仿真器"""
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def step(self, q_target: np.ndarray):
        """ [已废弃] 使用 set_q_target + start 替代 """
        self.set_q_target(q_target)

    def launch(self):
        """启动可视化界面"""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer

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