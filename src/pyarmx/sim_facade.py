import sys
from pathlib import Path

import numpy as np
from loguru import logger
from rich import print as rprint
from scipy.spatial.transform import Rotation as R

from pyarmx.ik import IKSolver
from pyarmx.input.keyboard import PlaybackInput
from pyarmx.input.zmq_sub import PoseReceiver
from pyarmx.interp import RuckigPosePlanner
from pyarmx.sim import ArmSimulator
from pyarmx.structs import Pose

from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate, Timer


### 高层仿真包装器
class ArmSimFacade:
    def __init__(self, model_path: str | Path, arm_dof: int = 6):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            logger.error(f"模型文件不存在: {self.model_path}")
            sys.exit(1)
        
        # 初始化基础仿真
        self.sim = ArmSimulator(str(self.model_path), arm_dof)
        
        # 初始化IK求解器
        self.ik_solver = IKSolver(
            fk_func=self.sim.get_fk_mat,
            jac_func=self.sim.get_jacobian,
            arm_dof=arm_dof,
            q_min=self.sim.model.jnt_range[:arm_dof, 0].copy(),
            q_max=self.sim.model.jnt_range[:arm_dof, 1].copy(),
            rot_weight=0.15,
        )
        
        # 初始化轨迹规划器
        self.planner = RuckigPosePlanner(buffer_size=10)
        
        # 初始化状态与位姿
        self.q_current = self.sim.get_q_current()
        init_pos, init_quat = self.sim.get_fk_quat(self.q_current)
        init_pose_7d = np.concatenate([init_pos, init_quat])
        
        # 启动轨迹插值
        self.planner.set_init_pose(init_pose_7d)
        self.planner.start()
        
        # 启动可视化
        self.sim.viewer = self.sim.launch()
        logger.info("高层仿真类初始化完成")

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """获取当前末端真实位置和四元数"""
        return self.sim.get_fk_quat(self.q_current)

    def set_ee_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """设置末端执行器目标位姿"""
        target_7d = np.concatenate([pos, quat])
        self.planner.set_target(target_7d)
        self.sim.update_target_dot(pos)

    def step(self, pause: bool = False) -> bool:
        """执行单步仿真，返回可视化窗口是否存活"""
        if not self.sim.viewer.is_running(): # type: ignore
            return False

        if pause:
            self.sim.viewer.sync() # type: ignore
            return True

        # 获取平滑插值
        smooth_pose = self.planner.get_pose(block=False, timeout=0)
        if smooth_pose is None:
            exec_pos, exec_quat = self.sim.get_fk_quat(self.q_current)
        else:
            exec_pos, exec_quat = smooth_pose[:3], smooth_pose[3:]

        # 求解IK并执行
        q_command = self.ik_solver.solve(self.q_current, exec_pos, exec_quat)
        if q_command is None or np.any(np.isnan(q_command)):
            q_command = self.q_current

        self.sim.step(q_command)
        self.q_current = self.sim.get_q_current()
        
        return True

    def get_q_current(self) -> np.ndarray:
        return self.sim.get_q_current()


if __name__ == "__main__":

    model_path: str | Path = "xml/L801/scene.xml"
    arm = ArmSimFacade(model_path)
    
    pose_input = PoseReceiver()
    playback = PlaybackInput()
    
    # 初始化目标位姿
    init_pos = np.array([0.008, 0.072, 0.086])
    init_quat = np.array([0.006, -0.005, -0.022, 1.000])
    target_7d = Pose.from_pos_quat(init_pos, init_quat)
    
    loop = Rate(hz=100)
    timer = Timer(duration=0.1)
    
    logger.info("系统就绪。使用键盘移动红色目标点，机械臂将平滑追踪。")

    try:
        # arm.step() 返回 False 说明窗口被关闭
        while arm.sim.viewer.is_running() and loop.sleep(): # type: ignore
            
            # 更新输入与目标
            pose_input.update(target_7d)
            arm.set_ee_pose(target_7d.pos, target_7d.quat)
            arm.step() 

            # # 监控日志
            # if timer.done:
            #     pos, quat = arm.get_ee_pose()

            #     print(f"\rTrack Err P:{p_err:.4f} R:{r_err:.4f} | Target P:{fmt_arr(target_7d.array)}", end="")
            #     timer.reset()

    except KeyboardInterrupt:
        logger.info("\n收到中断信号，准备退出...")
