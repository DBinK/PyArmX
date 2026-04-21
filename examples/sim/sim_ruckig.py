import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from pyarmx.interp import RuckigPosePlanner 
from pyarmx.ik import IKSolver
from pyarmx.sim import ArmSimulator
from pyarmx.input.keyboard import PoseInput

from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate, Timer

import rerun as rr

# 初始化 Rerun 会话，指定应用名称
rr.init("dt")
# rr.save("tmp/data.rrd")  # 保存到文件
rr.spawn()  # 启动本地可视化查看器

def rrlog_dt(dt):
    rr.set_time("time", timestamp=time.time())
    rr.log("dt", rr.Scalars(float(dt)))


# MODEL_PATH = "xml/L20/scene.xml"
MODEL_PATH = "xml/L801/scene.xml"
ARM_DOF = 6

# 初始化仿真与控制
sim = ArmSimulator(MODEL_PATH, arm_dof=ARM_DOF)
controller = PoseInput()

ik_solver = IKSolver(
    fk_func=sim.get_fk_mat,
    jac_func=sim.get_jacobian,
    arm_dof=ARM_DOF,
    q_min=sim.model.jnt_range[:ARM_DOF, 0].copy(),
    q_max=sim.model.jnt_range[:ARM_DOF, 1].copy(),
    rot_weight=0.1115,
)

# 初始化 Ruckig 规划器
planner = RuckigPosePlanner(control_period=sim.dt, buffer_size=100) 

# 初始位姿
q_current = sim.get_q_current()
init_pos = np.array([0.008, 0.072, 0.086])
init_quat = np.array([0.006, -0.005, -0.022, 1.000])
init_pose_7d = np.concatenate([init_pos, init_quat])

planner.set_init_pose(init_pose_7d)
planner.start()

sim.viewer = sim.launch()

# 主循环
final_target_pos = init_pos.copy()
final_target_quat = init_quat.copy()

print("[Sim] 系统就绪。使用键盘移动红色目标点，机械臂将平滑追踪。")

loop = Rate(hz=100)
timer = Timer(duration=0.1)

while sim.viewer.is_running() and loop.sleep():

    # 更新最终目标
    new_target_pos, new_target_quat = controller.update(
        final_target_pos, final_target_quat, sim.dt
    )
    
    pos_diff = np.linalg.norm(new_target_pos - final_target_pos)
    quat_diff = 1.0 - np.abs(np.dot(new_target_quat, final_target_quat))
    
    if pos_diff > 1e-4 or quat_diff > 1e-4:
        final_target_pos = new_target_pos
        final_target_quat = new_target_quat
        
        target_7d = np.concatenate([final_target_pos, final_target_quat])
        planner.set_target(target_7d)
        
    sim.update_target_dot(final_target_pos)

    # 获取平滑轨迹点
    smooth_pose = planner.get_pose(block=False, timeout=0)
    
    if smooth_pose is None:
        exec_pos, exec_quat = sim.get_fk_quat(q_current)
    else:
        exec_pos = smooth_pose[:3]
        exec_quat = smooth_pose[3:]

    # IK 求解并执行
    q_command = ik_solver.solve(q_current, exec_pos, exec_quat)
    
    if q_command is None or np.any(np.isnan(q_command)):
        q_command = q_current
        
    sim.step(q_command)
    q_current = q_command 

    # print(f"{loop.tick.delta:.6f} {loop.tick.on_time} ")

    rrlog_dt(loop.tick.delta)

    # 监控日志
    if timer.done:
        current_actual_pos, current_actual_quat = sim.get_fk_quat(q_current)
        
        p_err = np.linalg.norm(exec_pos - current_actual_pos)
        
        current_rot = R.from_quat(current_actual_quat).as_matrix()
        target_rot = R.from_quat(exec_quat).as_matrix()
        rot_diff = current_rot @ target_rot.T
        r_err = np.arccos((np.trace(rot_diff) - 1) / 2)

        print(
            f"\rTrack Err P:{p_err:.4f} R:{r_err:.4f} | Target P:{fmt_arr(final_target_pos)}",
            end="",
        )
        timer.reset()