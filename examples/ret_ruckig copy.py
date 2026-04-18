
import time

import numpy as np

from pyarmx.ik import IKSolver
from pyarmx.interp import RuckigPosePlanner
from pyarmx.sim import ArmSimulator
from pyarmx.input import PoseInput, PlaybackInput

from scipy.spatial.transform import Rotation as R

from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate, Timer

from pydamiao.bus import SerialBus
from pydamiao.arm.config import JointID, joint_cfgs
from pydamiao.arm.joint import JointManager
from pydamiao.structs import ControlMode

# 真实机械臂
bus = SerialBus("COM9", baudrate=921600, timeout=0.01)
manager = JointManager(bus)

# 注册joint
manager.register(joint_cfgs)

# 设置初始状态
manager.clean_error()
manager.enable()
# manager.set_teach_mode()
# manager.set_mode(ControlMode.POS_FORCE)
manager.set_mode(ControlMode.POS_VEL)


# 初始化仿真与控制
# MODEL_PATH = "xml/L20/scene.xml"
MODEL_PATH = "xml/L801/scene.xml"
ARM_DOF = 6

sim = ArmSimulator(MODEL_PATH, arm_dof=ARM_DOF)

pose_input = PoseInput()
playback = PlaybackInput()

ik_solver = IKSolver(
    fk_func=sim.get_fk_mat,
    jac_func=sim.get_jacobian,
    arm_dof=ARM_DOF,
    q_min=sim.model.jnt_range[:ARM_DOF, 0].copy(),
    q_max=sim.model.jnt_range[:ARM_DOF, 1].copy(),
    rot_weight=0.15,
)

# 初始化 Ruckig 规划器
planner = RuckigPosePlanner(buffer_size=100) 

# 初始位姿
q_current = np.asanyarray(manager.get_joints_pos_list())
init_pos = np.array([0.008, 0.072, 0.086])
init_quat = np.array([0.006, -0.005, -0.022, 1.000])
init_pose_7d = np.concatenate([init_pos, init_quat])

planner.set_init_pose(init_pose_7d)
planner.start()

sim.viewer = sim.launch()


# 矩形参数
z = 0.0463
ret_pos = np.array([
    [0.008, 0.072, 0.086],
    [0.102, 0.120, z],
    [0.102, 0.120, z],
    [0.102, 0.120, z],
    [0.102, 0.120, z],
])
pos_i = 0  # 矩形位置索引


# 主循环
final_target_pos = init_pos.copy()
final_target_quat = init_quat.copy()

print("[Sim] 系统就绪。使用键盘移动红色目标点，机械臂将平滑追踪。")

loop = Rate(hz=1000)
timer = Timer(duration=0.1)

switch_timer = Timer(duration=2.0)

rec_timer = Timer(duration=20.0)
rec_data = []

while sim.viewer.is_running() and loop.sleep():

    # 更新最终目标
    # new_target_pos, new_target_quat = pose_input.update(
    #     final_target_pos, final_target_quat, sim.dt
    # )
    new_target_pos = final_target_pos
    new_target_quat = final_target_quat

    if switch_timer.done:
        switch_timer.reset()
    # if pause := playback.is_switch(): 
        pos_i += 1
        new_target_pos = ret_pos[pos_i % len(ret_pos)]
        print(f"切换到 {pos_i % len(ret_pos)} 号点")
    
    pos_diff = np.linalg.norm(new_target_pos - final_target_pos)
    quat_diff = 1.0 - np.abs(np.dot(new_target_quat, final_target_quat))
    
    if pos_diff > 1e-4 or quat_diff < 0.9999:
        final_target_pos = new_target_pos
        final_target_quat = new_target_quat
        
        target_7d = np.concatenate([final_target_pos, final_target_quat])
        planner.set_target(target_7d)
        
    sim.update_target_dot(new_target_pos)

    # 获取平滑轨迹点
    smooth_pose = planner.get_pose(block=False, timeout=0)
    
    if smooth_pose is None:
        exec_pos, exec_quat = sim.get_fk_quat(q_current)
    else:
        exec_pos  = smooth_pose[:3]
        exec_quat = smooth_pose[3:]

    # IK 求解并执行
    q_command = ik_solver.solve(q_current, exec_pos, exec_quat)
    if q_command is None or np.any(np.isnan(q_command)):  # 检测是否异常
        q_command = q_current
    
    # 设置关节
    if pause := playback.is_pause():  # 暂停
        sim.step(q_command)
        manager.set_pos_list(q_command.tolist(), ControlMode.POS_VEL)
    else:
        sim.viewer.sync()
        rec_timer.reset()

    # 更新当前状态, 此处直接用 q_command , 真机可以考虑用真实的 q_current
    q_current = sim.get_q_current() 

    # # 监控日志
    # if timer.done:
    #     current_actual_pos, current_actual_quat = sim.get_fk_quat(q_current)
        
    #     p_err = np.linalg.norm(exec_pos - current_actual_pos)
        
    #     current_rot = R.from_quat(current_actual_quat).as_matrix()
    #     target_rot = R.from_quat(exec_quat).as_matrix()
    #     r_err = np.linalg.norm(IKSolver._rotation_error(current_rot, target_rot))

    #     print(
    #         f"\rTrack Err P:{p_err:.4f} R:{r_err:.4f} | Target P:{fmt_arr(final_target_pos)}",
    #         end="",
    #     )
    #     timer.reset()

    
    # 记录电机状态变化
    # if rec_timer.done:
    #     import csv 
    #     with open("tmp/torque_ig.csv", "w", newline="") as f:
    #         writer = csv.writer(f)
    #         for motor_state in rec_data:
    #             writer.writerow(motor_state)
    #     break
    # else:
    #     elbow = manager.get_joint_by_id(JointID.elbow)
    #     pos, vel, torque = elbow.motor.get_state()

    #     rec_data.append([pos, vel, torque])
        