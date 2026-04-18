
from math import e
import time

import numpy as np

from pyarmx.ik import IKSolver
from pyarmx.sim import ArmSimulator
from pyarmx.input import PoseInput, PlaybackInput

from scipy.spatial.transform import Rotation as R

from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate, Timer

from pydamiao.bus import SerialBus
from pydamiao.arm.config import joint_cfgs, JointID
from pydamiao.arm.joint import JointManager
from pydamiao.structs import ControlMode

# --- 真实机械臂 --- #
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

# MODEL_PATH = "xml/L20/scene.xml"
MODEL_PATH = "xml/L801/scene.xml"
ARM_DOF = 6 

sim = ArmSimulator(MODEL_PATH, arm_dof=ARM_DOF)

controller = PoseInput()
playback = PlaybackInput()

ik_solver = IKSolver(
    fk_func=sim.get_fk_mat,
    jac_func=sim.get_jacobian,
    arm_dof=ARM_DOF,
    q_min=sim.model.jnt_range[:ARM_DOF, 0].copy(),
    q_max=sim.model.jnt_range[:ARM_DOF, 1].copy(),
    rot_weight=0.1115,
)

# 初始状态
q_current = sim.get_q_current()
target_pos, target_quat = sim.get_fk_quat(q_current)

target_pos = np.array([0.008, 0.072, 0.086])
target_quat = np.array([0.006, -0.005, -0.022, 1.000])  # [x, y, z, w] 格式

# 矩形参数
dx = 0.00
dy = 0.08
ret_pos = np.array([
    [0.008, 0.072, 0.086],
    [0.008 + dx, 0.072, 0.086],
    [0.008 + dx, 0.072 + dy, 0.086],
    [0.008, 0.072 + dy, 0.086],
])
pos_i = 0  # 矩形位置索引

# 启动仿真
# sim.start()
sim.viewer = sim.launch()

# 主循环
loop = Rate(hz=1000)
timer = Timer(duration=0.1)

switch_timer = Timer(duration=2.0)

rec_timer = Timer(duration=20.0)
torque_list = []

while sim.viewer.is_running() and loop.sleep(): # type: ignore

    # 输入层
    # target_pos, target_quat = controller.update(
    #     target_pos, target_quat, sim.dt
    # )

    if switch_timer.done:
        switch_timer.reset()
    # if pause := playback.is_switch(): 
        pos_i += 1 
        target_pos = ret_pos[pos_i % 4]
        print(f"切换到 {pos_i % 4} 号点")

    # 目标点可视化
    sim.update_target_dot(target_pos)

    # IK + 控制
    q_command = ik_solver.solve(q_current, target_pos, target_quat)
    if q_command is None or np.any(np.isnan(q_command)):  # 检测是否异常
        q_command = q_current
    
    # 设置关节
    if pause := playback.is_pause():  # 暂停
        sim.step(q_command)
        manager.set_pos_list(q_command.tolist(), ControlMode.POS_VEL)
    else:
        sim.viewer.sync()
        rec_timer.reset()

    # 更新当前状态, 此处仿真直接用 q_command , 真机可以考虑用真实的 q_current
    q_current = sim.get_q_current() 

        

    # 监控
    # if timer.done:
    #     current_rot = sim.data.site_xmat[sim.site_id].reshape(3, 3)
    #     target_rot = R.from_quat(target_quat).as_matrix()

    #     r_err = np.linalg.norm(IKSolver._rotation_error(current_rot, target_rot))
    #     p_err = np.linalg.norm(target_pos - sim.data.site_xpos[sim.site_id])

    #     q_str = fmt_arr(q_current)
    #     p_str = fmt_arr(target_pos)
    #     quat_str = fmt_arr(target_quat)

    #     print(
    #         f"\rPos Err: {p_err:.4f} | Rot Err: {r_err:.4f} | Q: {q_str} | P: {p_str} | Quat: {quat_str} {8 * ' '}",
    #         end="",
    #     )
    #     # print(f"\rPos Err: {p_err:.4f} | Rot Err: {r_err:.4f}", end="")
        
    #     timer.reset()


    # 记录力矩变化
    elbow_torque = manager.get_joints_torque()[JointID.elbow]
    print(f"{elbow_torque}")

    if rec_timer.done:
        import csv 
        with open("tmp/torque_ik.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for torque_data in torque_list:
                writer.writerow([torque_data])
        break
    else:
        torque_list.append(elbow_torque)
        