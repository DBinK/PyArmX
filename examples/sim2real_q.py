
import time

import numpy as np

from pyarmx.ik import IKSolver
from pyarmx.sim import ArmSimulator
from pyarmx.input import JointInput, PoseInput

from scipy.spatial.transform import Rotation as R

from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate, Timer

from pydamiao.bus import SerialBus
from pydamiao.arm.config import joint_cfgs
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

MODEL_PATH = "xml/L20/scene.xml"
ARM_DOF = 6 

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

# 初始状态
q_current = sim.get_q_current()

# 启动仿真
# sim.start()
sim.viewer = sim.launch()

# 初始化关节输入控制器
joint_input = JointInput(joint_speed=1.0)


# 主循环
loop = Rate(hz=100)
timer = Timer(duration=0.1)

while sim.viewer.is_running() and loop.sleep(): # type: ignore

    current_q = sim.get_q_current()
    
    # 根据键盘输入更新目标关节角
    q_target = joint_input.update(current_q, sim.dt)
    
    # 步进仿真
    sim.step(q_target)

    manager.set_pos_list(q_target.tolist(), ControlMode.POS_VEL)

    # 更新当前状态, 此处仿真直接用 q_command , 真机可以考虑用真实的 q_current
    q_current = q_target 

    # 监控
    if timer.done:
        print(f"\rq_target: {fmt_arr(q_target)}", end="")
        timer.reset()