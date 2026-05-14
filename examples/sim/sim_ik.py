
import time

import numpy as np

from pyarmx.ik import IKSolver
from pyarmx.sim import ArmSimulator
from pyarmx.input.keyboard import PoseInput

from scipy.spatial.transform import Rotation as R

from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate, Timer


# MODEL_PATH = "xml/L20/scene.xml"
MODEL_PATH = "xml/L801/scene.xml"
ARM_DOF = 6 

sim = ArmSimulator(MODEL_PATH, arm_dof=ARM_DOF)
controller = PoseInput()
ik_solver = IKSolver(
    fk_func=sim.get_fk_mat,
    jac_func=sim.get_jacobian,
    arm_dof=ARM_DOF,
    q_min=sim.model.jnt_range[:ARM_DOF, 0].copy(),
    q_max=sim.model.jnt_range[:ARM_DOF, 1].copy(),
    rot_weight=0.15,
)

# 初始状态
q_current = sim.get_q_current()
target_pos, target_quat = sim.get_fk_quat(q_current)

target_pos = np.array([0.01, 0.070, 0.080])
target_quat = np.array([0.006, -0.005, -0.022, 1.000])   # 朝下   # [x, y, z, w] 格式

# 启动仿真
sim.viewer = sim.launch()
# sim.start()

# 主循环
loop = Rate(hz=100)
timer = Timer(duration=0.1)

while sim.viewer.is_running() and loop.sleep(): # type: ignore

    # 输入层
    target_pos, target_quat = controller.update(
        target_pos, target_quat, sim.dt
    )

    # 目标点可视化
    sim.update_target_dot(target_pos)

    # TODO: 待修复 IK
    # IK + 控制 
    q_command = ik_solver.solve(q_current, target_pos, target_quat)
    sim.step(q_command)
    # sim.set_q_target(q_command)

    # 更新当前状态, 此处仿真直接用 q_command , 真机可以考虑用真实的 q_current
    q_current = q_command 

    # 监控
    if timer.done:
        current_rot = sim.data.site_xmat[sim.site_id].reshape(3, 3)
        target_rot = R.from_quat(target_quat).as_matrix()

    #     r_err = np.linalg.norm(IKSolver._rotation_error(current_rot, target_rot))
    #     p_err = np.linalg.norm(target_pos - sim.data.site_xpos[sim.site_id])

    #     q_str = fmt_arr(q_current)
    #     p_str = fmt_arr(target_pos)
    #     quat_str = fmt_arr(target_quat)

        print(
            f"\rPos Err: {p_err:.4f} | Rot Err: {r_err:.4f} | Q: {q_str} | P: {p_str} | Quat: {quat_str} {8 * ' '}",
            end="",
        )
        # print(f"\rPos Err: {p_err:.4f} | Rot Err: {r_err:.4f}", end="")
        
        timer.reset()
