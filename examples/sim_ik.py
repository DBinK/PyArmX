
import time

import numpy as np

from pyarmx.ik import IKSolver
from pyarmx.sim import ArmSimulator, KeyboardController

from scipy.spatial.transform import Rotation as R

from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate, Timer



MODEL_PATH = "xml/mjcf/scene.xml"
ARM_DOF = 6

sim = ArmSimulator(MODEL_PATH, arm_dof=ARM_DOF)
controller = KeyboardController()
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

# 启动仿真
sim.launch()
sim.start()

# 主循环
# --- 主循环 --- #
loop = Rate(hz=100)
timer = Timer(duration=0.5)

while sim.viewer.is_running() and loop.sleep(): # type: ignore

    # 输入层
    target_pos, target_quat = controller.update(
        target_pos, target_quat, sim.dt
    )

    # 目标点可视化
    sim.update_target_dot(target_pos)

    # TODO: 待修复 IK
    # IK + 控制 
    # q_command = ik_solver.solve(q_current, target_pos, target_quat)
    q_command = [0.0] * 6

    sim.set_q_target(np.asanyarray(q_command))

    # 更新当前状态, 此处仿真直接用 q_command , 真机可以考虑用真实的 q_current
    q_current = q_command 

    print(f"q: {fmt_arr(q_command)}, pos: {fmt_arr(target_pos)}, {loop.tick.delta:.6f}, {loop.tick.on_time}")
    # 监控
    # if timer.done:  # 限频打印
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
