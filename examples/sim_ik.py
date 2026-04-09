
import time

import numpy as np

from pyarmx.ik import IKSolver
from pyarmx.sim import ArmSimulator, KeyboardController

from scipy.spatial.transform import Rotation as R

def fmt_arr(arr, precision=3):
    """将 numpy 数组格式化为固定小数位的字符串，如 [0.000, 1.234]"""
    return "[" + ", ".join(f"{x:.{precision}f}" for x in arr) + "]"


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
target_quat = np.array([1.000, 0.006, -0.005, -0.022])

# 启动仿真
sim.viewer = sim.launch()

# 主循环
last_print_time = 0.0
while sim.viewer.is_running():

    t_start = time.perf_counter()  # 获取循环开始时间戳, 稍后用于帧率控制

    # 输入层
    target_pos, target_quat = controller.update(
        target_pos, target_quat, sim.dt
    )

    # 目标点可视化
    sim.update_target_dot(target_pos)

    # IK + 控制
    q_command = ik_solver.solve(q_current, target_pos, target_quat)
    sim.step(q_command)

    # 更新当前状态, 此处仿真直接用 q_command , 真机可以考虑用真实的 q_current
    q_current = q_command 

    # 监控
    now = time.perf_counter()
    if now - last_print_time > 0.1:
        current_rot = sim.data.site_xmat[sim.site_id].reshape(3, 3)
        target_rot = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()

        r_err = np.linalg.norm(IKSolver._rotation_error(current_rot, target_rot))
        p_err = np.linalg.norm(target_pos - sim.data.site_xpos[sim.site_id])

        q_str = fmt_arr(q_current)
        p_str = fmt_arr(target_pos)
        quat_str = fmt_arr(target_quat)

        print(
            f"\rPos Err: {p_err:.4f} | Rot Err: {r_err:.4f} | Q: {q_str} | P: {p_str} | Quat: {quat_str} {8 * ' '}",
            end="",
        )
        # print(f"\rPos Err: {p_err:.4f} | Rot Err: {r_err:.4f}", end="")
        last_print_time = now

    # 帧率控制
    sleep_time = max(0.0, sim.dt - (time.perf_counter() - t_start))
    if sleep_time > 0:
        time.sleep(sleep_time)

