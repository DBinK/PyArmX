
import time

import numpy as np

from pyarmx.ik import IKSolver
from pyarmx.sim import ArmSimulator, KeyboardController

from scipy.spatial.transform import Rotation as R

MODEL_PATH = "xml/mjcf/scene.xml"
ARM_DOF = 6

sim = ArmSimulator(MODEL_PATH, arm_dof=ARM_DOF)
controller = KeyboardController()

ik_solver = IKSolver(
    fk_func=sim.get_fk,
    jac_func=sim.get_jacobian,
    arm_dof=ARM_DOF,
    q_min=sim.model.jnt_range[:ARM_DOF, 0].copy(),
    q_max=sim.model.jnt_range[:ARM_DOF, 1].copy(),
    rot_weight=0.1115,
)

target_id = sim.model.body("target").id

q_current = sim.data.qpos[:ARM_DOF].copy()
target_pos = np.array([0.008, 0.072, 0.086])
target_quat = np.array([1.000, 0.006, -0.005, -0.022])

last_print_time = 0.0

sim.viewer = sim.launch()

while sim.viewer.is_running():

    t_start = time.perf_counter()
    dt = sim.dt

    # ✅ 输入层（完全解耦）
    target_pos, target_quat = controller.update(
        target_pos, target_quat, dt
    )

    # 可视化
    sim.model.body_pos[target_id] = target_pos

    # IK + 控制
    q_command = ik_solver.solve(q_current, target_pos, target_quat)
    sim.step(q_command)
    q_current = q_command

    # 监控
    now = time.perf_counter()
    if now - last_print_time > 0.1:
        current_rot = sim.data.site_xmat[sim.site_id].reshape(3, 3)
        target_rot = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()

        r_err = np.linalg.norm(IKSolver._rotation_error(current_rot, target_rot))
        p_err = np.linalg.norm(target_pos - sim.data.site_xpos[sim.site_id])

        print(f"\rPos Err: {p_err:.4f} | Rot Err: {r_err:.4f}", end="")
        last_print_time = now

    sleep_time = max(0.0, dt - (time.perf_counter() - t_start))
    if sleep_time > 0:
        time.sleep(sleep_time)

