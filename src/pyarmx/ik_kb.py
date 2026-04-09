import time
import keyboard
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

# =========================
# 🔧 数学工具
# =========================
def rotation_error(R_current, R_target):
    R_err = R_target @ R_current.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])

def clamp_norm(x, max_norm):
    n = np.linalg.norm(x)
    if n > max_norm and n > 1e-12:
        return x / n * max_norm
    return x

def adaptive_damping_from_svd(J, lam_min=1e-4, lam_max=5e-2, sigma_ref=0.05):
    s = np.linalg.svd(J, compute_uv=False)
    sigma_min = s[-1] if len(s) > 0 else 0.0
    ratio = np.clip(sigma_min / sigma_ref, 0.0, 1.0)
    lam = lam_max * (1.0 - ratio) ** 2 + lam_min
    return lam, sigma_min

def solve_ik_position_level(
    model, data, site_id, q_init, target_pos, target_rot, arm_dof, q_min, q_max,
    max_iters=8, pos_weight=1.0, rot_weight=0.3, step_max=0.08, pos_tol=1e-4, rot_tol=2e-3,
):
    q = q_init.copy()
    for _ in range(max_iters):
        data.qpos[:arm_dof] = q
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[site_id].copy()
        current_rot = data.site_xmat[site_id].reshape(3, 3).copy()

        pos_err = target_pos - current_pos
        rot_err = rotation_error(current_rot, target_rot)

        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
            break

        err = np.concatenate([pos_weight * pos_err, rot_weight * rot_err])
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([pos_weight * jacp[:, :arm_dof], rot_weight * jacr[:, :arm_dof]])

        lam, _ = adaptive_damping_from_svd(J)
        H = J.T @ J + lam * np.eye(arm_dof)
        g = J.T @ err

        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dq = np.linalg.pinv(H) @ g

        dq = clamp_norm(dq, step_max)
        q = np.clip(q + dq, q_min, q_max)
    return q

# =========================
# 🔧 加载模型
# =========================
model = mujoco.MjModel.from_xml_path("xml/gripper/scene.xml")
data = mujoco.MjData(model)
site_id = model.site("ee").id
target_id = model.body("target").id

ARM_DOF = 6
dt = model.opt.timestep
mujoco.mj_forward(model, data)

# 状态变量
target_pos = model.body_pos[target_id].copy()
lock_rot = data.site_xmat[site_id].reshape(3, 3).copy()
q_min, q_max = model.jnt_range[:ARM_DOF, 0].copy(), model.jnt_range[:ARM_DOF, 1].copy()
q_sol = data.qpos[:ARM_DOF].copy()

# 控制参数
target_speed = 0.15
rot_speed = 1.0  # rad/s
ik_max_iters = 8
pos_weight = 1.0
rot_weight = 0.5  # 增加姿态权重以提高跟随精度
last_print_time = 0.0

# =========================
# 🚀 仿真主循环
# =========================
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.perf_counter()

        # --- 键盘位置控制 ---
        move_dir = np.zeros(3)
        speed_scale = 4.0 if keyboard.is_pressed('space') else 1.0
        
        if keyboard.is_pressed('up'):    move_dir[1] += 1.0
        if keyboard.is_pressed('down'):  move_dir[1] -= 1.0
        if keyboard.is_pressed('left'):  move_dir[0] -= 1.0
        if keyboard.is_pressed('right'): move_dir[0] += 1.0
        if keyboard.is_pressed('alt'):   move_dir[2] += 1.0
        if keyboard.is_pressed('ctrl'):  move_dir[2] -= 1.0

        if np.linalg.norm(move_dir) > 1e-12:
            target_pos += (move_dir / np.linalg.norm(move_dir)) * target_speed * speed_scale * dt

        # --- 键盘姿态控制 (旋转轴) ---
        rot_vec = np.zeros(3)
        # X轴: - 和 =
        if keyboard.is_pressed('-'): rot_vec[0] -= 1.0
        if keyboard.is_pressed('='): rot_vec[0] += 1.0
        # Y轴: [ 和 ]
        if keyboard.is_pressed('['): rot_vec[1] -= 1.0
        if keyboard.is_pressed(']'): rot_vec[1] += 1.0
        # Z轴: ; 和 '
        if keyboard.is_pressed(';'): rot_vec[2] -= 1.0
        if keyboard.is_pressed("'"): rot_vec[2] += 1.0

        if np.linalg.norm(rot_vec) > 1e-12:
            # 计算这一帧的旋转增量 (Axis-Angle to Rotation Matrix)
            delta_ang = rot_vec * rot_speed * speed_scale * dt
            delta_rot_mat = R.from_rotvec(delta_ang).as_matrix()
            # 更新目标旋转矩阵 (左乘表示在世界坐标系下旋转，右乘表示在末端坐标系下旋转)
            lock_rot = delta_rot_mat @ lock_rot

        # 重置位姿
        if keyboard.is_pressed('r'):
            mujoco.mj_forward(model, data)
            lock_rot = data.site_xmat[site_id].reshape(3, 3).copy()
            target_pos = data.site_xpos[site_id].copy()

        # 更新可视化目标
        model.body_pos[target_id] = target_pos
        # 如果需要可视化目标朝向，可以更新 target body 的成对属性（需 XML 支持）

        # --- 执行 IK ---
        q_sol = solve_ik_position_level(
            model, data, site_id, q_sol, target_pos, lock_rot,
            ARM_DOF, q_min, q_max, ik_max_iters, pos_weight, rot_weight
        )

        # --- 驱动与仿真 ---
        data.ctrl[:ARM_DOF] = q_sol
        mujoco.mj_step(model, data)
        viewer.sync()

        # --- 打印误差 ---
        now = time.perf_counter()
        if now - last_print_time > 0.1:
            current_rot = data.site_xmat[site_id].reshape(3, 3)
            r_err_norm = np.linalg.norm(rotation_error(current_rot, lock_rot))
            p_err_norm = np.linalg.norm(target_pos - data.site_xpos[site_id])
            print(f"\rPos Err: {p_err_norm:.4f} | Rot Err: {r_err_norm:.4f}", end="")
            last_print_time = now

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, dt - elapsed))