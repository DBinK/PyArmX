import time
import keyboard
import mujoco
import mujoco.viewer
import numpy as np


# =========================
# 🔧 数学工具
# =========================
def rotation_error(R_current, R_target):
    """
    计算从当前姿态到目标姿态的旋转误差
    使用小角度近似，返回 3 维旋转向量
    """
    R_err = R_target @ R_current.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])


def clamp_norm(x, max_norm):
    """
    限制向量范数
    """
    n = np.linalg.norm(x)
    if n > max_norm and n > 1e-12:
        return x / n * max_norm
    return x


def adaptive_damping_from_svd(J, lam_min=1e-4, lam_max=5e-2, sigma_ref=0.05):
    """
    基于最小奇异值的连续阻尼
    比按 manipulability 分段跳变更平滑
    """
    s = np.linalg.svd(J, compute_uv=False)
    sigma_min = s[-1] if len(s) > 0 else 0.0

    ratio = np.clip(sigma_min / sigma_ref, 0.0, 1.0)
    lam = lam_max * (1.0 - ratio) ** 2 + lam_min
    return lam, sigma_min


def solve_ik_position_level(
    model,
    data,
    site_id,
    q_init,
    target_pos,
    target_rot,
    arm_dof,
    q_min,
    q_max,
    max_iters=8,
    pos_weight=1.0,
    rot_weight=0.3,
    step_max=0.08,
    pos_tol=1e-4,
    rot_tol=2e-3,
):
    """
    位置级 IK：
    输入当前初值 q_init 和目标末端位姿
    输出一组关节角 q，使末端尽可能接近目标位姿

    特点：
    - 每一帧直接求关节角，不走速度积分外环
    - 使用 LM / DLS 风格更新
    - 使用上一帧结果做 warm start
    """
    q = q_init.copy()

    for _ in range(max_iters):
        # 把当前迭代关节角写入 qpos
        data.qpos[:arm_dof] = q
        mujoco.mj_forward(model, data)

        # 当前末端位姿
        current_pos = data.site_xpos[site_id].copy()
        current_rot = data.site_xmat[site_id].reshape(3, 3).copy()

        # 位姿误差
        pos_err = target_pos - current_pos
        rot_err = rotation_error(current_rot, target_rot)

        # 收敛判断
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
            break

        # 构造加权误差
        err = np.concatenate([
            pos_weight * pos_err,
            rot_weight * rot_err
        ])

        # Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J = np.vstack([
            pos_weight * jacp[:, :arm_dof],
            rot_weight * jacr[:, :arm_dof]
        ])

        # 自适应连续阻尼
        lam, _ = adaptive_damping_from_svd(J)

        # LM / DLS 更新
        H = J.T @ J + lam * np.eye(arm_dof)
        g = J.T @ err

        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dq = np.linalg.pinv(H) @ g

        # 单步更新限幅，防止一次跳太大
        dq = clamp_norm(dq, step_max)

        # 更新关节角，并限制在 joint range 内
        q = np.clip(q + dq, q_min, q_max)

    return q


# =========================
# 🔧 加载模型
# =========================
model = mujoco.MjModel.from_xml_path("xml/gripper/scene.xml")
data = mujoco.MjData(model)

site_id = model.site("ee").id
target_id = model.body("target").id

# 假设前 6 个自由度是机械臂
ARM_DOF = 6

# 用模型里的时间步，而不是手写错 dt
dt = model.opt.timestep

# 初始 forward
mujoco.mj_forward(model, data)

# 初始 target 位置：从 target body 当前 body_pos 读出来
target_pos = model.body_pos[target_id].copy()

# 初始锁定姿态：锁定当前末端姿态
lock_rot = data.site_xmat[site_id].reshape(3, 3).copy()

# 关节范围
q_min = model.jnt_range[:ARM_DOF, 0].copy()
q_max = model.jnt_range[:ARM_DOF, 1].copy()

# warm start：上一帧解
q_sol = data.qpos[:ARM_DOF].copy()

# 控制参数
target_speed = 0.15      # m/s
ik_max_iters = 8         # 每帧 IK 迭代次数
pos_weight = 1.0
rot_weight = 0.25        # 姿态权重先低一些，更稳
step_max = 0.06          # 每次迭代的最大关节步长

# 显示用
last_print_time = 0.0


# =========================
# 🚀 viewer
# =========================
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.perf_counter()

        # ==================================
        # ⌨️ 键盘控制目标点（按时间缩放）
        # ==================================
        move_dir = np.zeros(3, dtype=np.float64)

        speed_scale = 4.0 if keyboard.is_pressed('space') else 1.0

        if keyboard.is_pressed('up'):
            move_dir[1] += 1.0
        if keyboard.is_pressed('down'):
            move_dir[1] -= 1.0
        if keyboard.is_pressed('left'):
            move_dir[0] -= 1.0
        if keyboard.is_pressed('right'):
            move_dir[0] += 1.0
        if keyboard.is_pressed('alt'):
            move_dir[2] += 1.0
        if keyboard.is_pressed('ctrl'):
            move_dir[2] -= 1.0

        # 重新锁定当前姿态
        if keyboard.is_pressed('r'):
            mujoco.mj_forward(model, data)
            lock_rot = data.site_xmat[site_id].reshape(3, 3).copy()

        # 目标点按时间更新，而不是按帧更新
        if np.linalg.norm(move_dir) > 1e-12:
            move_dir = move_dir / np.linalg.norm(move_dir)
        target_pos = target_pos + move_dir * target_speed * speed_scale * dt

        # 更新 target body 的显示位置
        model.body_pos[target_id] = target_pos

        # ==================================
        # 🧠 位置级 IK
        # ==================================
        q_seed = q_sol.copy()

        q_sol = solve_ik_position_level(
            model=model,
            data=data,
            site_id=site_id,
            q_init=q_seed,
            target_pos=target_pos,
            target_rot=lock_rot,
            arm_dof=ARM_DOF,
            q_min=q_min,
            q_max=q_max,
            max_iters=ik_max_iters,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            step_max=step_max,
            pos_tol=1e-4,
            rot_tol=2e-3,
        )

        # ==================================
        # 🎮 把 IK 解交给 actuator
        # ==================================
        data.ctrl[:ARM_DOF] = q_sol

        # 仿真一步
        mujoco.mj_step(model, data)
        viewer.sync()

        # ==================================
        # 📊 打印状态
        # ==================================
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[site_id].copy()
        current_rot = data.site_xmat[site_id].reshape(3, 3).copy()
        pos_err = target_pos - current_pos
        rot_err = rotation_error(current_rot, lock_rot)

        now = time.perf_counter()
        if now - last_print_time > 0.05:
            print(
                f"\rpos_err={np.linalg.norm(pos_err):.5f} "
                f"rot_err={np.linalg.norm(rot_err):.5f} "
                f"target=({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})",
                end=""
            )
            last_print_time = now

        # 尽量贴近仿真步长节奏
        elapsed = time.perf_counter() - t0
        sleep_time = max(0.0, dt - elapsed)
        time.sleep(sleep_time)