import mujoco
import mujoco.viewer
import numpy as np
import keyboard
import time

# =========================
# 🔧 工具函数：姿态误差
# =========================
def rotation_error(R_current, R_target):
    # 计算 so(3) 误差（轴角近似）
    R_err = R_target @ R_current.T
    return np.array([
        R_err[2,1] - R_err[1,2],
        R_err[0,2] - R_err[2,0],
        R_err[1,0] - R_err[0,1]
    ]) * 0.5


# =========================
# 🔧 加载模型
# =========================
model = mujoco.MjModel.from_xml_path("xml/gripper/scene.xml")
data = mujoco.MjData(model)

site_id = model.site("ee").id
target_id = model.body("target").id

target_pos = np.array([0.1, 0.2, 0.3])

# 上一帧 dq（滤波）
dq_prev = np.zeros(6)

# =========================
# 🚀 viewer
# =========================
with mujoco.viewer.launch_passive(model, data) as viewer:

    # 先 forward 一次
    mujoco.mj_forward(model, data)

    # 🔒 记录初始姿态（锁死用）
    init_rot = data.site_xmat[site_id].reshape(3, 3).copy()

    while viewer.is_running():

        # =========================
        # ⌨️ 键盘控制
        # =========================
        dx, dy, dz = 0.0, 0.0, 0.0

        speed = 5 if keyboard.is_pressed('space') else 1

        if keyboard.is_pressed('up'):
            dy = 1.0 * speed
        if keyboard.is_pressed('down'):
            dy = -1.0 * speed
        if keyboard.is_pressed('left'):
            dx = -1.0 * speed
        if keyboard.is_pressed('right'):
            dx = 1.0 * speed
        if keyboard.is_pressed('alt'):
            dz = 1.0 * speed
        if keyboard.is_pressed('ctrl'):
            dz = -1.0 * speed

        # 更新 target
        target_pos += np.array([dx, dy, dz]) * 0.001
        model.body_pos[target_id] = target_pos

        # =========================
        # 🔁 FK
        # =========================
        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[site_id].copy()
        current_rot = data.site_xmat[site_id].reshape(3, 3)

        pos_err = target_pos - current_pos
        rot_err = rotation_error(current_rot, init_rot)

        # =========================
        # 🎯 6D task velocity
        # =========================
        kp_pos = 3.0
        kp_rot = 2.0

        v_pos = kp_pos * pos_err
        v_rot = kp_rot * rot_err

        v = np.concatenate([v_pos, v_rot])

        # 限制 task 速度（关键）
        v_max = 30.0
        v_norm = np.linalg.norm(v)
        if v_norm > v_max:
            v = v / v_norm * v_max

        # =========================
        # 🧮 Jacobian
        # =========================
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J = np.vstack([jacp, jacr])[:, :6]  # 6x6

        # =========================
        # 🧠 DLS IK
        # =========================
        lam = 0.05
        dq = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6)) @ v

        # 限制关节速度
        dq_max = 1.0
        dq_norm = np.linalg.norm(dq)
        if dq_norm > dq_max:
            dq = dq / dq_norm * dq_max

        # 低通滤波（抗抖）
        dq = 0.8 * dq_prev + 0.2 * dq
        dq_prev = dq

        # =========================
        # 🧭 积分
        # =========================
        dt = 5.02  # 50Hz
        q = data.qpos[:6].copy()
        q_target = q + dq * dt

        print(f"\r q_target: {q_target}", end="")

        # =========================
        # 🎮 控制
        # =========================
        data.ctrl[:6] = q_target

        mujoco.mj_step(model, data)
        viewer.sync()

        # time.sleep(dt)