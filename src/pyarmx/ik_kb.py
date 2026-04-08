import mujoco
import mujoco.viewer
import numpy as np
import keyboard
import time

# ... existing code ...

model = mujoco.MjModel.from_xml_path("xml/gripper/scene.xml")
data = mujoco.MjData(model)

site_id = model.site("ee").id
target_id = model.body("target").id

target_pos = np.array([0.1, 0.2, 0.3])

# 上一帧 dq（用于滤波）
dq_prev = np.zeros(6)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # ⌨️ 键盘输入
        dx = 0.0
        dy = 0.0
        dz = 0.0

        if keyboard.is_pressed('space'):
            speed = 5
        else:
            speed = 1

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

        # 更新目标
        target_pos[0] += dx * 0.001
        target_pos[1] += dy * 0.001
        target_pos[2] += dz * 0.001

        model.body_pos[target_id] = target_pos

        # === FK ===
        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[site_id].copy()
        error = target_pos - current_pos

        # =========================
        # ✅ Resolved-rate IK 核心
        # =========================

        # 1️⃣ task space velocity
        kp = 3.0
        v = kp * error

        # 2️⃣ 限制末端速度（非常关键）
        v_max = 30.3
        v_norm = np.linalg.norm(v)
        if v_norm > v_max:
            v = v / v_norm * v_max

        # 3️⃣ Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J = jacp[:, :6]

        # 4️⃣ DLS IK（解速度）
        lam = 0.05
        dq = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3)) @ v

        # 5️⃣ 限制关节速度（非常重要）
        dq_max = 1.0
        dq_norm = np.linalg.norm(dq)
        if dq_norm > dq_max:
            dq = dq / dq_norm * dq_max

        # 6️⃣ 低通滤波（抗抖）
        dq = 0.8 * dq_prev + 0.2 * dq
        dq_prev = dq

        # 7️⃣ 积分（关键：这里只能这样用）
        dt = 5.02  # 50Hz
        q = data.qpos[:6].copy()
        q_target = q + dq * dt

        print(f"\r q_target: {q_target}" , end="")

        # 位置控制
        data.ctrl[:6] = q_target

        for _ in range(1):
            mujoco.mj_step(model, data)

        viewer.sync()
        
        time.sleep(0.02)  # 保持50Hz频率
