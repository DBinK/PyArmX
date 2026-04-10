import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# 假设你的类保存在 pyarmx.planner 模块中
from pyarmx.interp import RuckigPosePlanner 
from pyarmx.ik import IKSolver
from pyarmx.sim import ArmSimulator, KeyboardController
from pyarmx.utils.log import fmt_arr

MODEL_PATH = "xml/mjcf/scene.xml"
ARM_DOF = 6

# ================= 1. 初始化仿真与控制 =================
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

# ================= 2. 初始化 Ruckig 规划器 =================
# control_period 建议与 sim.dt 保持一致，保证时间同步
planner = RuckigPosePlanner(control_period=sim.dt, buffer_size=100)

# 获取仿真初始位姿
# 初始状态
q_current = sim.get_q_current()
# init_pos, init_quat = sim.get_fk_quat(q_current)

init_pos = np.array([0.008, 0.072, 0.086])
init_quat = np.array([0.006, -0.005, -0.022, 1.000])  # [x, y, z, w] 格式

# 【重要】确认四元数格式
# 假设 sim.get_fk_quat 返回 [x, y, z, w]，这与 scipy 和 RuckigPosePlanner 内部一致
# 如果 MuJoCo 原生是 [w, x, y, z]，请在此处转换: init_quat = np.roll(init_quat, -1)
init_pose_7d = np.concatenate([init_pos, init_quat])

# 设置规划器的初始状态（必须与机械臂当前状态一致，否则第一步会跳变）
planner.set_init_pose(init_pose_7d)

# 启动规划器后台线程
planner.start()

# 启动仿真可视化
sim.viewer = sim.launch()

# ================= 3. 主循环 =================
last_print_time = 0.0

# 用于记录键盘控制的“最终目标”，初始化为当前位置
final_target_pos = init_pos.copy()
final_target_quat = init_quat.copy()

print("[Sim] 系统就绪。使用键盘移动红色目标点，机械臂将平滑追踪。")

while sim.viewer.is_running():
    t_start = time.perf_counter()

    # --- A. 输入层：键盘更新“最终目标” ---
    # controller.update 返回的是用户期望的最终位姿
    # 注意：这里我们只更新变量，不直接发给 IK
    new_target_pos, new_target_quat = controller.update(
        final_target_pos, final_target_quat, sim.dt
    )
    
    # 检测目标是否变化，如果变化则推送给 Planner
    # 简单的阈值判断，避免微小抖动导致频繁重规划
    pos_diff = np.linalg.norm(new_target_pos - final_target_pos)
    quat_diff = 1.0 - np.abs(np.dot(new_target_quat, final_target_quat))
    
    if pos_diff > 1e-4 or quat_diff < 0.9999:
        final_target_pos = new_target_pos
        final_target_quat = new_target_quat
        
        # 【关键步骤】将新的最终目标送入规划器
        # Planner 内部队列会自动覆盖旧目标，并重新计算轨迹
        target_7d = np.concatenate([final_target_pos, final_target_quat])
        planner.set_target(target_7d)
        
    # print(f"\n[Debug] set_target called! pos_diff={pos_diff:.4f}, quat_diff={quat_diff:.6f}")


    # 可视化：显示用户设定的“最终目标”
    sim.update_target_dot(final_target_pos)

    # --- B. 规划层：获取当前时刻的“平滑中间点” ---
    # 从 Planner 获取这一帧应该到达的位姿
    smooth_pose = planner.get_pose(block=False, timeout=0)
    
    if smooth_pose is None:
        # 如果 Planner 还没准备好（例如刚启动），使用当前实际位姿作为临时目标
        # 避免 IK 报错或机械臂乱动
        exec_pos, exec_quat = sim.get_fk_quat(q_current)
    else:
        exec_pos = smooth_pose[:3]
        exec_quat = smooth_pose[3:]

    # --- C. 控制层：IK 求解并执行 ---
    # 【关键区别】IK 求解的是 smooth_pose (中间点)，而不是 final_target (终点)
    q_command = ik_solver.solve(q_current, exec_pos, exec_quat)
    
    # 安全检查
    if q_command is None or np.any(np.isnan(q_command)):
        # IK 失败时保持静止或缓慢回退，这里简单保持上一帧
        q_command = q_current
        
    sim.step(q_command)

    # 更新当前关节状态
    q_current = q_command 

    # --- D. 监控日志 ---
    now = time.perf_counter()
    if now - last_print_time > 0.1:
        # 计算误差：实际位姿 vs 平滑轨迹点
        current_actual_pos, current_actual_quat = sim.get_fk_quat(q_current)
        
        # 计算位置误差
        p_err = np.linalg.norm(exec_pos - current_actual_pos)
        
        # 计算姿态误差
        current_rot = R.from_quat(current_actual_quat).as_matrix()
        target_rot = R.from_quat(exec_quat).as_matrix()
        r_err = np.linalg.norm(IKSolver._rotation_error(current_rot, target_rot))

        print(
            f"\rTrack Err P:{p_err:.4f} R:{r_err:.4f} | Target P:{fmt_arr(final_target_pos)}",
            end="",
        )
        last_print_time = now

    # --- E. 帧率控制 ---
    elapsed = time.perf_counter() - t_start
    sleep_time = max(0.0, sim.dt - elapsed)
    if sleep_time > 0:
        time.sleep(sleep_time)
