
import time

import numpy as np
from rich import print as rprint

from pyarmx.sim import ArmSimulator
from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate


from pydamiao.arm.config import joint_cfgs
from pydamiao.arm.joint import JointManager
# from pydamiao.arm.vis import rrlog_joints
from pydamiao.bus import SerialBus

bus = SerialBus("COM9", baudrate=921600, timeout=0.01)
manager = JointManager(bus)

# 注册joint
manager.register(joint_cfgs)

# 设置初始状态
manager.clean_error()
# manager.set_zero_hard()
manager.enable()
manager.set_teach_mode()

q_command = manager.get_joints_pos()
rprint(q_command)

input("已进入示教模式, 按下回车开始录制")

MODEL_PATH = "xml/mjcf/scene.xml"
ARM_DOF = 6

sim = ArmSimulator(MODEL_PATH, arm_dof=ARM_DOF)

# 初始状态
q_current = sim.get_q_current()

# 启动仿真
sim.viewer = sim.launch()

# 主循环
last_print_time = 0.0
while sim.viewer.is_running():

    t_start = time.perf_counter()  # 获取循环开始时间戳, 稍后用于帧率控制

    manager.update()
    q_command = manager.get_joints_pos_list()

    sim.step(np.asanyarray(q_command))

    # 监控
    now = time.perf_counter()
    if now - last_print_time > 0.1:

        q_str = fmt_arr(q_command)

        print(f"Q: {q_str}")
        # print(f"\rPos Err: {p_err:.4f} | Rot Err: {r_err:.4f}", end="")
        last_print_time = now

    # 帧率控制
    # sleep_time = max(0.0, sim.dt - (time.perf_counter() - t_start))
    # if sleep_time > 0:
    #     time.sleep(sleep_time)

