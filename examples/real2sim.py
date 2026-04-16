
import time

import numpy as np
from pydamiao.arm.config import joint_cfgs
from pydamiao.arm.joint import JointManager

# from pydamiao.arm.vis import rrlog_joints
from pydamiao.bus import SerialBus
from rich import print as rprint

from pyarmx.sim import ArmSimulator
from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate, Timer

# --- 真实机械臂 --- #
bus = SerialBus("COM9", baudrate=921600, timeout=0.01)
manager = JointManager(bus)

# 注册joint
manager.register(joint_cfgs)

# 设置初始状态
manager.clean_error()
manager.enable()
manager.set_teach_mode()


# --- 仿真机械臂 --- #
MODEL_PATH = "xml/L20/scene.xml"

sim = ArmSimulator(MODEL_PATH)
sim.start()

# --- 主循环 --- #
loop = Rate(hz=100)
timer = Timer(duration=0.5)

i = 0.01

# while sim.viewer.is_running() and loop.sleep():
while loop.sleep():

    manager.set_teach_mode(False)
    q_real = manager.get_joints_pos_list()

    # bs = np.sin(i:=i + 0.02)
    # q_command = np.asanyarray([bs, 0.0, 0.0, 0.0, 0.0, 0.0])
    # q_command = [0.0] * 6

    q_real_str = fmt_arr(q_real)
    # q_command_str = ""

    # sim.step(np.asanyarray(q_command))
    sim.set_q_target(np.asanyarray(q_real))

    q_sim = sim.get_q_current()
    q_sim_str = fmt_arr(q_sim)

    # q_current_str = ""

    print(f"{q_real_str=}, {q_sim_str=}, on_time: {loop.tick.delta:.6f} {loop.tick.on_time}")

    # if timer.done:  # 限频打印
    #     q_str = fmt_arr(q_command)
    #     print(f"Q: {q_str}")

    #     timer.reset()  # 重置计时器
