
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

# # --- 真实机械臂 --- #
# bus = SerialBus("COM9", baudrate=921600, timeout=0.01)
# manager = JointManager(bus)

# # 注册joint
# manager.register(joint_cfgs)

# # 设置初始状态
# manager.clean_error()
# manager.enable()
# manager.set_teach_mode()


# --- 仿真机械臂 --- #
MODEL_PATH = "xml/mjcf/scene.xml"

sim = ArmSimulator(MODEL_PATH)
sim.viewer = sim.launch()  # 启动仿真 


# --- 主循环 --- #
loop = Rate(hz=100)
timer = Timer(duration=0.5)

while sim.viewer.is_running() and loop.sleep():

    # manager.update()
    # q_command = manager.get_joints_pos_list()
    q_command = [10.0] * 6

    sim.step(np.asanyarray(q_command))

    if timer.done:  # 限频打印
        q_str = fmt_arr(q_command)
        print(f"Q: {q_str}")

        timer.reset()  # 重置计时器
