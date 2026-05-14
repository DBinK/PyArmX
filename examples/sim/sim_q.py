import time

import numpy as np

from pyarmx.sim import ArmSimulator
from pyarmx.input.keyboard import JointInput

from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate, Timer


# MODEL_PATH = "xml/L20/scene.xml"
MODEL_PATH = "xml/L801/scene.xml"
ARM_DOF = 6 

sim = ArmSimulator(MODEL_PATH, arm_dof=ARM_DOF)

# 启动仿真
sim.viewer = sim.launch()
# sim.start()

# 初始化关节输入控制器
joint_input = JointInput(joint_speed=1.0)

# 主循环
loop  = Rate(hz=100)
timer = Timer(duration=0.1)

# 获取初始关节角作为起始目标
q_target = sim.get_q_current()

while sim.viewer.is_running() and loop.sleep():  # type: ignore

    # 获取当前关节角（如果是仿真环境，可能需要从 sim 获取当前实际位置）
    # 假设 sim.get_q() 返回当前关节角
    current_q = sim.get_q_current()
    
    # 根据键盘输入更新目标关节角
    q_target = joint_input.update(current_q, sim.dt)
    
    # 步进仿真
    sim.step(q_target)

    # 监控
    if timer.done:
        print(f"\rq_target: {fmt_arr(q_target)}", end="")
        timer.reset()