
import numpy as np

from pyarmx.input.keyboard import PoseInput
from pyarmx.input.zmq_pub import PoseSender

from pyarmx.structs import Pose

from pyarmx.utils.log import fmt_arr
from pyarmx.utils.loops import Rate

sender = PoseSender()
controller = PoseInput()

# 初始位姿
init_pos = [0.008, 0.072, 0.086]
init_quat = [0.006, -0.005, -0.022, 1.000]

target_7d = Pose.from_pos_quat(init_pos, init_quat)
last_7d = Pose.from_pos_quat(init_pos, init_quat)

hz = 100
loop = Rate(hz)


for tick in loop:

    new_pos, new_quat = controller.update(last_7d.pos, last_7d.quat, 1/hz)
    target_7d.update(new_pos, new_quat)

    # pos_diff = target_7d.pos_dist(last_7d)
    # quat_diff = target_7d.quat_dist(last_7d)

    # if pos_diff > 1e-4 or quat_diff > 1e-4:
    #     print(f"{pos_diff=}, {quat_diff=}")
    # else:
    #     print(f"{fmt_arr(last_7d.array)=} , {fmt_arr(target_7d.array)=}")

    sender.send_target(target_7d.array)

    last_7d.update(new_pos, new_quat)

    print(f"\r{fmt_arr(target_7d.pos)=}, {fmt_arr(target_7d.quat)=}", end="")
