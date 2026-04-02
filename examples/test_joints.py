import time

from looptick import LoopTick
from pydamiao import SerialBus
from pydamiao.structs import MotorType
from rich import print as rprint

from pyarmx.joint import Joint, JointCfg, JointManager

# 创建串口总线
bus = SerialBus("COM9", baudrate=921600, timeout=0.01)

# 创建关节参数
wrist_3_cfg  = JointCfg(MotorType.DM4310, 0x06, 0x16, "wrist_3", pos_min=-30, pos_max=30)
wrist_2_cfg  = JointCfg(MotorType.DM4310, 0x05, 0x15, "wrist_2", pos_min=-10, pos_max=10)
wrist_1_cfg  = JointCfg(MotorType.DM4310, 0x04, 0x14, "wrist_1", pos_min=-10, pos_max=10)
elbow_cfg    = JointCfg(MotorType.DM4340, 0x03, 0x13, "elbow", pos_min=-10, pos_max=10)
shoulder_cfg = JointCfg(MotorType.DM4340, 0x02, 0x12, "shoulder", pos_min=-10, pos_max=10)
base_cfg     = JointCfg(MotorType.DM4340, 0x01, 0x11, "base", pos_min=-10, pos_max=10)

# 创建关节
wrist_3 = Joint(wrist_3_cfg, bus)
wrist_2 = Joint(wrist_2_cfg, bus)
wrist_1 = Joint(wrist_1_cfg, bus)
elbow = Joint(elbow_cfg, bus)
shoulder = Joint(shoulder_cfg, bus)
base = Joint(base_cfg, bus)

# 注册到关节管理器
manager = JointManager(bus)
joints = [wrist_3, wrist_2, wrist_1, elbow, shoulder, base]
for joint in joints:
    manager.add_joint(joint)


# 控制示例
loop = LoopTick()

timeout = 115
now = time.time()


manager.set_zero()
manager.enable()

while (time.time() - now) < timeout:

    time.sleep(0.001)

    pos_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    manager.set_pos_list(pos_list)
    # manager.update()

    # wrist_3.set_pos(0.0)

    pos_list = manager.get_joints_pos()

    ms = loop.tick_ms()
    print(f"\r{ms:.2f} {pos_list}", end="")



