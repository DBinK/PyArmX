import numpy as np

from pyarmx.input.keyboard import PlaybackInput
from pyarmx.input.zmq_sub import PoseReceiver

from pyarmx.real_facade import ArmRealFacade
from pyarmx.sim_facade import ArmSimFacade

from pyarmx.structs import Pose

from pyarmx.utils.log import logger
from pyarmx.utils.loops import Rate, Timer

if __name__ == "__main__":

    # 初始化真实机械臂
    # real = ArmRealFacade(mock=True)  # 仅仿真测试
    real = ArmRealFacade(mock=False)  

    # 初始化仿真机械臂
    model_path= "xml/L801/scene.xml"
    sim = ArmSimFacade(model_path)
    
    # 初始化输入源
    pose_input = PoseReceiver()
    playback = PlaybackInput()
    
    # 初始化目标位姿
    init_pos  = [0.008, 0.072, 0.086]           # 正前方可达位置 
    init_quat = [0.006, -0.005, -0.022, 1.000]  # 方向向下
    target_7d = Pose.from_pos_quat(init_pos, init_quat)
    current_7d = Pose.from_pos_quat(init_pos, init_quat)
    
    # 初始化循环调度器
    loop = Rate(hz=100)
    timer = Timer(duration=0.1)
    
    try:
        logger.info("系统就绪。使用键盘移动红色目标点，机械臂将平滑追踪。")

        while sim.running and loop.sleep(): 
            
            # 更新输入与目标
            pose_input.update(target_7d)
            sim.set_ee_pose(target_7d.pos, target_7d.quat)
            sim.step() 

            real.set_q_current(sim.q_current)  # 使用仿真中的关节角控制真实机械臂

            # # 监控日志
            if timer.done:
                current_pos, current_quat = sim.get_ee_pose()
                current_7d.update(current_pos, current_quat)
                pos_err = np.linalg.norm(current_7d.pos - target_7d.pos)
                quat_err = 1.0 - np.abs(np.dot(current_7d.quat, target_7d.quat))

                print(f"\r{pos_err=:.6f}, {quat_err=:.6f}", end="")
                timer.reset()  # 重设定时器

    except KeyboardInterrupt:
        logger.info("\n收到中断信号，准备退出...")
