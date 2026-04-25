from queue import Queue
import time

import numpy as np
from rich import print as rprint

from pyarmx.input.keyboard import PlaybackInput
from pyarmx.input.zmq_sub import PoseReceiver

from pyarmx.real_facade import ArmRealFacade
from pyarmx.sim_facade import ArmSimFacade

from pyarmx.structs import Pose

from pyarmx.utils.log import logger
from pyarmx.utils.loops import Rate, Timer

from rose.core import Subscriber
from rose.message import RetMsg

ret_addr = "tcp://127.0.0.1:5555"
sub = Subscriber(ret_addr, "ret", RetMsg)


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
    # init_pos  = [-0.0100, 0.0200, 0.086]           # 正前方可达位置 
    init_pos  = [0.008, 0.072, 0.086]           # 正前方可达位置 
    init_quat = [0.006, -0.005, -0.022, 1.000]  # 方向向下
    target_7d = Pose.from_pos_quat(init_pos, init_quat)
    current_7d = Pose.from_pos_quat(init_pos, init_quat)
    
    # 初始化循环调度器
    loop = Rate(hz=100)
    timer = Timer(duration=0.1)

    tasks = []
    working_task = None
    def unpack_msg(ret: RetMsg, tasks: list):
        # 解析 RetMsg 并装填到 tasks 列表
        if hasattr(ret, 'objs') and hasattr(ret, 'acts'):
            objs = ret.objs  # {'nailong': [1807, -479], 'sticker': [2079, -625]}
            acts = ret.acts  # [['move_to', 'nailong'], ['grip'], ['move_to', 'sticker']]
            
            for act in acts:
                if len(act) >= 2 and act[0] == 'move_to':
                    obj_name = act[1]
                    if obj_name in objs:
                        coords = objs[obj_name]
                        coords_3d = [coords[1]/1000, -coords[0]/1000, init_pos[2]]
                        task = ["move_to", coords_3d, False]
                        tasks.append(task)
                else:
                    tasks.append([*act, False])


    try:
        logger.info("系统就绪。使用键盘移动红色目标点，机械臂将平滑追踪。")

        while sim.running and loop.sleep(): 

            ret = sub.recv(False)
            if ret is not None:
                rprint(ret)
                unpack_msg(ret, tasks)
            
            # 更新输入与目标
            # 更新输入与目标
            if working_task is None and len(tasks) > 0:
                for task in tasks:
                    if not task[-1]:  # 找到第一个未完成的任务
                        working_task = task
                        break
            
            if working_task is not None:
                task_type = working_task[0]
                if task_type == "move_to" and len(working_task) >= 2:
                    target_7d.update(working_task[1], target_7d.quat)
                    print(f"正在前往{working_task[0]}, pos: {working_task[1]}")

                elif task_type == "grip":
                    print("正在抓取")
                    pass  # 处理抓取动作
                    working_task[-1] = True  # 标记当前任务为完成
                    working_task = None  # 清空当前工作任务
                    time.sleep(5)

                elif task_type == "release":
                    print("正在释放")
                    pass  # 处理抓取动作
                    working_task[-1] = True  # 标记当前任务为完成
                    working_task = None  # 清空当前工作任务
                    time.sleep(5)
                    
                else:
                    print("动作解析失败")
            
            sim.set_ee_pose(target_7d.pos, target_7d.quat)
            sim.step() 

            real.set_q_current(sim.q_current)  # 使用仿真中的关节角控制真实机械臂

            # # 监控日志
            if timer.done:
                current_pos, current_quat = sim.get_ee_pose()
                current_7d.update(current_pos, current_quat)
                pos_err = np.linalg.norm(current_7d.pos - target_7d.pos)
                quat_err = 1.0 - np.abs(np.dot(current_7d.quat, target_7d.quat))

                if working_task is not None and working_task[0] == "move_to":
                    if pos_err < 0.01:  # 降低阈值，更精确
                        print(f"{working_task} 完成")
                        working_task[-1] = True  # 标记当前任务为完成
                        working_task = None  # 清空当前工作任务
               
                # elif working_task is not None and working_task[0] == "grip":
                #     # 当前直接完成, 未处理
                #     print(f"{working_task} 完成")
                #     working_task[-1] = True  # 标记当前任务为完成
                #     working_task = None  # 清空当前工作任务
                          
                # elif working_task is not None and working_task[0] == "release":
                #     # 当前直接完成, 未处理
                #     print(f"{working_task} 完成")
                #     working_task[-1] = True  # 标记当前任务为完成
                #     working_task = None  # 清空当前工作任务
 

                print(f"\r{pos_err=:.6f}, {quat_err=:.6f}", end="release")
                timer.reset()  # 重设定时器

    except KeyboardInterrupt:
        logger.info("\n收到中断信号，准备退出...")
