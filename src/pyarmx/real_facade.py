from enum import IntEnum

import numpy as np
from loguru import logger
from pydamiao.arm.joint import JointCfg, JointManager
from pydamiao.bus import SerialBus
from pydamiao.structs import ControlMode, MotorType

class JointID(IntEnum):
    """与 slave_id 映射"""
    base     = 0x01
    shoulder = 0x02
    elbow    = 0x03
    wrist_1  = 0x04
    wrist_2  = 0x05
    wrist_3  = 0x06

joint_cfgs = [                                    
    JointCfg(MotorType.DM4340, JointID.base, 0x11, "base",     direction=-1),
    JointCfg(MotorType.DM4340, JointID.shoulder, 0x12, "shoulder", direction=-1),
    JointCfg(MotorType.DM4340, JointID.elbow, 0x13, "elbow",    direction=-1),
    JointCfg(MotorType.DM4310, JointID.wrist_1, 0x14, "wrist_1",  direction=-1),
    JointCfg(MotorType.DM4310, JointID.wrist_2, 0x15, "wrist_2",  direction=-1, offset=np.deg2rad(-45)),
    JointCfg(MotorType.DM4310, JointID.wrist_3, 0x16, "wrist_3",  direction=-1),  
]



class ArmRealFacade:
    def __init__(self, port: str = "COM9", baudrate: int = 921600, mock: bool = False):
        self.port = port
        self.dry_run = mock
        
        # 维护一个内部的虚拟关节状态，初始假设全为 0
        self._mock_q = np.zeros(len(joint_cfgs))

        if self.dry_run:
            logger.warning("已开启 Dry Run 模式，程序将绕过物理硬件通信")
            return

        self.bus = SerialBus(self.port, baudrate=baudrate, timeout=0.01)
        self.manager = JointManager(self.bus)
        
        # 注册关节并初始化状态
        self.manager.register(joint_cfgs)
        self.manager.clean_error()
        self.manager.enable()
        self.manager.set_mode(ControlMode.POS_VEL)
        
        logger.info(f"真实机械臂硬件就绪，连接端口: {self.port}")

    def get_q_current(self) -> np.ndarray:
        """获取真实机械臂当前关节角"""
        if self.dry_run:
            # 仿真硬件返回状态
            return self._mock_q.copy()
            
        return np.asanyarray(self.manager.get_joints_pos_list())

    def set_q_current(self, q_command: np.ndarray) -> None:
        """接收关节角指令并下发至硬件"""
        if self.dry_run:
            # 拦截指令并更新虚拟状态
            self._mock_q = q_command.copy()
            return
            
        self.manager.set_pos_list(q_command.tolist(), ControlMode.POS_VEL)

    def close(self) -> None:
        """安全下电"""
        if self.dry_run:
            logger.info("Dry Run 模式安全退出")
            return
            
        self.manager.disable()
        self.bus.close()
        logger.warning("真实机械臂已安全下电")


if __name__ == "__main__":
    # 使用 dry_run=True 即可脱离硬件进行逻辑测试
    arm = ArmRealFacade(mock=True)
    
    test_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    arm.set_q_current(test_q)
    print("当前读取的关节角:", arm.get_q_current())
    
    arm.close()