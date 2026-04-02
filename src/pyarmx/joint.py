import math
import time
from dataclasses import dataclass

from pydamiao import Motor, MotorState, Result, SerialBus
from pydamiao.structs import MotorId, MotorType
from pydamiao.utils import limit_min_max


# 控制电机运动示例
@dataclass
class JointCfg():
    # 电机属性
    motor_type: MotorType
    slave_id: MotorId
    master_id: MotorId
    name: str

    # MIT 控制参数
    pos: float    = 0.0  # 默认位置
    vel: float    = 0.0
    kp: float     = 15.0
    kd: float     = 1.0
    torque: float = 0.0

    # 机器人关节参数
    pos_min: float = -math.pi    # 默认最小位置
    pos_max: float = math.pi     # 默认最大位置  
    reverse: bool  = False


class Joint:
    def __init__(self, cfg: JointCfg, bus: SerialBus):
        self.cfg = cfg
        self.name = cfg.name
        self.slave_id = cfg.slave_id
        self.motor = Motor(bus, cfg.motor_type, cfg.slave_id, cfg.master_id, cfg.name)
    
    def update(self) -> Result[MotorState]:
        """更新关节状态"""
        return self.motor.refresh_state()

    def set_zero(self):
        """将当前关节位置设置为零点"""
        self.motor.set_zero()

    def get_pos(self) -> float:
        """获取当前关节位置"""
        return self.motor.pos
    
    def get_vel(self) -> float:
        """获取当前关节速度"""
        return self.motor.vel
    
    def get_torque(self) -> float:
        """获取当前关节力矩"""
        return self.motor.torque
    
    def set_pos(self, pos: float):
        """设置关节位置"""
        limit_min_max(pos, self.cfg.pos_min, self.cfg.pos_max)
        self.motor.set_mit(pos, self.cfg.vel, self.cfg.kp, self.cfg.kd, self.cfg.torque)


class JointManager:
    def __init__(self, bus: SerialBus):
        self.bus = bus
        self.joints_by_name: dict[str, Joint] = {}
        self.joints_by_slave_id: dict[MotorId, Joint] = {}
    
    ### 关节管理 ###

    def add_joint(self, joint: Joint):
        self.joints_by_name[joint.name] = joint
        self.joints_by_slave_id[joint.slave_id] = joint

    def register(self, joint_cfgs: list[JointCfg]) -> dict[MotorId, Joint]:
        for joint_cfg in joint_cfgs:
            joint = Joint(joint_cfg, self.bus)
            self.add_joint(joint)
        return self.joints_by_slave_id

    def get_joint(self, name: str) -> Joint:
        return self.joints_by_name[name]
    
    def get_joint_by_id(self, slave_id: MotorId) -> Joint:
        return self.joints_by_slave_id[slave_id]
    
    @property
    def joints(self) -> list[Joint]:
        sorted_ids = sorted(self.joints_by_slave_id.keys())
        return [self.joints_by_slave_id[slave_id] for slave_id in sorted_ids]

    ### 关节状态 ###

    def update(self):
        for joint in self.joints_by_name.values():
            joint.update()

    def get_joints_pos(self) -> dict[MotorId, float]:
        return {joint.slave_id: joint.motor.pos for joint in self.joints_by_name.values()}
    
    def get_joints_vel(self) -> dict[MotorId, float]:
        return {joint.slave_id: joint.motor.vel for joint in self.joints_by_name.values()}
    
    def get_joints_torque(self) -> dict[MotorId, float]:
        return {joint.slave_id: joint.motor.torque for joint in self.joints_by_name.values()}

    ### 关节控制 ###
    def clean_error(self):
        for joint in self.joints_by_name.values():
            joint.motor.clean_error()

    def enable(self):
        for joint in self.joints_by_name.values():
            joint.motor.enable()
    
    def disable(self):
        for joint in self.joints_by_name.values():
            joint.motor.disable()

    def set_zero(self):
        for joint in self.joints_by_name.values():
            joint.set_zero()

    def set_pos(self, pos_dict: dict[MotorId, float]):
        for slave_id, pos in pos_dict.items():
            joint = self.joints_by_slave_id[slave_id]
            joint.set_pos(pos)

    def set_pos_list(self, pos_list: list[float]):
        pos_dict = self.pos_list_to_dict(pos_list)
        self.set_pos(pos_dict)

    ### 辅助函数 ###
    def pos_list_to_dict(self, pos_list: list[float]) -> dict[MotorId, float]:
        sorted_ids = sorted(self.joints_by_slave_id.keys())
        return {slave_id: pos for slave_id, pos in zip(sorted_ids, pos_list)}

    def pos_dict_to_list(self, pos_dict: dict[MotorId, float]) -> list[float]:
        sorted_ids = sorted(self.joints_by_slave_id.keys())
        return [pos_dict[slave_id] for slave_id in sorted_ids]
    