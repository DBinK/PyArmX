

import zmq
import msgpack
import numpy as np

from pyarmx.utils.log import logger
from pyarmx.structs import Pose

class PoseReceiver:
    """基于 msgpack 和 ZMQ CONFLATE 的极速位姿接收器"""
    def __init__(self, port: int = 5554):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        
        # 开启 CONFLATE：只保留最新一条消息，丢弃旧消息，实现零延迟
        self.socket.setsockopt(zmq.CONFLATE, 1)
        
        # 接收端作为稳定的核心节点，使用 bind
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        
        self._latest_target = None

    def fetch_latest(self) -> dict | None:
        """
        非阻塞获取最新指令。
        返回解析后的字典，若本周期无新消息则返回 None。
        """
        try:
            # 接收原始二进制流
            raw_data = self.socket.recv(flags=zmq.NOBLOCK)
            
            # raw=False 保证字符串的键名被正确解码为 Python 的 str
            latest_msg = msgpack.unpackb(raw_data, raw=False)
            self._latest_target = latest_msg
            
            return self._latest_target
            
        except zmq.Again:
            return None  # 队列为空，直接跳过
        except Exception as e:
            logger.error(f"msgpack 解析异常: {e}")
            return None
        
    def update(self, pose: Pose) -> bool:
        """
        将最新接收到的位姿数据原地更新到传入的 pose 对象中。
        避免创建新对象，适合高频控制循环。
        
        Args:
            pose: 需要更新的 Pose 对象引用（会被直接修改）
            
        Returns:
            bool: 更新成功返回 True，失败或无新数据返回 False
        """
        if self._latest_target is None:
            return False
        
        try:
            pose_7d = np.asarray(self._latest_target['pose_7d'], dtype=float)
            pose.array = pose_7d
            return True
            
        except KeyError as e:
            logger.error(f"消息格式错误，缺少字段: {e}")
            return False
        except Exception as e:
            logger.error(f"位姿更新失败: {e}")
            return False


    def close(self) -> None:
        self.socket.close()
        self.ctx.term()


if __name__ == "__main__":
    import time
    
    # 创建接收器实例
    receiver = PoseReceiver()

    # 循环接收并打印消息
    while True:
        msg = receiver.fetch_latest()
        if msg is not None:
            print(f"Received message: {msg}")

        # 模拟其他操作
        time.sleep(0.1)