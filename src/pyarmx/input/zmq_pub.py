import time
import zmq
import msgpack
import numpy as np

class PoseSender:
    """基于 msgpack 的极速位姿发送器"""
    def __init__(self, ip: str = "127.0.0.1", port: int = 5554):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        
        # 发送端作为灵活的节点，使用 connect 连入主循环
        self.socket.connect(f"tcp://{ip}:{port}")
        
        # 解决 ZMQ 的 "Slow Joiner" 综合征：
        # connect 是异步的，刚连上立刻发数据可能会丢包，休眠一小会儿保证链路建立
        time.sleep(0.1) 

    def send_target(self, pose_array: np.ndarray | list, **kwargs) -> None:
        """
        发送目标位姿，支持附加任意额外参数 (kwargs)。
        """
        # 如果是 numpy 数组，必须转为 list 才能被标准 msgpack 序列化
        if isinstance(pose_array, np.ndarray):
            pose_array = pose_array.tolist()
            
        # 构建消息字典
        msg = {
            "pose_7d": pose_array,
            "timestamp": time.time(),
        }
        # 将额外的自定义参数（如速度比例、夹爪状态）合并进去
        msg.update(kwargs)
        
        # 序列化并发送 (use_bin_type=True 提高二进制打包效率)
        packed_data = msgpack.packb(msg, use_bin_type=True)
        self.socket.send(packed_data)

    def close(self) -> None:
        self.socket.close()
        self.ctx.term()

if __name__ == "__main__":
    sender = PoseSender()
    while True:
        sender.send_target([1, 2, 3, 4, 5, 6, 7])
        time.sleep(1)