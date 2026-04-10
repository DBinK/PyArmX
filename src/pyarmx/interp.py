import time
import queue
import threading

import numpy as np
import ruckig

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class RuckigSlerpRunner:
    def __init__(self, control_period=0.008, buffer_size=100):

        self.dt = control_period

        # ===== Ruckig（只做 xyz）=====
        self.ruckig = ruckig.Ruckig(3, self.dt)
        self.input_param = ruckig.InputParameter(3)
        self.output_param = ruckig.OutputParameter(3)

        self.input_param.max_velocity = [0.5, 0.5, 0.5]
        self.input_param.max_acceleration = [5.0, 5.0, 5.0]
        self.input_param.max_jerk = [5.0, 5.0, 5.0]

        self.input_param.current_position = [0.0, 0.0, 0.0]
        self.input_param.current_velocity = [0.0, 0.0, 0.0]
        self.input_param.current_acceleration = [0.0, 0.0, 0.0]

        self.input_param.target_position = [0.0, 0.0, 0.0]

        # ===== 姿态（四元数）=====
        self.current_quat = np.array([0, 0, 0, 1], dtype=float)
        self.target_quat = self.current_quat.copy()

        self.slerp = None
        self.slerp_start_time = None
        self.slerp_duration = 1.0

        # ===== 内部队列（封装）=====
        self.target_queue = queue.Queue(maxsize=1)   # 只保留最新目标
        self.output_queue = queue.Queue(maxsize=buffer_size)

        # ===== 线程控制 =====
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

        self.initialized = False

    # ================= 初始化 =================
    def set_initial_current(self, pose: list[float]):
        """
        pose: [x,y,z,qx,qy,qz,qw]
        """
        with self.lock:
            self.input_param.current_position = pose[:3]
            self.input_param.target_position = pose[:3]

            q = np.array(pose[3:], dtype=float)
            q = q / np.linalg.norm(q)

            self.current_quat = q
            self.target_quat = q.copy()

            self.initialized = True

        print("[Init] 初始位姿:", pose)

    # ================= 外部接口：推送目标 =================
    def set_target(self, pose: list[float]):
        """
        pose: [x,y,z,qx,qy,qz,qw]
        """
        try:
            # 清空旧目标（关键）
            while not self.target_queue.empty():
                self.target_queue.get_nowait()
        except queue.Empty:
            pass

        self.target_queue.put(pose)

    # ================= 外部接口：获取输出 =================
    def get_pose(self, block=False, timeout=None) -> list[float] | None:
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    # ================= 内部：设置目标 =================
    def _set_target(self, pose: list[float]):
        with self.lock:
            self.input_param.target_position = pose[:3]

            q = np.array(pose[3:], dtype=float)
            q = q / np.linalg.norm(q)

            if np.dot(self.current_quat, q) < 0:
                q = -q

            self.target_quat = q

            # 获取轨迹时间
            self.ruckig.update(self.input_param, self.output_param)

            duration = self.output_param.trajectory.duration
            if duration <= 0:
                duration = self.dt

            self.slerp_duration = duration

            key_times = [0, duration]
            rots = R.from_quat([self.current_quat, self.target_quat])

            self.slerp = Slerp(key_times, rots)
            self.slerp_start_time = time.time()

    # ================= 主循环 =================
    def run_loop(self):
        print("[Runner] 启动循环")

        while not self.stop_event.is_set():
            t0 = time.time()

            if not self.initialized:
                time.sleep(self.dt)
                continue

            # ===== 取最新目标 =====
            try:
                while True:
                    target = self.target_queue.get_nowait()
                    self._set_target(target)
            except queue.Empty:
                pass

            # ===== Ruckig =====
            with self.lock:
                res = self.ruckig.update(self.input_param, self.output_param)

                if res == ruckig.Result.Error:
                    print("[Error] Ruckig失败")
                    continue

                new_pos = list(self.output_param.new_position)

                self.input_param.current_position = new_pos
                self.input_param.current_velocity = list(self.output_param.new_velocity)
                self.input_param.current_acceleration = list(self.output_param.new_acceleration)

            # ===== Slerp =====
            if self.slerp is not None and self.slerp_start_time is not None:
                t = time.time() - self.slerp_start_time
                t = np.clip(t, 0.0, self.slerp_duration)

                rot = self.slerp([t])[0]
                self.current_quat = rot.as_quat()

            # ===== 输出（防堆积）=====
            full_pose = new_pos + self.current_quat.tolist()

            if self.output_queue.full():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass

            self.output_queue.put(full_pose)

            # ===== 控制周期 =====
            elapsed = time.time() - t0
            remain = self.dt - elapsed
            if remain > 0:
                time.sleep(remain)

        print("[Runner] 结束")

    # ================= 控制 =================
    def start(self):
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self.run_loop,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()


# ================= Demo =================
if __name__ == "__main__":

    from pyarmx.utils.log import fmt_arr

    runner = RuckigSlerpRunner(0.008)

    runner.set_initial_current([0.2, 0.0, 0.1, 0, 0, 0, 1])
    runner.start()

    q1 = R.from_euler('z', np.pi).as_quat()
    runner.set_target([0.3, 0.0, 0.1, *q1])

    time.sleep(1)

    q2 = R.from_euler('xyz', [np.pi/2, 0, np.pi/2]).as_quat()
    runner.set_target([0.15, 0.15, 0.2, *q2])
    
    for i in range(5000):
        pose = runner.get_pose(timeout=0.1)
        if pose:
            print(i, fmt_arr(pose))
        time.sleep(0.001)

    runner.stop()