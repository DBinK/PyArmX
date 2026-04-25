import time
import queue
import threading

import numpy as np
import ruckig

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class RuckigPosePlanner:
    def __init__(self, control_period=0.008, buffer_size=100):

        self.dt = control_period

        # ===== Ruckig（只做 xyz）=====
        self.ruckig = ruckig.Ruckig(3, self.dt)
        self.input_param = ruckig.InputParameter(3)
        self.output_param = ruckig.OutputParameter(3)

        self.input_param.max_velocity = [500.0, 500.0, 500.0]
        self.input_param.max_acceleration = [500.0, 500.0, 500.0]
        self.input_param.max_jerk = [5.0, 5.0, 5.0]
        self._current_position = np.zeros(3, dtype=np.float64)
        self._current_velocity = np.zeros(3, dtype=np.float64)
        self._current_acceleration = np.zeros(3, dtype=np.float64)
        self._target_position = np.zeros(3, dtype=np.float64)

        self.input_param.current_position = self._current_position.tolist()
        self.input_param.current_velocity = self._current_velocity.tolist()
        self.input_param.current_acceleration = self._current_acceleration.tolist()
        self.input_param.target_position = self._target_position.tolist()

        # ===== 姿态（四元数）=====
        self.current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.target_quat = self.current_quat.copy()

        self.slerp = None
        self.slerp_start_time = None
        self.slerp_duration = 1.0

        # ===== 内部队列（封装）=====
        self.target_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=buffer_size)

        # ===== 线程控制 =====
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

        self.initialized = False

    # ================= 初始化 =================
    def set_init_pose(self, pose: list[float] | np.ndarray):
        """
        pose: [x,y,z,qx,qy,qz,qw] 或 numpy array
        """
        pose_arr = np.asarray(pose, dtype=np.float64)
        
        with self.lock:
            self._current_position = pose_arr[:3].copy()
            self._target_position = pose_arr[:3].copy()
            
            self.input_param.current_position = self._current_position.tolist()
            self.input_param.target_position = self._target_position.tolist()

            q = pose_arr[3:]
            q = q / np.linalg.norm(q)

            self.current_quat = q
            self.target_quat = q.copy()

            self.initialized = True

        print("[Init] 初始位姿:", pose_arr.tolist())

    # ================= 外部接口：推送目标 =================
    def set_target(self, pose: list[float] | np.ndarray):
        """
        pose: [x,y,z,qx,qy,qz,qw] 或 numpy array
        """
        try:
            while not self.target_queue.empty():
                self.target_queue.get_nowait()
        except queue.Empty:
            pass

        self.target_queue.put(np.asarray(pose, dtype=np.float64))

    # ================= 外部接口：获取输出 =================
    def get_pose(self, block=False, timeout=None) -> np.ndarray | None:
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    # ================= 内部：设置目标 =================
    def _set_target(self, pose: np.ndarray):
        with self.lock:
            self._target_position = pose[:3].copy()
            self.input_param.target_position = self._target_position.tolist()

            q = pose[3:].copy()
            q = q / np.linalg.norm(q)

            if np.dot(self.current_quat, q) < 0:
                q = -q

            self.target_quat = q

            # 获取轨迹时间
            try:
                res = self.ruckig.update(self.input_param, self.output_param)
                if res != ruckig.Result.Working and res != ruckig.Result.Finished:
                    print(f"[Warn] _set_target 规划失败 (Result: {res})，保持上一目标")
                    # 恢复 target_position 为上一次成功的状态，或者保持当前 input_param 不变
                    # 这里选择简单处理：不更新 slerp，保持原有运动趋势
                    return

                duration = self.output_param.trajectory.duration
                if duration <= 0:
                    duration = self.dt

                self.slerp_duration = duration

                key_times = [0.0, duration]
                rots = R.from_quat(np.vstack([self.current_quat, self.target_quat]))

                self.slerp = Slerp(key_times, rots)
                # 只在首次初始化或上一段轨迹完成后才重置时间
                if self.slerp_start_time is None or (time.time() - self.slerp_start_time) >= self.slerp_duration:
                    self.slerp_start_time = time.time()
                    
            except ruckig.RuckigError as e:
                print(f"[Error] _set_target 发生异常: {e}，跳过本次目标更新")
                # 关键：发生异常时不要崩溃，直接返回，保持机器人按原轨迹运动
                return

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
            
            new_pos = self._current_position.copy()
            new_vel = self._current_velocity.copy()
            new_acc = self._current_acceleration.copy()
            success = False

            # ===== Ruckig =====
            with self.lock:
                try:
                    res = self.ruckig.update(self.input_param, self.output_param)

                    if res == ruckig.Result.Finished:
                        # 轨迹完成，保持目标状态作为当前状态
                        new_pos = np.array(self.input_param.target_position, dtype=np.float64)
                        new_vel = np.zeros(3, dtype=np.float64)
                        new_acc = np.zeros(3, dtype=np.float64)
                        success = True
                    elif res == ruckig.Result.Working:
                        new_pos = np.array(self.output_param.new_position, dtype=np.float64)
                        new_vel = np.array(self.output_param.new_velocity, dtype=np.float64)
                        new_acc = np.array(self.output_param.new_acceleration, dtype=np.float64)
                        success = True
                    else:
                        # 对应 Result.Error
                        print(f"[Warn] Ruckig 规划失败 (Result: {res})，跳过本次更新")
                        
                except ruckig.RuckigError as e:
                    # 捕获底层 C++ 抛出的错误
                    # print(f"[Error] Ruckig 发生异常: {e}，跳过本次更新")
                    # print(f"[Error] Ruckig 发生异常，跳过本次更新")
                    pass

                if success:
                    self._current_position = new_pos
                    self._current_velocity = new_vel
                    self._current_acceleration = new_acc
                    
                    self.input_param.current_position = new_pos.tolist()
                    self.input_param.current_velocity = new_vel.tolist()
                    self.input_param.current_acceleration = new_acc.tolist()
                # else:
                    # 如果失败，保持 new_pos 等为上一帧的状态（已在循环开头初始化）

            # ===== Slerp =====
            # 只有在规划成功或者轨迹尚未结束时才更新姿态，避免时间溢出导致插值错误
            if self.slerp is not None and self.slerp_start_time is not None:
                t = time.time() - self.slerp_start_time
                # 如果规划失败，我们依然让姿态随时间推进，直到达到 duration
                # 但如果想严格同步，可以在 success 为 False 时暂停时间推进，这里选择继续推进以保持流畅
                t = np.clip(t, 0.0, self.slerp_duration)

                rot = self.slerp([t])[0]
                self.current_quat = rot.as_quat()

            # ===== 输出（防堆积）=====
            # new_pos 已经在循环开头初始化为当前状态，如果规划成功则已更新
            full_pose = np.concatenate([new_pos, self.current_quat])

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

    runner = RuckigPosePlanner(0.008)

    runner.set_init_pose([0.2, 0.0, 0.1, 0, 0, 0, 1])
    runner.start()

    quat1 = R.from_euler('z', np.pi).as_quat()
    pos1 = [0.2, 0.0, 0.1]
    runner.set_target(np.concatenate([pos1, quat1]))

    time.sleep(1)

    quat2 = R.from_euler('xyz', [np.pi/2, 0, np.pi/2]).as_quat()
    pos2 = [0.15, 0.15, 0.2]
    runner.set_target(np.concatenate([pos2, quat2]))
    
    for i in range(5000):
        pose = runner.get_pose(timeout=0.1)
        if pose is not None:
            print(i, fmt_arr(pose.tolist()))
        time.sleep(0.001)

    runner.stop()