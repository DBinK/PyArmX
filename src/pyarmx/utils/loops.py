import time
from typing import NamedTuple


def wait_until(target_time: float):
    """
    等待直到 target_time 秒（基于 perf_counter）

    三段策略：
    - 远：sleep(0) 让出CPU
    - 近：busy-wait 保证精度
    """
    while True:
        now = time.perf_counter()
        dt = target_time - now

        if dt <= 0:  # 已超时
            return dt

        if dt > 0.002:
            time.sleep(0)  # 让出CPU
        else:
            pass  # busy-wait


class Time(NamedTuple):
    """计时器状态快照"""

    elapsed: float  # 已用时间
    remaining: float  # 剩余时间
    ok: bool  # 是否未超时


class Tick(NamedTuple):
    """循环节拍快照"""

    elapsed: float  # 总运行时间
    delta: float  # 本轮循环中任务的实际耗时
    ok: bool  # 是否未超时


class Timer:
    def __init__(self, duration: float, auto_start=True):
        self.duration = duration
        self._start: float | None = None
        self._end: float | None = None
        if auto_start:
            self.reset()

    def reset(self):
        self._start = time.perf_counter()
        self._end = self._start + self.duration
        return self

    @property
    def done(self) -> bool:
        """保持属性，方便简单的 if 判断"""
        if self._start is None or self._end is None:
            self.reset()
        assert self._end is not None
        return time.perf_counter() >= self._end

    def step(self) -> Time:
        """获取当前计时状态，永不返回 None"""
        if self._start is None or self._end is None:
            self.reset()

        start, end = self._start, self._end
        assert start is not None and end is not None

        now = time.perf_counter()
        elapsed = now - start
        remaining = max(0.0, end - now)
        is_ok = now < end

        return Time(elapsed=elapsed, remaining=remaining, ok=is_ok)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> Time:
        state = self.step()
        if not state.ok:
            raise StopIteration
        return state


class Rate:
    def __init__(self, hz: float, duration: float | None = None, warn=False):
        self.period = 1.0 / hz
        self.duration = duration
        self.warn = warn
        self.missed = 0
        self._start = self._next = self._end = self._last = None

    def reset(self):
        now = time.perf_counter()
        self._start = self._next = self._last = now
        self._end = (now + self.duration) if self.duration is not None else None
        self.missed = 0
        return self

    def sleep(self) -> Tick:
        if self._start is None:
            self.reset()
        start, next_t, last = self._start, self._next, self._last
        assert start is not None and next_t is not None and last is not None

        now = time.perf_counter()
        dt = now - last

        if now > next_t:
            self.missed += 1
            if self.warn:
                print(f"[Rate] Miss! dt:{dt:.3f}s")
            self._next = now + self.period
        else:
            wait_until(next_t)
            self._next = next_t + self.period

        now_post = time.perf_counter()
        self._last = now_post

        is_ok = True
        if self._end is not None and now_post >= self._end:
            is_ok = False

        return Tick(elapsed=now_post - start, delta=dt, ok=is_ok)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> Tick:
        tick = self.sleep()
        if not tick.ok:
            raise StopIteration
        return tick


if __name__ == "__main__":
    # --- 1. Timer 测试 ---
    print("--- 1. 测试 Timer (for 循环) ---")
    for t_stat in Timer(0.5):
        print(f"Timer For: {t_stat.elapsed=:.6f}, {t_stat.remaining=:.6f}")
        time.sleep(0.1)

    print("\n--- 1. 测试 Timer (while + 海象运算符) ---")
    timer = Timer(0.5)
    # 一行完成：采样、赋值给 t_stat、判断是否未超时
    while (t_stat := timer.step()).ok:
        print(f"Timer Walrus: {t_stat.elapsed=:.6f}, {t_stat.remaining=:.6f}")
        time.sleep(0.1)

    print("\n--- 1. 测试 Timer (while 传统写法) ---")
    timer.reset()
    t_stat = timer.step()  # 循环前手动赋初值
    while t_stat.ok:
        print(f"Timer Classic: {t_stat.elapsed=:.6f}, {t_stat.remaining=:.6f}")
        time.sleep(0.1)
        t_stat = timer.step()  # 循环末尾手动更新

    # --- 2. Rate 测试 (有限时长) ---
    print("\n--- 2. 测试 Rate (for 循环, 10Hz, 0.5s) ---")
    for tick in Rate(10, duration=0.5):
        print(f"Rate For: {tick.elapsed=:.6f}, {tick.delta=:.6f}")

    print("\n--- 2. 测试 Rate (while + 海象运算符, 10Hz) ---")
    rate = Rate(10, duration=0.5)
    while (tick := rate.sleep()).ok:
        print(f"Rate Walrus: {tick.elapsed=:.6f}, {tick.delta=:.6f}")
        time.sleep(0.02)

    print("\n--- 2. 测试 Rate (while 传统写法, 10Hz) ---")
    rate.reset()
    while True:
        tick = rate.sleep()
        if not tick.ok:  # 手动判断状态并跳出
            break
        print(f"Rate Classic: {tick.elapsed=:.6f}, {tick.delta=:.6f}")
        time.sleep(0.02)

    # --- 3. Rate 无限循环测试 ---
    print("\n--- 3. 测试 Rate (无限循环, 演示前 50 次) ---")
    inf_rate = Rate(100)  # 不传 duration, 默认无限循环
    count = 0
    while True:
        tick = inf_rate.sleep()
        # 无限循环下无需判断 tick.ok，除非内部有其他中断逻辑
        print(f"Inf Classic: {tick.elapsed=:.6f}, {tick.delta=:.6f}")
        count += 1
        if count >= 50:
            break
