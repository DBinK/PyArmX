
import keyboard
import numpy as np
from scipy.spatial.transform import Rotation as R


class PlaybackInput:
    def __init__(self):
        self.pause = False
        self._key_states = {}
    
    def _check_key_with_debounce(self, key_name):
        """通用消抖检查：只在按键刚按下时返回 True"""
        key_pressed = keyboard.is_pressed(key_name)
        last_state = self._key_states.get(key_name, False)
        
        result = key_pressed and not last_state
        
        self._key_states[key_name] = key_pressed
        return result
    
    def is_pause(self):
        """键盘输入 -> 暂停（带消抖）"""
        if self._check_key_with_debounce("\\"):
            self.pause = not self.pause
        return self.pause

    def is_switch(self):
        """键盘输入 -> 切换模式（带消抖）"""
        return self._check_key_with_debounce("enter")
    

class PoseInput:
    def __init__(self, target_speed=0.15, rot_speed=1.0):
        self.target_speed = target_speed
        self.rot_speed = rot_speed

    def update(self, target_pos, target_quat, dt):
        """键盘输入 -> 新的 target"""

        move_dir = np.zeros(3)
        speed_scale = 4.0 if keyboard.is_pressed("space") else 1.0

        # --- 平移 ---
        if keyboard.is_pressed("up"): move_dir[1] += 1.0
        if keyboard.is_pressed("down"): move_dir[1] -= 1.0
        if keyboard.is_pressed("left"): move_dir[0] -= 1.0
        if keyboard.is_pressed("right"): move_dir[0] += 1.0
        if keyboard.is_pressed("alt"): move_dir[2] += 1.0
        if keyboard.is_pressed("ctrl"): move_dir[2] -= 1.0

        if np.linalg.norm(move_dir) > 1e-12:
            target_pos = target_pos + (move_dir / np.linalg.norm(move_dir)) * self.target_speed * speed_scale * dt

        # --- 旋转 ---
        rot_vec = np.zeros(3)
        if keyboard.is_pressed("-"): rot_vec[0] -= 1.0
        if keyboard.is_pressed("="): rot_vec[0] += 1.0
        if keyboard.is_pressed("["): rot_vec[1] -= 1.0
        if keyboard.is_pressed("]"): rot_vec[1] += 1.0
        if keyboard.is_pressed(";"): rot_vec[2] -= 1.0
        if keyboard.is_pressed("'"): rot_vec[2] += 1.0

        if np.linalg.norm(rot_vec) > 1e-12:
            delta_ang = rot_vec * self.rot_speed * speed_scale * dt
            delta_R = R.from_rotvec(delta_ang)
            current_R = R.from_quat(target_quat)
            new_R = delta_R * current_R
            target_quat = new_R.as_quat()
            target_quat /= np.linalg.norm(target_quat)

        return target_pos, target_quat


class JointInput:
    def __init__(self, joint_speed=0.5):
        self.joint_speed = joint_speed
        
        # 映射策略：每个关节对应两个键，一个正向，一个负向
        # 格式: "键名": (关节索引, 方向系数)
        # 方向系数: 1.0 为正向, -1.0 为负向
        self.key_map = {
            # 关节 0
            "-": (0, -1.0),
            "=": (0, 1.0),
            
            # 关节 1
            "[": (1, -1.0),
            "]": (1, 1.0),
            
            # 关节 2
            ";": (2, -1.0),
            "'": (2, 1.0),
            
            # 关节 3
            ",": (3, -1.0),
            ".": (3, 1.0),
            
            # 关节 4
            "home": (4, -1.0),
            "page up": (4, 1.0),
            
            # 关节 5
            "end": (5, -1.0),
            "page down": (5, 1.0),
        }
        
        # 定义基础步长 (弧度/秒)
        self.base_speed_rad = 30

    def update(self, current_q, dt):
        """
        键盘输入 -> 新的关节角 target
        :param current_q: 当前关节角数组
        :param dt: 时间步长
        :return: 新的目标关节角数组
        """
        q_target = np.copy(current_q)
        
        # 加速逻辑
        speed_scale = 4.0 if keyboard.is_pressed("space") else 1.0
        
        # 取反逻辑: 如果按住 Ctrl, 所有方向反转
        direction_scale = -1.0 if keyboard.is_pressed("ctrl") else 1.0

        for key, (joint_idx, base_direction) in self.key_map.items():
            try:
                if keyboard.is_pressed(key):
                    if joint_idx < len(q_target):
                        # 最终方向 = 按键固有方向 * Ctrl取反 * 速度倍率
                        final_direction = base_direction * direction_scale
                        
                        # 计算增量: 角速度 * 方向 * 时间
                        delta = self.base_speed_rad * final_direction * speed_scale * dt
                        q_target[joint_idx] += delta
            except ValueError:
                continue
                    
        return q_target