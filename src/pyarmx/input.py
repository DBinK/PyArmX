
import keyboard
import numpy as np

class KeyboardController:
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
