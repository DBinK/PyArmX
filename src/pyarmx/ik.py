from typing import Callable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


class IKSolver:
    def __init__(
        self,
        fk_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        jac_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        arm_dof: int,
        q_min: np.ndarray,
        q_max: np.ndarray,
        max_iters: int = 8,
        pos_weight: float = 1.0,
        rot_weight: float = 0.3,
        step_max: float = 0.08,
        pos_tol: float = 1e-4,
        rot_tol: float = 2e-3,
    ):
        self.fk_func = fk_func
        self.jac_func = jac_func
        self.arm_dof = arm_dof
        self.q_min = q_min
        self.q_max = q_max
        self.max_iters = max_iters
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.step_max = step_max
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol

    @staticmethod
    def _rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
        R_err = R_target @ R_current.T
        return 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])

    @staticmethod
    def _clamp_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
        n = np.linalg.norm(x)
        if n > max_norm and n > 1e-12:
            return x / n * max_norm
        return x

    @staticmethod
    def _adaptive_damping(J: np.ndarray, lam_min=1e-4, lam_max=5e-2, sigma_ref=0.05) -> float:
        s = np.linalg.svd(J, compute_uv=False)
        sigma_min = s[-1] if len(s) > 0 else 0.0
        ratio = np.clip(sigma_min / sigma_ref, 0.0, 1.0)
        return lam_max * (1.0 - ratio) ** 2 + lam_min

    def solve(self, q_init: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        target_rot = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()
        q = q_init.copy()

        for _ in range(self.max_iters):
            current_pos, current_rot = self.fk_func(q)
            pos_err = target_pos - current_pos
            rot_err = self._rotation_error(current_rot, target_rot)

            if np.linalg.norm(pos_err) < self.pos_tol and np.linalg.norm(rot_err) < self.rot_tol:
                break

            jacp_full, jacr_full = self.jac_func(q)
            J_p = jacp_full[:, :self.arm_dof]
            J_r = jacr_full[:, :self.arm_dof]

            J = np.vstack([
                self.pos_weight * J_p,
                self.rot_weight * J_r
            ])

            lam = self._adaptive_damping(J)
            H = J.T @ J + lam * np.eye(self.arm_dof)
            err = np.concatenate([self.pos_weight * pos_err, self.rot_weight * rot_err])
            g = J.T @ err

            try:
                dq = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(H) @ g

            dq = self._clamp_norm(dq, self.step_max)
            q = np.clip(q + dq, self.q_min, self.q_max)

        return q
