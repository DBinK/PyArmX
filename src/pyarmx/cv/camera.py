import cv2
from dataclasses import dataclass


@dataclass
class CamCfg:
    cam_id: int | str = 0  # 可以用视频文件路径
    width: int = 1280
    height: int = 720
    fps: int = 30

    fourcc: str = "MJPG"
    buffer_size: int = 1

    # 可选参数（None 表示跳过）
    auto_exposure: int | None = None
    exposure: int | None = None
    gain: int | None = None
    auto_wb: int | None = None
    wb_temperature: int | None = None
    contrast: int | None = None

    backend: int | None = None  # Windows 推荐用 cv2.CAP_DSHOW  


class UVCamera:
    def __init__(self, config: CamCfg = CamCfg()):
        self.cfg = config
        self.settings = {}
        self.cap = self._init_camera()

    def _init_camera(self):
        if self.cfg.backend is None:
            cap = cv2.VideoCapture(self.cfg.cam_id)
        else:
            cap = cv2.VideoCapture(self.cfg.cam_id, self.cfg.backend)

        # FOURCC 单独处理
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self.cfg.fourcc))

        # 参数映射表
        self.settings = {
            cv2.CAP_PROP_FRAME_WIDTH: self.cfg.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.cfg.height,
            cv2.CAP_PROP_FPS: self.cfg.fps,
            cv2.CAP_PROP_BUFFERSIZE: self.cfg.buffer_size,

            # 扩展参数
            cv2.CAP_PROP_AUTO_EXPOSURE: self.cfg.auto_exposure,
            cv2.CAP_PROP_EXPOSURE: self.cfg.exposure,
            cv2.CAP_PROP_GAIN: self.cfg.gain,
            cv2.CAP_PROP_AUTO_WB: self.cfg.auto_wb,
            cv2.CAP_PROP_WB_TEMPERATURE: self.cfg.wb_temperature,
            cv2.CAP_PROP_CONTRAST: self.cfg.contrast,
        }

        # 统一设置（跳过 None）
        for prop, value in self.settings.items():
            if value is not None:
                cap.set(prop, value)

        return cap

    def read(self):
        """读取一帧"""
        return self.cap.read()
    
    def read_video(self):
        """读取一帧（视频文件）"""
        ret, frame = self.cap.read()
        # 如果是视频文件且读取失败（通常意味着到达文件末尾），则重置并重新读取
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def shot(self, path):
        ret, frame = self.read()
        if ret:
            cv2.imwrite(path, frame)
            return True
        else:
            return False

    def release(self):
        self.cap.release()

    def info(self):
        from pprint import pprint
        pprint(self.cfg)
        for prop in self.settings.keys():
            print(f"{prop}: {self.cap.get(prop)}")


if __name__ == "__main__":
    cam = UVCamera(CamCfg())
    cam.info()
    cam.release()