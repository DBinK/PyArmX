import cv2
from dataclasses import dataclass


@dataclass
class CamCfg:
    cam_id: int = 0
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

    api: int | None = cv2.CAP_DSHOW  


class UVCamera:
    def __init__(self, config: CamCfg):
        self.cfg = config
        self.settings = {}
        self.cap = self._init_camera()

    def _init_camera(self):
        if self.cfg.api is None:
            cap = cv2.VideoCapture(self.cfg.cam_id)
        else:
            cap = cv2.VideoCapture(self.cfg.cam_id, self.cfg.api)

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
        ret, frame = self.cap.read()
        return ret, frame

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