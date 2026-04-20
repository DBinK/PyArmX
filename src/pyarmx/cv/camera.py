# camera.py

import time

import cv2
import numpy as np

# 摄像头参数
camera_params = {
    # 'camera_id': "/dev/video-4k",
    # 'camera_id': 1,
    "camera_id": r"C:\Pictures\Camera Roll\WIN_20260420_23_46_38_Pro.mp4",
    # 'image_width': 1920,
    # 'image_height': 1080,
    "image_width": 1280,
    "image_height": 720,
    "auto_exposure": 0,
    "exposure_time": 1000,
    "fps": 30,
    "gain": 100,
    "auto_wb": 0,
    "wb_temperature": 5000,
    "contrast": 42,
}


class USBCamera:
    def __init__(self, camera_params: dict):
        # 获取相机参数
        self.fps = camera_params.get("fps", 60)
        self.camera_id = camera_params.get("camera_id", 0)
        self.image_width = camera_params.get("image_width", 1280)
        self.image_height = camera_params.get("image_height", 720)
        self.auto_exposure = camera_params.get("auto_exposure", 1)
        self.exposure_time = camera_params.get("exposure_time", 100)
        self.gain = camera_params.get("gain", 0)
        self.auto_wb = camera_params.get("auto_wb", 0)
        self.wb_temperature = camera_params.get("wb_temperature", 5000)
        self.contrast = camera_params.get("contrast", 27)

        # 初始化相机
        print(f"开始初始化 {self.camera_id} 号相机相机...")
        # self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        # self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_FFMPEG)
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区大小为 1, 只读取最新一帧

        print(f"初始化了 {self.camera_id} 号相机, 开始设置参数...")
        self.set_camera_parameters()  # 设置相机参数

        # 注册需要暴露的数据
        self.board_chess_colors = []
        self.center_points = []
        self.black_coords = []
        self.white_coords = []

        # # print('启动USB相机图像捕捉循环')
        # self.cam_thread = threading.Thread(target=self.loop)
        # self.cam_thread.start()

    def set_camera_parameters(self):
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.auto_exposure)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_time)
        # self.cap.set(cv2.CAP_PROP_GAIN, self.gain)
        # self.cap.set(cv2.CAP_PROP_AUTO_WB, self.auto_wb)  # 关闭自动白平衡
        # self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.wb_temperature)  # 设置白平衡色温
        # self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)

        print(f"设置的相机: {self.camera_id} 号相机")
        print(
            f"设置的分辨率: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
        )
        print(f"设置的帧率: {self.cap.get(cv2.CAP_PROP_FPS)}")


if __name__ == "__main__":
    from pyarmx.cv.tags import (
        TagLocator,
        TagVisualizer,
    )
    
    locator = TagLocator()
    vis = TagVisualizer()

    cam = USBCamera(camera_params)
    # cap = cv2.VideoCapture(1,  cv2.CAP_DSHOW)

    while True:
        ret = True
        frame = cv2.imread(r"img\pick\WIN_20260420_23_49_29_Pro.jpg")

        # ret, frame = cam.cap.read()
        # ret, frame = cap.read()

        if ret and frame is not None:
            H_desk2pix, H_pix2desk, detections = locator.locate_target(frame, 14)

            if H_desk2pix is not None:
                frame_rets = vis.draw_tags(frame, detections)
                frame_rets = vis.draw_grid(frame_rets, H_desk2pix, 20)
            else:
                frame_rets = frame

            cv2.imshow("frame_rets", frame_rets)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
