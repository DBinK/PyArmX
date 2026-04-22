
import cv2

from pyarmx.cv.camera import UVCamera, CamCfg
from pyarmx.cv.tags import (
    Point2D,
    TagLocator,
    TagVisualizer,
)

if __name__ == "__main__":
    
    locator = TagLocator()
    vis = TagVisualizer()

    cam = UVCamera(CamCfg(cam_id=1))
    # cap = cv2.VideoCapture(1,  cv2.CAP_DSHOW)

    while True:
        # ret = True
        # frame = cv2.imread(r"img\pick\WIN_20260420_23_49_29_Pro.jpg")

        ret, frame = cam.cap.read()
        # ret, frame = cap.read()

        if ret and frame is not None:
            H_desk2pix, H_pix2desk, detections = locator.locate_target(frame, 14)

            if H_desk2pix is not None:
                frame_rets = vis.draw_tags(frame, detections)
                frame_rets = vis.draw_grid(frame_rets, H_desk2pix, 20)
            else:
                frame_rets = frame

            # pix = locator.desk_to_pixel(Point2D(300, -50))
            # frame_rets = vis.draw_point(frame_rets, pix, text="nailong") # type: ignore

            cv2.namedWindow("frame_rets", cv2.WINDOW_NORMAL)
            cv2.imshow("frame_rets", frame_rets)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
