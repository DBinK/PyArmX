
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
        ret = True
        frame = cv2.imread(r"img\pick\WIN_20260420_23_49_29_Pro.jpg")

        # ret, frame = cam.cap.read()
        # ret, frame = cap.read()

        ret, img_raw = cam.read_video()
        if not ret:
            continue

        tag_ret  = locator.locate_target(img_raw, 14)
        img_tag = vis.draw_tag_result(img_raw, tag_ret)

        cv2.namedWindow("tag", cv2.WINDOW_NORMAL)
        cv2.imshow("tag", img_tag)

        # cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
        # cv2.imshow("raw", img_raw)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        else:
            break
