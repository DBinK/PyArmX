import cv2
from rich import print as rprint

from pyarmx.cv.camera import UVCamera, CamCfg
from pyarmx.cv.planner import VLMPlanner, ModelID
from pyarmx.cv.tags import Point2D, TagLocator, TagVisualizer

from rose.core import Subscriber, Publisher
from rose.message import ChatMsg, RetMsg


from pyarmx.utils.loops import Rate

cam = UVCamera(CamCfg(
    cam_id=r"img\env\nl.mp4",
    # cam_id=1,
    # backend=cv2.CAP_DSHOW,  
))

tag = TagLocator(0.1)
tag_vis = TagVisualizer()

plan = VLMPlanner(ModelID.LMS)
# plan = VLMPlanner(ModelID.ALIYUN)

loop = Rate(30)


status_addr = "tcp://127.0.0.1:5559"
sub = Subscriber(status_addr, "status", ChatMsg)

ret_addr = "tcp://127.0.0.1:5555"
pub = Publisher(ret_addr)

# 相机主循环
while loop.sleep():
    print(f"\r{loop.tick.delta=:.3f}, {loop.tick.on_time}  ", end="")

    ret, img_raw = cam.read_video()
    if not ret:
        continue

    tag_ret  = tag.locate_target(img_raw, 14)
    img_tag = tag_vis.draw_tag_result(img_raw, tag_ret)

    text = sub.recv(False)
    # print(text)
    if text is not None:
        ret_dict_pix = plan.analyze(img_raw, text.content)
        if ret_dict_pix is not None:
            img_plan = plan.draw(img_raw, ret_dict_pix)
            cv2.namedWindow("plan", cv2.WINDOW_NORMAL)
            cv2.imshow("plan", img_plan)

            rprint(ret_dict_pix)

            ret_dict_desk = {"objs": {}, "acts": [[]]}
            for k, v in ret_dict_pix.get("objs", {}).items():
                center = plan.calculate_bbox_center(v)
                ret_dict_desk["objs"][k] = tag.desk_to_pixel(Point2D(center[0], center[1]))

                # img_tag = tag_vis.draw_point(img_tag, Point2D(center[0], center[1]), text=k)

            ret_dict_desk["acts"] = ret_dict_pix.get("acts", {})

            rprint(ret_dict_desk)

            pub.publish("ret", RetMsg(**ret_dict_desk))


    cv2.namedWindow("tag", cv2.WINDOW_NORMAL)
    cv2.imshow("tag", img_tag)

    # cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
    # cv2.imshow("raw", img_raw)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

