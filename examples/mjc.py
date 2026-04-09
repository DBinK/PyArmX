import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("tmp/mjcf/scene.xml")
data = mujoco.MjData(model)

print(data.site_xpos[model.site("ee").id])

with mujoco.viewer.launch_passive(model, data) as viewer:

    t = 0

    while viewer.is_running():
        
        # print(data.site_xpos[model.site("ee").id])

        t += 0.01

        data.ctrl[5] = np.sin(t)
        data.ctrl[1] = np.sin(t)*0.5
        data.ctrl[2] = np.cos(t)

        mujoco.mj_step(model, data)
        
        viewer.sync()