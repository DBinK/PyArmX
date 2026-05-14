# PyArmX

PyArmX 是一个面向机械臂控制的 Python 框架，提供仿真与真实硬件的统一抽象接口，支持视觉语言模型（VLM）驱动的智能任务规划。

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/25c10e53-bd82-4d67-b323-029ead934265" />


## 项目架构

<img width="1535" height="1024" alt="image12" src="https://github.com/user-attachments/assets/c43d192f-393c-4beb-b4bc-09fd7c9d7d17" />

```
pyarmx/
├── cv/              # 计算机视觉模块
│   ├── camera.py    # 相机捕获
│   ├── planner.py   # VLM 任务规划器
│   └── tags.py      # AprilTag 定位
├── input/           # 输入控制模块
│   ├── keyboard.py  # 键盘控制
│   ├── zmq_pub.py   # ZMQ 发布
│   └── zmq_sub.py   # ZMQ 订阅
├── utils/           # 工具模块
│   ├── log.py       # 日志
│   ├── loops.py     # 循环调度器
│   └── lowpass.py   # 低通滤波
├── vlm/             # 视觉语言模型
│   ├── chatbot.py   # 对话接口
│   └── prompts.py   # 提示词模板
├── sim_facade.py    # 仿真机械臂抽象
├── real_facade.py   # 真实机械臂抽象
├── ik.py            # 逆运动学求解
├── motion.py        # 运动控制
└── structs.py       # 数据结构
```

## 核心功能

<img width="1536" height="1024" alt="image24" src="https://github.com/user-attachments/assets/7d7dd5d7-4ec8-4f16-9ad4-2d15e7df655b" />

### 1. 双层机械臂抽象

**仿真模式** - 基于 MuJoCo 的高精度物理仿真：
```python
from pyarmx.sim_facade import ArmSimFacade

sim = ArmSimFacade("xml/L801/scene.xml")
sim.set_ee_pose(target_pos, target_quat)
sim.step()
```

**真实模式** - 支持达妙机械臂硬件：
```python
from pyarmx.real_facade import ArmRealFacade

# mock=True 时脱离硬件运行
real = ArmRealFacade(port="COM9", mock=False)
real.set_q_current(joint_angles)
```

### 2. 视觉语言规划

集成多种 VLM 模型进行任务理解和规划：
```python
from pyarmx.cv.planner import VLMPlanner, ModelID

plan = VLMPlanner(ModelID.ALIYUN)  # 或 ModelID.LMS
result = plan.analyze(image, "把黄色方块放到胶带处")
```

### 3. 实时通信

基于 ZMQ 的发布/订阅模式实现进程间通信：
- `pub_node.py` - 发布任务指令
- `sub_node.py` - 订阅状态消息

## 快速开始

### 安装依赖

```bash
pip install -e .
```

### 运行示例

#### 键盘控制机械臂
```bash
python examples/facade/key_ctrl.py
```

#### 仿真到真实映射
```bash
python examples/facade/sim2real.py
```

#### VLM 视觉规划
```bash
python examples/facade/agent.py
```

#### 指令发布器
```bash
python examples/facade/pub_node.py
```

## 典型工作流

1. **启动仿真** → 加载模型并初始化 IK 求解器
2. **视觉感知** → 相机捕获 + Tag 定位
3. **任务规划** → VLM 分析自然语言指令
4. **轨迹执行** → 平滑插值 + 关节控制
5. **硬件同步** → 仿真关节角映射到真实机械臂

## 技术特点

- **统一接口**：仿真与真实硬件使用相同 API
- **实时性**：支持 100Hz 控制频率
- **智能规划**：集成 VLM 实现自然语言指令理解
- **可扩展性**：模块化设计，易于添加新功能

## 许可证

MIT License
