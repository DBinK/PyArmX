import numpy as np

class LowPassFilter:
    def __init__(self, alpha=0.5):
        """
        初始化低通滤波器
        :param alpha: 滤波系数，值越小滤波效果越强，范围通常在0-1之间
        """
        self.alpha = alpha
        self.previous_output = None

    def update(self, input_data):
        """
        更新数据并计算低通滤波结果
        :param input_data: 输入数据 [x, y, z]
        :return: 滤波后的数据 [x, y, z]
        """
        # 如果是第一次输入，则直接输出输入数据
        if self.previous_output is None:
            self.previous_output = np.array(input_data, dtype=float)
            return input_data
        
        # 计算低通滤波结果: output = alpha * input + (1 - alpha) * previous_output
        input_array = np.array(input_data, dtype=float)
        current_output = self.alpha * input_array + (1 - self.alpha) * self.previous_output
        
        # 更新previous_output为当前输出
        self.previous_output = current_output
        
        return current_output.tolist()

    def reset(self):
        """
        重置滤波器状态
        """
        self.previous_output = None


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt 

    # 创建低通滤波器实例，alpha=0.3
    filter = RealTimeLowPassFilter(alpha=0.3)
    
    # 模拟一些带有高频噪声的数据
    test_data = [
        [1.0, 0, 0],
        [1.1, 0, 0],
        [1.2, 0, 0],
        [1.3, 0, 0],
        [8.0, 0, 0],  # 噪声
        [1.3, 0, 0],
        [1.2, 0, 0],
        [1.1, 0, 0],
        [1.0, 0, 0],
        [6.0, 0, 0],  # 噪声
        [1.0, 0, 0],
        [1.0, 0, 0],
    ]
    
    print("低通滤波器测试:")
    print("Alpha系数:", filter.alpha)
    print("-" * 50)

    # 存储输入和输出数据用于绘图
    input_x = []
    output_x = []
    
    # 对每个数据点应用滤波器
    for i, data in enumerate(test_data):
        filtered_data = filter.update(data)
        print(f"输入 {i+1}: {data} -> 输出: {filtered_data}")
        input_x.append(data[0])
        output_x.append(filtered_data[0])
    
    print("-" * 50)

    # 绘制输入和输出数据的对比图
    plt.figure(figsize=(10, 6))
    plt.plot(input_x, label='input_x', marker='o')
    plt.plot(output_x, label='output_x', marker='s')
    plt.xlabel('points')
    plt.ylabel('values')
    plt.title('RealTimeLowPassFilter')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 测试重置功能
    print("测试重置功能:")
    filter.reset()
    new_data = [2.0, 3.0, 4.0]
    filtered_new_data = filter.update(new_data)
    print(f"重置后输入: {new_data} -> 输出: {filtered_new_data}")

    import time
    while True:
        pass
        time.sleep(1)