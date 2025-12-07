import cv2
import numpy as np
import random

def motion_blur_kernel(length, angle):
    """生成运动模糊核（模拟雨的运动轨迹）"""
    angle_rad = np.deg2rad(angle)
    kernel_size = length
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    # 核中心与线段端点计算
    center = kernel_size // 2
    x_offset = int(np.round(np.cos(angle_rad) * (kernel_size // 2)))
    y_offset = int(np.round(np.sin(angle_rad) * (kernel_size // 2)))
    start = (center + x_offset, center + y_offset)
    end = (center - x_offset, center - y_offset)
    
    # 在核上绘制运动方向的线段
    cv2.line(kernel, start, end, 1.0, thickness=1)
    # 归一化核
    kernel = kernel / np.sum(kernel) if np.sum(kernel) != 0 else kernel
    return kernel

def add_rain(original_image, rain_count_range=(2000, 3000), rain_length_range=(15, 35), 
             rain_width_range=(1, 2), rain_alpha_range=(0.2, 0.6), blur_angle_range=(45, 135), blur_length_range=(10, 20),
             rain_brightness=220, rain_color=(255, 255, 255)):
    """
    给图像添加雨效果（修复雨滴发黑问题）
    :param original_image: 原始图像（BGR格式）
    :param rain_count_range: 雨线数量区间
    :param rain_length_range: 雨线长度区间
    :param rain_width_range: 雨线宽度区间
    :param rain_alpha_range: 雨线透明度区间
    :param blur_angle_range: 雨的倾斜角度区间
    :param blur_length_range: 运动模糊的长度区间
    :param rain_brightness: 雨滴亮度（0-255），越大越亮
    :param rain_color: 雨滴颜色，默认白色
    :return: 加雨效果后的图像
    """
    img = original_image.copy()
    h, w = img.shape[:2]
    
    # 创建白色的雨层（用于加法混合）
    rain_layer = np.zeros_like(img, dtype=np.uint8)
    
    # 生成运动模糊核
    blur_angle = random.uniform(*blur_angle_range)
    blur_length = random.randint(*blur_length_range)
    blur_kernel = motion_blur_kernel(blur_length, blur_angle)
    
    # 随机生成雨线数量
    rain_count = random.randint(*rain_count_range)
    
    for _ in range(rain_count):
        # 随机生成雨线起点（允许从图像上方外部开始）
        x1 = random.randint(0, w)
        y1 = random.randint(-rain_length_range[1], h)
        
        # 随机雨线长度
        length = random.randint(*rain_length_range)
        
        # 计算雨线终点（根据角度）
        theta = np.deg2rad(blur_angle)
        x2 = x1 + int(np.round(length * np.cos(theta)))
        y2 = y1 + int(np.round(length * np.sin(theta)))
        
        # 确保终点在图像范围内
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # 随机雨线宽度
        width = random.randint(*rain_width_range)
        
        # 使用更亮的雨滴颜色
        rain_color_adjusted = (
            min(255, int(rain_color[0] * rain_brightness / 255)),
            min(255, int(rain_color[1] * rain_brightness / 255)),
            min(255, int(rain_color[2] * rain_brightness / 255))
        )
        
        # 绘制雨线
        cv2.line(rain_layer, (x1, y1), (x2, y2), rain_color_adjusted, thickness=width)
    
    # 对雨层应用运动模糊
    rain_layer = cv2.filter2D(rain_layer, -1, blur_kernel)
    
    # 应用高斯模糊（柔化雨线边缘）
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    
    # 使用"加法"混合模式防止雨滴变黑
    rain_alpha = random.uniform(*rain_alpha_range)
    
    # 创建雨滴掩码
    rain_mask = cv2.cvtColor(rain_layer, cv2.COLOR_BGR2GRAY)
    _, rain_mask = cv2.threshold(rain_mask, 10, 255, cv2.THRESH_BINARY)
    rain_mask = rain_mask.astype(np.float32) / 255.0
    
    # 将原图转换为浮点数以便计算
    img_float = img.astype(np.float32)
    rain_layer_float = rain_layer.astype(np.float32)
    
    # 使用屏幕混合（Screen Blend）模式，防止雨滴变黑
    # 屏幕混合公式：1 - (1-a)*(1-b)，结果会更亮
    rainy_img_float = np.zeros_like(img_float)
    for c in range(3):  # 对每个颜色通道
        a = img_float[:, :, c] / 255.0
        b = rain_layer_float[:, :, c] / 255.0 * rain_alpha
        # 屏幕混合公式
        result = 1.0 - (1.0 - a) * (1.0 - b)
        rainy_img_float[:, :, c] = result * 255.0
    
    rainy_img = np.clip(rainy_img_float, 0, 255).astype(np.uint8)
    
    return rainy_img


if __name__ == "__main__":
    # 1. 读取原始图像
    original_path = "datasets/ground_truth.jpg"
    original_img = cv2.imread(original_path)
    if original_img is None:
        print("错误：无法读取图像，请检查路径！")
        exit()
    
    # 2. 添加雨效果 - 针对您的图片优化参数
    rainy_result = add_rain(
        original_image=original_img,                    # 原始图像
        rain_count_range=(1000, 3000),                  # 适当减少雨线数量，避免过度密集
        rain_length_range=(12, 25),                     # 中等长度雨线
        rain_width_range=(1, 2),                        # 细雨线
        rain_alpha_range=(0.2, 0.5),                    # 提高透明度，使雨滴更明显但不发黑
        blur_angle_range=(60, 120),                     # 雨滴倾斜角度
        blur_length_range=(15, 45),                     # 适中的运动模糊
        rain_brightness=240,                            # 提高雨滴亮度
        rain_color=(255, 255, 255)                      # 白色雨滴
    )
    
    # 3. 保存+显示结果
    cv2.imwrite("rainy_result.jpg", rainy_result)
    # cv2.imshow("Rain Effect", rainy_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()