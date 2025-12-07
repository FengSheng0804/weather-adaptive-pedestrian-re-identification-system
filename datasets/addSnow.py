import cv2
import numpy as np
import random
import math

def create_motion_blur_kernel(size=15, angle=0):
    """创建任意角度的运动模糊核"""
    kernel = np.zeros((size, size))
    center = size // 2
    
    # 根据角度计算线段端点
    angle_rad = math.radians(angle)
    length = size // 2
    
    # 计算线段起点和终点
    start_x = center - int(length * math.cos(angle_rad))
    start_y = center - int(length * math.sin(angle_rad))
    end_x = center + int(length * math.cos(angle_rad))
    end_y = center + int(length * math.sin(angle_rad))
    
    # 在核上画线
    cv2.line(kernel, (start_x, start_y), (end_x, end_y), 1, thickness=1)
    
    # 归一化
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel = kernel / kernel_sum
    
    return kernel

def add_snow_to_image(
    original_image,
    snow_count=(800, 2500),
    snow_size_range=((1, 2), (3, 4)),
    small_radio=(0.75, 0.95),
    alpha=(0.2, 0.3),
    wind_speed=((1, 5), (1, 2)),
    blur_angle_variance=30,
    snow_brightness=255,
    snow_intensity=0.7
):
    """添加雪花效果的主要函数"""
    
    h, w = original_image.shape[:2]
    # 创建雪花蒙版（单通道）
    snow_mask = np.zeros((h, w), dtype=np.float32)
    
    # 计算主要风向角度
    dx, dy = random.randint(*wind_speed[0]), random.randint(*wind_speed[1])
    dx = dx if random.random() < 0.5 else -dx  # 随机决定风向
    main_angle = math.degrees(math.atan2(dy, dx))
    
    snow_count_value = random.randint(*snow_count)
    small_snow_count = int(snow_count_value * random.uniform(*small_radio))
    large_snow_count = snow_count_value - small_snow_count

    def add_snow_group(count, size_range, opacity_factor, base_kernel_min, base_kernel_max, kernel_min, kernel_max, temp_size_factor):
        for _ in range(count):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            size = random.randint(*size_range)
            opacity = opacity_factor * size * (snow_brightness / 255.0)
            
            # 为每个雪花生成略微不同的模糊角度
            angle_variation = random.uniform(-blur_angle_variance, blur_angle_variance)
            current_angle = main_angle + angle_variation
            
            # 动态计算模糊核大小（基于雪花大小和风速）
            base_kernel_size = max(base_kernel_min, min(base_kernel_max, size * 2 + int(math.sqrt(dx*dx + dy*dy))))
            kernel_size = random.randint(
                max(kernel_min, base_kernel_size - 2), 
                min(kernel_max, base_kernel_size + 2)
            )
            
            # 创建运动模糊核
            kernel = create_motion_blur_kernel(kernel_size, current_angle)
            
            # 创建雪花形状
            temp_size = max(size * temp_size_factor, kernel_size * temp_size_factor)  # 增大临时画布
            temp_mask = np.zeros((temp_size, temp_size), dtype=np.float32)
            center = (temp_size // 2, temp_size // 2)
            
            # 多种雪花形状
            shape_type = random.choice(["ellipse", "circle", "star"])
            
            if shape_type == "ellipse":
                axes = (size, max(1, size // 2))
                angle = random.randint(0, 360)
                cv2.ellipse(temp_mask, center, axes, angle, 0, 360, opacity, -1)
            elif shape_type == "circle":
                cv2.circle(temp_mask, center, size, opacity, -1)
            else:  # star
                # 简单的星形
                points = []
                for i in range(5):
                    angle_pt = math.radians(i * 72)
                    x_pt = center[0] + int(size * math.cos(angle_pt))
                    y_pt = center[1] + int(size * math.sin(angle_pt))
                    points.append((x_pt, y_pt))
                cv2.fillConvexPoly(temp_mask, np.array(points), opacity)

            
            # 应用运动模糊
            blurred_snow = cv2.filter2D(temp_mask, -1, kernel)
            
            # 将雪花添加到雪花蒙版
            x_start = max(0, x - temp_size//2)
            x_end = min(w, x + temp_size//2)
            y_start = max(0, y - temp_size//2)
            y_end = min(h, y + temp_size//2)
            
            if x_end > x_start and y_end > y_start:
                patch_width = x_end - x_start
                patch_height = y_end - y_start
                
                temp_x_start = max(0, temp_size//2 - (x - x_start))
                temp_y_start = max(0, temp_size//2 - (y - y_start))
                temp_x_end = temp_x_start + patch_width
                temp_y_end = temp_y_start + patch_height
                
                if (temp_x_end <= temp_size and temp_y_end <= temp_size and
                    temp_x_start < temp_x_end and temp_y_start < temp_y_end):
                    
                    snow_patch = blurred_snow[temp_y_start:temp_y_end, temp_x_start:temp_x_end]
                    
                    # 使用最大值混合，避免重叠部分过度积累
                    existing_patch = snow_mask[y_start:y_end, x_start:x_start+patch_width]
                    combined = np.maximum(existing_patch, snow_patch)
                    snow_mask[y_start:y_end, x_start:x_start+patch_width] = combined

    # 合并小雪花和大雪花的生成逻辑
    add_snow_group(small_snow_count, snow_size_range[0], random.uniform(*alpha), 5, 20, 3, 25, 4)
    add_snow_group(large_snow_count, snow_size_range[1], random.uniform(*alpha), 7, 25, 5, 30, 5)


    # 将雪花蒙版限制在合理范围内
    snow_mask = np.clip(snow_mask, 0, 1)
    
    # 创建白色雪花图层
    white_snow = np.ones((h, w, 3), dtype=np.float32) * (snow_brightness / 255.0)
    
    # 应用雪花蒙版
    snow_alpha = snow_mask[..., np.newaxis]  # 增加维度以便广播
    
    # 使用屏幕混合模式：1 - (1-a)*(1-b) 保持雪花亮度
    original_float = original_image.astype(np.float32) / 255.0
    
    # 计算屏幕混合
    screen_blend = 1 - (1 - original_float) * (1 - white_snow * snow_alpha)
    
    # 可选：添加一些发光效果增强雪花
    glow_strength = 0.3
    glow = white_snow * snow_alpha * glow_strength
    snowy_img = screen_blend + glow
    
    # 限制在有效范围内
    snowy_img = (np.clip(snowy_img, 0, 1) * 255.0).astype(np.uint8)
    
    # 调整雪花强度
    snow_mask = np.any(snowy_img > original_image, axis=2, keepdims=True)
    result = original_image.copy().astype(np.float32)
    snowy_float = snowy_img.astype(np.float32)
    
    # 线性混合
    result = result * (1 - snow_intensity * snow_mask) + snowy_float * snow_intensity * snow_mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    # 1. 读取原始图像
    original_path = "ground_truth.jpg"  # 替换为你的原图路径
    original_img = cv2.imread(original_path)
    if original_img is None:
        print("错误：无法读取图像，请检查路径！")
        exit()
    
    # 2. 添加雪花效果
    snowy_result = add_snow_to_image(
        original_image=original_img,
        snow_count=(200, 1500),  # 增加雪花数量
        snow_size_range=((1, 2), (2, 3)),  # 小雪花和大雪花的尺寸
        small_radio=(0.75, 0.95),  # 增加小雪花比例
        alpha=(0.2, 0.3),
        wind_speed=((1, 5), (1, 2)),  # 增加风速
        blur_angle_variance=20,
        snow_intensity=0.8  # 增加雪花强度
    )
    
    # 3. 保存+显示结果
    cv2.imwrite("snowy_result.jpg", snowy_result)
    # cv2.imshow("Snow Effect", snowy_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()