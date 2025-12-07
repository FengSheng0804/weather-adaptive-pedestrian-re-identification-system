import cv2
import numpy as np
import math

def add_fog(original_image, beta_range=(0.01, 0.08), brightness_range=(0.6, 0.8), use_depth_map=False, depth_map=None):
    """
    基于大气散射模型的图像加雾算法
    
    参数:
        original_image: 原始图像 (numpy数组)
        beta: 雾的浓度系数，值越大雾越浓 (默认(0.01, 0.08)，范围0.01-0.08)
        brightness: 大气光值，控制雾的亮度 (默认(0.6, 0.8)，范围0-1)
        use_depth_map: 是否使用深度图 (默认False)
        depth_map: 深度图，单通道 numpy数组
    
    返回:
        加雾后的图像
    """
    # 将图像转换为浮点数并归一化
    img_f = original_image.astype(np.float32) / 255.0
    
    # 获取图像尺寸
    if len(original_image.shape) == 3:
        row, col, chs = original_image.shape
    else:
        row, col = original_image.shape
        chs = 1
        img_f = img_f[:, :, np.newaxis]
    
    if use_depth_map and depth_map is not None:
        # 使用提供的深度图
        d = depth_map.astype(np.float32)
        # 归一化深度图
        d_normalized = (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-5)
        # 计算透射率
        beta = -np.random.uniform(beta_range[0], beta_range[1])
        t = np.exp(beta * d_normalized)
    else:
        # 生成模拟深度图（基于距离图像中心的距离）
        size = math.sqrt(max(row, col))  # 雾化尺寸
        center = (row // 2, col // 2)    # 雾化中心
        
        # 使用矢量化操作提高效率
        y, x = np.ogrid[:row, :col]
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        d = -0.04 * dist + size
        
        # 计算透射率
        beta = -np.random.uniform(beta_range[0], beta_range[1])
        t = np.exp(beta * d)
    
    # 扩展透射率到三通道（如果是彩色图像）
    if chs == 3:
        t = t[:, :, np.newaxis]
    
    # 大气散射模型: I = J * t + A * (1 - t)
    # 其中 I 是加雾后的图像，J 是原始图像，t 是透射率，A 是大气光值
    fogged_img = img_f * t + np.random.uniform(brightness_range[0], brightness_range[1]) * (1 - t)
    
    # 确保值在有效范围内并转换回uint8
    fogged_img = np.clip(fogged_img * 255, 0, 255).astype(np.uint8)
    
    return fogged_img

if __name__ == "__main__":
    # 读取原始图像
    original_path = "ground_truth.png"  # 请替换为您的图像路径
    original_img = cv2.imread(original_path)
    
    if original_img is None:
        print("错误：无法读取图像，请检查路径！")
        exit()
    
    fog_result = add_fog(
        original_image=original_img,
        beta_range=(0, 0.08),
        brightness_range=(0.6, 0.8)
    )
    
    # 保存结果
    cv2.imwrite("fog_result.jpg", fog_result)
    
    cv2.imshow("Foggy_image", fog_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()