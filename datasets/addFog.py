import cv2
import numpy as np
import math
import os

def add_fog(original_image, beta_range=(0.01, 0.08), brightness_range=(0.6, 0.8), use_depth_map=False, depth_map=None, return_mask=False):
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
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    fogged_img = img_f * t + brightness * (1 - t)
    fogged_img = np.clip(fogged_img * 255, 0, 255).astype(np.uint8)

    # 量化指标
    beta_val = float(-beta)
    brightness_val = float(brightness)

    # 归一化（区间可根据实际数据调整）
    beta_norm = (beta_val - 0.01) / (0.04 - 0.01)  # beta_range
    brightness_norm = (brightness_val - 0.6) / (0.8 - 0.6)  # brightness_range
    # 越大雾越重：beta↑, brightness↓
    brightness_score = 1 - np.clip(brightness_norm, 0, 1)
    beta_score = np.clip(beta_norm, 0, 1)
    # 综合分数
    fog_score = float(beta_score * 0.9 + brightness_score * 0.1)

    if return_mask:
        # 返回雾化区域掩码，全1
        mask = np.ones((row, col), dtype=np.uint8) * 255
        return fogged_img, fog_score, mask
    else:
        return fogged_img, fog_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=r'datasets/DefogDataset/train/ground_truth', help='输入图片文件夹')
    parser.add_argument('--output_dir', type=str, default=r'datasets/DefogDataset/train/foggy_image', help='输出图片文件夹')
    parser.add_argument('--num', type=int, default=5, help='每张图片生成的加雾效果数量')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for fname in os.listdir(args.input_dir):
        if fname.lower().endswith('.jpg'):
            img_path = os.path.join(args.input_dir, fname)
            img = cv2.imread(img_path)
            for i in range(1, args.num + 1):
                fog_img = add_fog(
                    original_image=img,
                    beta_range=(0.01, 0.04),
                    brightness_range=(0.6, 0.8)
                )
                out_name = f"{os.path.splitext(fname)[0]}_{i}{os.path.splitext(fname)[1]}"
                out_path = os.path.join(args.output_dir, out_name)
                cv2.imwrite(out_path, fog_img)
                print(f"已保存: {out_path}")


    # # 测试分数代码
    # input_dir = r'datasets/DefogDataset/train/ground_truth'
    # output_dir = r'datasets/DefogDataset/foggy_image'
    # num = 5
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # for fname in os.listdir(input_dir):
    #     if fname.lower().endswith('.jpg'):
    #         img_path = os.path.join(input_dir, fname)
    #         img = cv2.imread(img_path)
    #         for i in range(1, num + 1):
    #             fog_img, fog_score = add_fog(
    #                 original_image=img,
    #                 beta_range=(0.01, 0.04),
    #                 brightness_range=(0.6, 0.8)
    #             )
    #             out_path = os.path.join(output_dir, f"{fog_score:.4f}_" + fname)
    #             cv2.imwrite(out_path, fog_img)

    # # 测试带掩码的加雾
    # input_dir = r'datasets/DefogDataset/train/ground_truth'
    # output_dir = r'datasets/DefogDataset/train/fog_mask'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # for fname in os.listdir(input_dir):
    #     if fname.lower().endswith('.jpg'):
    #         img_path = os.path.join(input_dir, fname)
    #         img = cv2.imread(img_path)
    #         fog_img, fog_score, fog_mask = add_fog(
    #             original_image=img,
    #             beta_range=(0.01, 0.04),
    #             brightness_range=(0.6, 0.8),
    #             return_mask=True
    #         )
    #         out_path = os.path.join(output_dir, f"{fog_score:.4f}_" + fname)
    #         cv2.imwrite(out_path, fog_mask)