import os
import random
import cv2
import numpy as np
from collections import defaultdict
import re

# 导入您现有的物理模型函数
from addFog import add_fog
from addRain import add_rain
from addSnow import add_snow

def parse_filename(filename):
    """
    解析文件名以获取ID和摄像头/序列信息。
    适配 Market-1501/DukeMTMC 格式: 0001_c1s1_001051_00.jpg
    返回: (person_id, camera_seq_id)
    """
    # 简单的正则匹配，根据实际数据集文件名格式调整
    # Market-1501: 0002_c1s1_000550_01.jpg -> id=0002, seq=c1s1
    pattern = re.compile(r'([-\d]+)_c(\d+s\d+)_')
    match = pattern.match(filename)
    if match:
        return match.group(1), "c" + match.group(2)
    
    # 如果不匹配标准格式，尝试只取前缀作为ID
    return filename.split('_')[0], "unknown_seq"

def generate_consistent_weather_dataset(input_dir, output_dir, mode='fog'):
    """
    生成具有序列一致性的天气ReID数据集
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取并分组图片
    # 我们需要将属于同一个行人+同一个摄像头序列的图片分为一组 (Tracklet)
    # 以确保在这个片段内天气参数一致
    img_groups = defaultdict(list)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"找到 {len(files)} 张图片，正在分组...")

    for fname in files:
        pid, seq_id = parse_filename(fname)
        # key = (pid, seq_id) # 按轨迹分组 (更严格，推荐)
        key = pid           # 或者按 PID 分组 (所有该行人的图片天气都一样)
        img_groups[key].append(fname)

    print(f"共分为 {len(img_groups)} 个组 (基于PID/Sequence). 开始处理...")

    count = 0
    total = len(files)

    # 2. 遍历每一组，生成固定的天气参数，并应用到组内所有图片
    for group_key, group_files in img_groups.items():
        
        # --- 在这里随机生成该组的“天气配置” ---
        
        # 通用参数
        brightness_shift = random.uniform(0.6, 0.9) # 整体亮度倾向
        
        # 雾参数
        fog_beta = random.uniform(0.01, 0.08)
        fog_brightness = random.uniform(0.6, 0.8)

        # 雨参数
        rain_count = random.randint(200, 1000) # 雨量
        rain_angle = random.uniform(70, 110)   # 风向/角度
        rain_len = random.randint(10, 25)
        
        # 雪参数
        snow_count = random.randint(200, 1000)
        snow_alpha = random.uniform(0.2, 0.4)
        # 简化的风速向量，为了保持addSnow的一致性，这里稍微复杂点
        wind_x = random.randint(1, 3)
        wind_y = random.randint(1, 4)
        
        # 3. 处理组内每一张图片
        for fname in group_files:
            src_path = os.path.join(input_dir, fname)
            img = cv2.imread(src_path)
            if img is None:
                continue

            save_name = fname # 保持原文件名
            
            # 根据模式应用天气
            # 关键点：将随机范围设置为 (value, value)，强制函数使用我们锁定的参数
            
            if mode == 'fog':
                # 注意：addFog 内部使用 -np.random.uniform，所以不需要传负数，传正数范围即可
                out_img, _ = add_fog(
                    img, 
                    beta_range=(fog_beta, fog_beta),           # 锁定浓度
                    brightness_range=(fog_brightness, fog_brightness) # 锁定大气光
                )

            elif mode == 'rain':
                # 雨的位置在add_rain内部是随机的(good)，但雨的密度和角度被我们锁定了(good)
                out_img, _ = add_rain(
                    img,
                    rain_count_range=(rain_count, rain_count), # 锁定雨量
                    blur_angle_range=(rain_angle, rain_angle), # 锁定角度
                    rain_length_range=(rain_len, rain_len),    # 锁定雨线长度
                    # 其他参数也可以类似锁定，或者留给函数内部微调
                )

            elif mode == 'snow':
                out_img, _ = add_snow(
                    img,
                    snow_count=(snow_count, snow_count),       # 锁定雪量
                    alpha=(snow_alpha, snow_alpha),            # 锁定透明度
                    wind_speed=((wind_x, wind_x), (wind_y, wind_y)), # 锁定风速方向
                    # 其他参数...
                )
            
            else:
                out_img = img

            # 保存
            dst_path = os.path.join(output_dir, save_name)
            cv2.imwrite(dst_path, out_img)
            
            count += 1
            if count % 1000 == 0:
                print(f"已处理 {count}/{total} 张图片")

    print(f"完成！生成数据集保存在: {output_dir}")

if __name__ == "__main__":
    # 示例用法
    # 请修改为您实际的ReID数据集路径，例如 Market-1501 的 bounding_box_test
    INPUT_Data_Dir = r"datasets\DefogDataset\test\ground_truth" # 这里仅为示例，请替换为您的 ReID 数据集路径
    OUTPUT_Data_Dir = r"datasets\ReID_Fog_Test"
    
    # 模式可选: 'fog', 'rain', 'snow'
    generate_consistent_weather_dataset(INPUT_Data_Dir, OUTPUT_Data_Dir, mode='fog')
