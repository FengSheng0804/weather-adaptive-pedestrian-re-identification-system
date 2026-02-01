import os
import random
from addFog import add_fog
from addRain import add_rain
from addSnow import add_snow
import cv2

# 处理fog场景
def process_fog(single=2000, double=500, triple=300, mode='train'):

    ground_truth_dir = rf"datasets\DefogDataset\{mode}\ground_truth"
    dst_base_dir = rf"datasets\MoEDataset\{mode}"

    if mode == 'train':
        fog_dir = os.path.join(dst_base_dir, "1", "fog")
        fog_ground_truth_dir = os.path.join(dst_base_dir, "1", "fog_ground_truth")
        fog_mask_dir = os.path.join(dst_base_dir, "1", "fog_mask")
        fog_rain_dir = os.path.join(dst_base_dir, "2", "fog_rain")
        fog_rain_ground_truth_dir = os.path.join(dst_base_dir, "2", "fog_rain_ground_truth")
        fog_rain_mask_dir = os.path.join(dst_base_dir, "2", "fog_rain_mask")
        fog_snow_dir = os.path.join(dst_base_dir, "2", "fog_snow")
        fog_snow_ground_truth_dir = os.path.join(dst_base_dir, "2", "fog_snow_ground_truth")
        fog_snow_mask_dir = os.path.join(dst_base_dir, "2", "fog_snow_mask")
        fog_rain_snow_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow")
        fog_rain_snow_ground_truth_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow_ground_truth")
        fog_rain_snow_mask_dir = os.path.join(dst_base_dir, '3', "fog_rain_snow_mask")

        os.makedirs(fog_dir, exist_ok=True)
        os.makedirs(fog_ground_truth_dir, exist_ok=True)
        os.makedirs(fog_mask_dir, exist_ok=True)
        os.makedirs(fog_rain_dir, exist_ok=True)
        os.makedirs(fog_rain_ground_truth_dir, exist_ok=True)
        os.makedirs(fog_rain_mask_dir, exist_ok=True)
        os.makedirs(fog_snow_dir, exist_ok=True)
        os.makedirs(fog_snow_ground_truth_dir, exist_ok=True)
        os.makedirs(fog_snow_mask_dir, exist_ok=True)
        os.makedirs(fog_rain_snow_dir, exist_ok=True)
        os.makedirs(fog_rain_snow_ground_truth_dir, exist_ok=True)
        os.makedirs(fog_rain_snow_mask_dir, exist_ok=True)
    
    elif mode == 'test':
        fog_dir = os.path.join(dst_base_dir, "weather_image")
        fog_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        fog_rain_dir = os.path.join(dst_base_dir, "weather_image")
        fog_rain_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        fog_snow_dir = os.path.join(dst_base_dir, "weather_image")
        fog_snow_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        fog_rain_snow_dir = os.path.join(dst_base_dir, "weather_image")
        fog_rain_snow_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        
        os.makedirs(fog_dir, exist_ok=True)
        os.makedirs(fog_ground_truth_dir, exist_ok=True)

    gt_images = [f for f in os.listdir(ground_truth_dir) if os.path.isfile(os.path.join(ground_truth_dir, f)) and f.lower().endswith((".png", ".jpg"))]
    random.shuffle(gt_images)
    gt_num = len(gt_images)

    # fog单一场景
    fog_scores = []
    # 查找以.jpg结尾的文件数量，作为起始索引
    start_index = len([f for f in os.listdir(fog_dir) if f.lower().endswith(".jpg")])
    for idx in range(single):
        img_name = gt_images[idx % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        fog_img, fog_score, fog_mask = add_fog(
            original_image=img,
            beta_range=(0.01, 0.08),
            brightness_range=(0.6, 0.8),
            return_mask=True
        )
        fog_scores.append(f"fog:{fog_score:.2f}")

        dst_path = os.path.join(fog_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(fog_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, fog_img)
        cv2.imwrite(gt_dst_path, img)
        
        # 如果是train模式，保存掩码
        if mode == 'train':
            fog_mask_path = os.path.join(fog_mask_dir, f"{idx+1+start_index}_fog.png")
            cv2.imwrite(fog_mask_path, fog_mask)
        
        print(f"saved: {dst_path}, gt: {gt_dst_path}")
    
    # 将分数保存到文件
    with open(os.path.join(fog_dir, "scores.txt"), "a") as f:
        for score in fog_scores:
            f.write(score + "\n")

    # fog_rain场景
    fog_rain_scores = []
    start_index = len([f for f in os.listdir(fog_rain_dir) if f.lower().endswith(".jpg")])
    for idx in range(double):
        img_name = gt_images[(single + idx) % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        fog_img, fog_score, fog_mask = add_fog(
            original_image=img,
            beta_range=(0.01, 0.08),
            brightness_range=(0.6, 0.8),
            return_mask=True
        )
        rain_img, rain_score, rain_mask = add_rain(
            original_image=fog_img,
            rain_count_range=(20, 600),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.5),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255),
            return_mask=True
        )
        fog_rain_scores.append(f"fog:{fog_score:.2f},rain:{rain_score:.2f}")

        dst_path = os.path.join(fog_rain_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(fog_rain_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, rain_img)
        cv2.imwrite(gt_dst_path, img)

        # 如果是train模式，保存掩码
        if mode == 'train':
            fog_mask_path = os.path.join(fog_rain_mask_dir, f"{idx+1+start_index}_fog.png")
            rain_mask_path = os.path.join(fog_rain_mask_dir, f"{idx+1+start_index}_rain.png")
            cv2.imwrite(fog_mask_path, fog_mask)
            cv2.imwrite(rain_mask_path, rain_mask)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")
    
    # 将分数保存到文件
    with open(os.path.join(fog_rain_dir, "scores.txt"), "a") as f:
        for score in fog_rain_scores:
            f.write(score + "\n")

    # fog_snow场景
    fog_snow_scores = []
    start_index = len([f for f in os.listdir(fog_snow_dir) if f.lower().endswith(".jpg")])
    for idx in range(double):
        img_name = gt_images[(single + double + idx) % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        fog_img, fog_score, fog_mask = add_fog(
            original_image=img,
            beta_range=(0.01, 0.08),
            brightness_range=(0.6, 0.8),
            return_mask=True
        )
        fog_snow_img, snow_score, snow_mask = add_snow(
            original_image=fog_img,
            snow_count=(50, 1200),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8,
            return_mask=True
        )
        fog_snow_scores.append(f"fog:{fog_score:.2f},snow:{snow_score:.2f}")

        dst_path = os.path.join(fog_snow_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(fog_snow_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, fog_snow_img)
        cv2.imwrite(gt_dst_path, img)

        # 如果是train模式，保存掩码
        if mode == 'train':
            fog_mask_path = os.path.join(fog_snow_mask_dir, f"{idx+1+start_index}_fog.png")
            snow_mask_path = os.path.join(fog_snow_mask_dir, f"{idx+1+start_index}_snow.png")
            cv2.imwrite(fog_mask_path, fog_mask)
            cv2.imwrite(snow_mask_path, snow_mask)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # 将分数保存到文件
    with open(os.path.join(fog_snow_dir, "scores.txt"), "a") as f:
        for score in fog_snow_scores:
            f.write(score + "\n")

    # fog_rain_snow场景
    fog_rain_snow_scores = []
    start_index = len([f for f in os.listdir(fog_rain_snow_dir) if f.lower().endswith(".jpg")])
    for idx in range(triple):
        img_name = gt_images[(single + 2*double + idx) % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        fog_img, fog_score, fog_mask = add_fog(
            original_image=img,
            beta_range=(0.01, 0.08),
            brightness_range=(0.6, 0.8),
            return_mask=True
        )
        fog_rain_img, rain_score, rain_mask = add_rain(
            original_image=fog_img,
            rain_count_range=(20, 600),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.5),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255),
            return_mask=True
        )
        fog_rain_snow_img, snow_score, snow_mask = add_snow(
            original_image=fog_rain_img,
            snow_count=(50, 1200),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8,
            return_mask=True
        )
        fog_rain_snow_scores.append(f"fog:{fog_score:.2f},rain:{rain_score:.2f},snow:{snow_score:.2f}")

        dst_path = os.path.join(fog_rain_snow_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(fog_rain_snow_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, fog_rain_snow_img)
        cv2.imwrite(gt_dst_path, img)

        # 如果是train模式，返回掩码
        if mode == 'train':
            fog_mask_path = os.path.join(fog_rain_snow_mask_dir, f"{idx+1+start_index}_fog.png")
            rain_mask_path = os.path.join(fog_rain_snow_mask_dir, f"{idx+1+start_index}_rain.png")
            snow_mask_path = os.path.join(fog_rain_snow_mask_dir, f"{idx+1+start_index}_snow.png")
            cv2.imwrite(fog_mask_path, fog_mask)
            cv2.imwrite(rain_mask_path, rain_mask)
            cv2.imwrite(snow_mask_path, snow_mask)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # 将分数保存到文件
    with open(os.path.join(fog_rain_snow_dir, "scores.txt"), "a") as f:
        for score in fog_rain_snow_scores:
            f.write(score + "\n")

# 处理rain场景
def process_rain(single=2000, double=500, triple=300, mode='train'):

    ground_truth_dir = rf"datasets\DerainDataset\{mode}\ground_truth"
    dst_base_dir = rf"datasets\MoEDataset\{mode}"

    if mode == 'train':
        rain_dir = os.path.join(dst_base_dir, "1", "rain")
        rain_ground_truth_dir = os.path.join(dst_base_dir, "1", "rain_ground_truth")
        rain_mask_dir = os.path.join(dst_base_dir, "1", "rain_mask")
        rain_fog_dir = os.path.join(dst_base_dir, "2", "fog_rain")
        rain_fog_ground_truth_dir = os.path.join(dst_base_dir, "2", "fog_rain_ground_truth")
        rain_fog_mask_dir = os.path.join(dst_base_dir, "2", "fog_rain_mask")
        rain_snow_dir = os.path.join(dst_base_dir, "2", "rain_snow")
        rain_snow_ground_truth_dir = os.path.join(dst_base_dir, "2", "rain_snow_ground_truth")
        rain_snow_mask_dir = os.path.join(dst_base_dir, "2", "rain_snow_mask")
        rain_fog_snow_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow")
        rain_fog_snow_ground_truth_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow_ground_truth")
        rain_fog_snow_mask_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow_mask")

        os.makedirs(rain_dir, exist_ok=True)
        os.makedirs(rain_ground_truth_dir, exist_ok=True)
        os.makedirs(rain_mask_dir, exist_ok=True)
        os.makedirs(rain_fog_dir, exist_ok=True)
        os.makedirs(rain_fog_ground_truth_dir, exist_ok=True)
        os.makedirs(rain_fog_mask_dir, exist_ok=True)
        os.makedirs(rain_snow_dir, exist_ok=True)
        os.makedirs(rain_snow_ground_truth_dir, exist_ok=True)
        os.makedirs(rain_snow_mask_dir, exist_ok=True)
        os.makedirs(rain_fog_snow_dir, exist_ok=True)
        os.makedirs(rain_fog_snow_ground_truth_dir, exist_ok=True)
        os.makedirs(rain_fog_snow_mask_dir, exist_ok=True)

    elif mode == 'test':
        rain_dir = os.path.join(dst_base_dir, "weather_image")
        rain_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        rain_fog_dir = os.path.join(dst_base_dir, "weather_image")
        rain_fog_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        rain_snow_dir = os.path.join(dst_base_dir, "weather_image")
        rain_snow_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        rain_fog_snow_dir = os.path.join(dst_base_dir, "weather_image")
        rain_fog_snow_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        
        os.makedirs(rain_dir, exist_ok=True)
        os.makedirs(rain_ground_truth_dir, exist_ok=True)

    gt_images = [f for f in os.listdir(ground_truth_dir) if os.path.isfile(os.path.join(ground_truth_dir, f)) and f.lower().endswith((".png", ".jpg"))]
    random.shuffle(gt_images)
    gt_num = len(gt_images)

    # rain单一场景
    rain_scores = []
    start_index = len([f for f in os.listdir(rain_dir) if f.lower().endswith(".jpg")])
    for idx in range(single):
        img_name = gt_images[idx % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        rain_img, rain_score, rain_mask = add_rain(
            original_image=img,
            rain_count_range=(20, 600),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.5),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255),
            return_mask=True
        )
        rain_scores.append(f"rain:{rain_score:.2f}")

        dst_path = os.path.join(rain_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(rain_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, rain_img)
        cv2.imwrite(gt_dst_path, img)

        # 如果是train模式，保存掩码
        if mode == 'train':
            rain_mask_path = os.path.join(rain_mask_dir, f"{idx+1+start_index}.png")
            cv2.imwrite(rain_mask_path, rain_mask)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # 将分数保存到文件
    with open(os.path.join(rain_dir, "scores.txt"), "a") as f:
        for score in rain_scores:
            f.write(score + "\n")

    # rain_fog场景
    rain_fog_scores = []
    start_index = len([f for f in os.listdir(rain_fog_dir) if f.lower().endswith(".jpg")])
    for idx in range(double):
        img_name = gt_images[(single + idx) % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        rain_img, rain_score, rain_mask = add_rain(
            original_image=img,
            rain_count_range=(20, 600),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.5),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255),
            return_mask=True
        )
        rain_fog_img, fog_score, fog_mask = add_fog(
            original_image=rain_img,
            beta_range=(0.01, 0.08),
            brightness_range=(0.6, 0.8),
            return_mask=True
        )
        rain_fog_scores.append(f"rain:{rain_score:.2f},fog:{fog_score:.2f}")

        dst_path = os.path.join(rain_fog_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(rain_fog_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, rain_fog_img)
        cv2.imwrite(gt_dst_path, img)

        # 如果是train模式，保存掩码
        if mode == 'train':
            rain_mask_path = os.path.join(rain_fog_mask_dir, f"{idx+1+start_index}_rain.png")
            fog_mask_path = os.path.join(rain_fog_mask_dir, f"{idx+1+start_index}_fog.png")
            cv2.imwrite(rain_mask_path, rain_mask)
            cv2.imwrite(fog_mask_path, fog_mask)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # 将分数保存到文件
    with open(os.path.join(rain_fog_dir, "scores.txt"), "a") as f:
        for score in rain_fog_scores:
            f.write(score + "\n")

    # rain_snow场景
    rain_snow_scores = []
    start_index = len([f for f in os.listdir(rain_snow_dir) if f.lower().endswith(".jpg")])
    for idx in range(double):
        img_name = gt_images[(single + double + idx) % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        rain_img, rain_score, rain_mask = add_rain(
            original_image=img,
            rain_count_range=(20, 600),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.5),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255),
            return_mask=True
        )
        rain_snow_img, snow_score, snow_mask = add_snow(
            original_image=rain_img,
            snow_count=(50, 1200),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8,
            return_mask=True
        )
        rain_snow_scores.append(f"rain:{rain_score:.2f},snow:{snow_score:.2f}")

        dst_path = os.path.join(rain_snow_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(rain_snow_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, rain_snow_img)
        cv2.imwrite(gt_dst_path, img)

        # 如果是train模式，保存掩码
        if mode == 'train':
            rain_mask_path = os.path.join(rain_snow_mask_dir, f"{idx+1+start_index}_rain.png")
            snow_mask_path = os.path.join(rain_snow_mask_dir, f"{idx+1+start_index}_snow.png")
            cv2.imwrite(rain_mask_path, rain_mask)
            cv2.imwrite(snow_mask_path, snow_mask)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")
    # 将分数保存到文件
    with open(os.path.join(rain_snow_dir, "scores.txt"), "a") as f:
        for score in rain_snow_scores:
            f.write(score + "\n")

    # rain_fog_snow场景
    rain_fog_snow_scores = []
    start_index = len([f for f in os.listdir(rain_fog_snow_dir) if f.lower().endswith(".jpg")])
    for idx in range(triple):
        img_name = gt_images[(single + 2*double + idx) % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        rain_img, rain_score, rain_mask = add_rain(
            original_image=img,
            rain_count_range=(20, 600),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.5),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255),
            return_mask=True
        )
        rain_fog_img, fog_score, fog_mask = add_fog(
            original_image=rain_img,
            beta_range=(0.01, 0.08),
            brightness_range=(0.6, 0.8),
            return_mask=True
        )
        rain_fog_snow_img, snow_score, snow_mask = add_snow(
            original_image=rain_fog_img,
            snow_count=(50, 1200),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8,
            return_mask=True
        )
        rain_fog_snow_scores.append(f"rain:{rain_score:.2f},fog:{fog_score:.2f},snow:{snow_score:.2f}")

        dst_path = os.path.join(rain_fog_snow_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(rain_fog_snow_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, rain_fog_snow_img)
        cv2.imwrite(gt_dst_path, img)

        # 如果是train模式，保存掩码
        if mode == 'train':
            rain_mask_path = os.path.join(rain_fog_snow_mask_dir, f"{idx+1+start_index}_rain.png")
            fog_mask_path = os.path.join(rain_fog_snow_mask_dir, f"{idx+1+start_index}_fog.png")
            snow_mask_path = os.path.join(rain_fog_snow_mask_dir, f"{idx+1+start_index}_snow.png")
            cv2.imwrite(rain_mask_path, rain_mask)
            cv2.imwrite(fog_mask_path, fog_mask)
            cv2.imwrite(snow_mask_path, snow_mask)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # 将分数保存到文件
    with open(os.path.join(rain_fog_snow_dir, "scores.txt"), "a") as f:
        for score in rain_fog_snow_scores:
            f.write(score + "\n")

def process_snow(single=2000, double=500, triple=300, mode='train'):

    ground_truth_dir = rf"datasets\DesnowDataset\{mode}\ground_truth"
    dst_base_dir = rf"datasets\MoEDataset\{mode}"

    if mode == 'train':
        snow_dir = os.path.join(dst_base_dir, "1", "snow")
        snow_ground_truth_dir = os.path.join(dst_base_dir, "1", "snow_ground_truth")
        snow_mask_dir = os.path.join(dst_base_dir, "1", "snow_mask")
        snow_fog_dir = os.path.join(dst_base_dir, "2", "fog_snow")
        snow_fog_ground_truth_dir = os.path.join(dst_base_dir, "2", "fog_snow_ground_truth")
        snow_fog_mask_dir = os.path.join(dst_base_dir, "2", "fog_snow_mask")
        snow_rain_dir = os.path.join(dst_base_dir, "2", "rain_snow")
        snow_rain_ground_truth_dir = os.path.join(dst_base_dir, "2", "rain_snow_ground_truth")
        snow_rain_mask_dir = os.path.join(dst_base_dir, "2", "rain_snow_mask")
        snow_fog_rain_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow")
        snow_fog_rain_ground_truth_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow_ground_truth")
        snow_fog_rain_mask_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow_mask")

        os.makedirs(snow_dir, exist_ok=True)
        os.makedirs(snow_ground_truth_dir, exist_ok=True)
        os.makedirs(snow_mask_dir, exist_ok=True)
        os.makedirs(snow_fog_dir, exist_ok=True)
        os.makedirs(snow_fog_ground_truth_dir, exist_ok=True)
        os.makedirs(snow_fog_mask_dir, exist_ok=True)
        os.makedirs(snow_rain_dir, exist_ok=True)
        os.makedirs(snow_rain_ground_truth_dir, exist_ok=True)
        os.makedirs(snow_rain_mask_dir, exist_ok=True)
        os.makedirs(snow_fog_rain_dir, exist_ok=True)
        os.makedirs(snow_fog_rain_ground_truth_dir, exist_ok=True)
        os.makedirs(snow_fog_rain_mask_dir, exist_ok=True)
    elif mode == 'test':
        snow_dir = os.path.join(dst_base_dir, "weather_image")
        snow_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        snow_fog_dir = os.path.join(dst_base_dir, "weather_image")
        snow_fog_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        snow_rain_dir = os.path.join(dst_base_dir, "weather_image")
        snow_rain_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        snow_fog_rain_dir = os.path.join(dst_base_dir, "weather_image")
        snow_fog_rain_ground_truth_dir = os.path.join(dst_base_dir, "ground_truth")
        
        os.makedirs(snow_dir, exist_ok=True)
        os.makedirs(snow_ground_truth_dir, exist_ok=True)

    gt_images = [f for f in os.listdir(ground_truth_dir) if os.path.isfile(os.path.join(ground_truth_dir, f)) and f.lower().endswith((".png", ".jpg"))]
    random.shuffle(gt_images)
    gt_num = len(gt_images)

    # snow单一场景
    snow_scores = []
    start_index = len([f for f in os.listdir(snow_dir) if f.lower().endswith(".jpg")])
    for idx in range(single):
        img_name = gt_images[idx % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        snow_img, snow_score, snow_mask = add_snow(
            original_image=img,
            snow_count=(50, 1200),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8,
            return_mask=True
        )
        snow_scores.append(f"snow:{snow_score:.2f}")

        dst_path = os.path.join(snow_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(snow_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, snow_img)
        cv2.imwrite(gt_dst_path, img)
        
        # 如果是train模式，保存掩码
        if mode == 'train':
            mask_dst_path = os.path.join(snow_mask_dir, f"{idx+1+start_index}.jpg")
            cv2.imwrite(mask_dst_path, snow_mask)
        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # 将分数保存到文件
    with open(os.path.join(snow_dir, "scores.txt"), "a") as f:
        for score in snow_scores:
            f.write(score + "\n")

    # snow_fog场景
    snow_fog_scores = []
    start_index = len([f for f in os.listdir(snow_fog_dir) if f.lower().endswith(".jpg")])
    for idx in range(double):
        img_name = gt_images[(single + idx) % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        snow_img, snow_score, snow_mask = add_snow(
            original_image=img,
            snow_count=(50, 1200),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8,
            return_mask=True
        )
        snow_fog_img, fog_score, fog_mask = add_fog(
            original_image=snow_img,
            beta_range=(0.01, 0.08),
            brightness_range=(0.6, 0.8),
            return_mask=True
        )
        snow_fog_scores.append(f"snow:{snow_score:.2f},fog:{fog_score:.2f}")

        dst_path = os.path.join(snow_fog_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(snow_fog_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, snow_fog_img)
        cv2.imwrite(gt_dst_path, img)
        
        # 如果是train模式，保存掩码
        if mode == 'train':
            snow_mask_dst_path =  os.path.join(snow_fog_mask_dir, f"{idx+1+start_index}_snow.jpg")
            fog_mask_dst_path =  os.path.join(snow_fog_mask_dir, f"{idx+1+start_index}_fog.jpg")
            cv2.imwrite(snow_mask_dst_path, snow_mask)
            cv2.imwrite(fog_mask_dst_path, fog_mask)
        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # 将分数保存到文件
    with open(os.path.join(snow_fog_dir, "scores.txt"), "a") as f:
        for score in snow_fog_scores:
            f.write(score + "\n")

    # snow_rain场景
    snow_rain_scores = []
    start_index = len([f for f in os.listdir(snow_rain_dir) if f.lower().endswith(".jpg")])
    for idx in range(double):
        img_name = gt_images[(single + double + idx) % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        snow_img, snow_score, snow_mask = add_snow(
            original_image=img,
            snow_count=(50, 1200),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8,
            return_mask=True
        )
        snow_rain_img, rain_score, rain_mask = add_rain(
            original_image=snow_img,
            rain_count_range=(20, 600),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.5),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255),
            return_mask=True
        )
        snow_rain_scores.append(f"snow:{snow_score:.2f},rain:{rain_score:.2f}")

        dst_path = os.path.join(snow_rain_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(snow_rain_ground_truth_dir, f"{idx+1+start_index}.jpg")
        cv2.imwrite(dst_path, snow_rain_img)
        cv2.imwrite(gt_dst_path, img)
        
        # 如果是train模式，保存掩码
        if mode == 'train':
            snow_mask_dst_path =  os.path.join(snow_rain_mask_dir, f"{idx+1+start_index}_snow.jpg")
            rain_mask_dst_path =  os.path.join(snow_rain_mask_dir, f"{idx+1+start_index}_rain.jpg")
            cv2.imwrite(snow_mask_dst_path, snow_mask)
            cv2.imwrite(rain_mask_dst_path, rain_mask)
            print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # 将分数保存到文件
    with open(os.path.join(snow_rain_dir, "scores.txt"), "a") as f:
        for score in snow_rain_scores:
            f.write(score + "\n")

    # snow_fog_rain场景
    snow_fog_rain_scores = []
    start_index = len([f for f in os.listdir(snow_fog_rain_dir) if f.lower().endswith(".jpg")])
    for idx in range(triple):
        img_name = gt_images[(single + 2*double + idx) % gt_num]
        src_path = os.path.join(ground_truth_dir, img_name)
        img = cv2.imread(src_path)
        snow_img, snow_score, snow_mask = add_snow(
            original_image=img,
            snow_count=(50, 1200),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8,
            return_mask=True
        )
        snow_fog_img, fog_score, fog_mask = add_fog(
            original_image=snow_img,
            beta_range=(0.01, 0.08),
            brightness_range=(0.6, 0.8),
            return_mask=True
        )
        snow_fog_rain_img, rain_score, rain_mask = add_rain(
            original_image=snow_fog_img,
            rain_count_range=(20, 600),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.5),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255),
            return_mask=True
        )
        snow_fog_rain_scores.append(f"snow:{snow_score:.2f},fog:{fog_score:.2f},rain:{rain_score:.2f}")
        dst_path = os.path.join(snow_fog_rain_dir, f"{idx+1+start_index}.jpg")
        gt_dst_path = os.path.join(snow_fog_rain_ground_truth_dir, f"{idx+1+start_index}.jpg")
        
        cv2.imwrite(dst_path, snow_fog_rain_img)
        cv2.imwrite(gt_dst_path, img)
        
        # 如果是train模式，保存掩码
        if mode == 'train':
            snow_mask_dst_path =  os.path.join(snow_fog_rain_mask_dir, f"{idx+1+start_index}_snow.jpg")
            fog_mask_dst_path =  os.path.join(snow_fog_rain_mask_dir, f"{idx+1+start_index}_fog.jpg")
            rain_mask_dst_path =  os.path.join(snow_fog_rain_mask_dir, f"{idx+1+start_index}_rain.jpg")
            cv2.imwrite(snow_mask_dst_path, snow_mask)
            cv2.imwrite(fog_mask_dst_path, fog_mask)
            cv2.imwrite(rain_mask_dst_path, rain_mask)
        
        print(f"saved: {dst_path}, gt: {gt_dst_path}")
    # 将分数保存到文件
    with open(os.path.join(snow_fog_rain_dir, "scores.txt"), "a") as f:
        for score in snow_fog_rain_scores:
            f.write(score + "\n")

if __name__ == "__main__":
    process_fog(single=2000, double=500, triple=300, mode='train')
    process_rain(single=2000, double=500, triple=300, mode='train')
    process_snow(single=2000, double=500, triple=300, mode='train')

    process_fog(single=20, double=20, triple=20, mode='test')
    process_rain(single=20, double=20, triple=20, mode='test')
    process_snow(single=20, double=20, triple=20, mode='test')