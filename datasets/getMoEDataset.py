import os
import random
from addFog import add_fog
from addRain import add_rain
from addSnow import add_snow
import cv2


# 从DefogDataser中随机选择2000张作为fog单一场景，保存在MoEDataset/1/fog目录下。

# 再从剩下的中随机选择500+500张，分别执行add_rain和add_snow，作为fog_rain和fog_snow场景，
# 保存在MoEDataset/2/fog_rain和MoEDataset/2/fog_snow目录下。

# 再从剩下的中随机选择300张，执行add_rain和add_snow，作为fog_rain_snow场景，
# 保存在MoEDataset/3/fog_rain_snow目录下。

# 按照fog->rain->snow的顺序命名。

# 处理fog场景
def process_fog(single=2000, double=500, triple=300):

    fog_src_dir = r"datasets\DefogDataset\train\foggy_image"
    ground_truth_dir = r"datasets\DefogDataset\train\ground_truth"
    dst_base_dir = r"datasets\MoEDataset"

    fog_dir = os.path.join(dst_base_dir, "1", "fog")
    fog_ground_truth_dir = os.path.join(dst_base_dir, "1", "fog_ground_truth")
    fog_rain_dir = os.path.join(dst_base_dir, "2", "fog_rain")
    fog_rain_ground_truth_dir = os.path.join(dst_base_dir, "2", "fog_rain_ground_truth")
    fog_snow_dir = os.path.join(dst_base_dir, "2", "fog_snow")
    fog_snow_ground_truth_dir = os.path.join(dst_base_dir, "2", "fog_snow_ground_truth")
    fog_rain_snow_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow")
    fog_rain_snow_ground_truth_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow_ground_truth")

    os.makedirs(fog_dir, exist_ok=True)
    os.makedirs(fog_ground_truth_dir, exist_ok=True)
    os.makedirs(fog_rain_dir, exist_ok=True)
    os.makedirs(fog_rain_ground_truth_dir, exist_ok=True)
    os.makedirs(fog_snow_dir, exist_ok=True)
    os.makedirs(fog_snow_ground_truth_dir, exist_ok=True)
    os.makedirs(fog_rain_snow_dir, exist_ok=True)
    os.makedirs(fog_rain_snow_ground_truth_dir, exist_ok=True)

    fog_images = [f for f in os.listdir(fog_src_dir) if os.path.isfile(os.path.join(fog_src_dir, f)) and f.lower().endswith((".png", ".jpg"))]
    random.shuffle(fog_images)

    # fog单一场景直接复制
    start_idx = len([f for f in os.listdir(fog_dir) if os.path.isfile(os.path.join(fog_dir, f))])
    for idx, img_name in enumerate(fog_images[:single]):
        src_path = os.path.join(fog_src_dir, img_name)
        dst_path = os.path.join(fog_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        cv2.imwrite(dst_path, image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(fog_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # fog_rain场景: 在foggy_image基础上加rain
    start_idx = len([f for f in os.listdir(fog_rain_dir) if os.path.isfile(os.path.join(fog_rain_dir, f))])
    for idx, img_name in enumerate(fog_images[single:single+double]):
        src_path = os.path.join(fog_src_dir, img_name)
        dst_path = os.path.join(fog_rain_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        rainy_image = add_rain(
            original_image=image,
            rain_count_range=(100, 800),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.6),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255)
        )
        cv2.imwrite(dst_path, rainy_image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(fog_rain_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # fog_snow场景: 在foggy_image基础上加snow
    start_idx = len([f for f in os.listdir(fog_snow_dir) if os.path.isfile(os.path.join(fog_snow_dir, f))])
    for idx, img_name in enumerate(fog_images[single+double:single+2*double]):
        src_path = os.path.join(fog_src_dir, img_name)
        dst_path = os.path.join(fog_snow_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        snowy_image = add_snow(
            original_image=image,
            snow_count=(200, 1500),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8
        )
        cv2.imwrite(dst_path, snowy_image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(fog_snow_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # fog_rain_snow场景: 在foggy_image基础上加rain再加snow
    start_idx = len([f for f in os.listdir(fog_rain_snow_dir) if os.path.isfile(os.path.join(fog_rain_snow_dir, f))])
    for idx, img_name in enumerate(fog_images[single+2*double:single+2*double+triple]):
        src_path = os.path.join(fog_src_dir, img_name)
        dst_path = os.path.join(fog_rain_snow_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        rainy_image = add_rain(
            original_image=image,
            rain_count_range=(100, 800),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.6),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255)
        )
        snowy_rainy_image = add_snow(
            original_image=rainy_image,
            snow_count=(200, 1500),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8
        )
        cv2.imwrite(dst_path, snowy_rainy_image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(fog_rain_snow_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

# 处理rain场景
def process_rain(single=2000, double=500, triple=300):

    rain_src_dir = r"datasets\DerainDataset\train\rainy_image"
    ground_truth_dir = r"datasets\DerainDataset\train\ground_truth"
    dst_base_dir = r"datasets\MoEDataset"

    rain_dir = os.path.join(dst_base_dir, "1", "rain")
    rain_ground_truth_dir = os.path.join(dst_base_dir, "1", "rain_ground_truth")
    rain_fog_dir = os.path.join(dst_base_dir, "2", "fog_rain")
    rain_fog_ground_truth_dir = os.path.join(dst_base_dir, "2", "fog_rain_ground_truth")
    rain_snow_dir = os.path.join(dst_base_dir, "2", "rain_snow")
    rain_snow_ground_truth_dir = os.path.join(dst_base_dir, "2", "rain_snow_ground_truth")
    rain_fog_snow_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow")
    rain_fog_snow_ground_truth_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow_ground_truth")

    os.makedirs(rain_dir, exist_ok=True)
    os.makedirs(rain_ground_truth_dir, exist_ok=True)
    os.makedirs(rain_fog_dir, exist_ok=True)
    os.makedirs(rain_fog_ground_truth_dir, exist_ok=True)
    os.makedirs(rain_snow_dir, exist_ok=True)
    os.makedirs(rain_snow_ground_truth_dir, exist_ok=True)
    os.makedirs(rain_fog_snow_dir, exist_ok=True)
    os.makedirs(rain_fog_snow_ground_truth_dir, exist_ok=True)

    rain_images = [f for f in os.listdir(rain_src_dir) if os.path.isfile(os.path.join(rain_src_dir, f)) and f.lower().endswith((".png", ".jpg"))]
    random.shuffle(rain_images)

    # rain单一场景直接复制
    start_idx = len([f for f in os.listdir(rain_dir) if os.path.isfile(os.path.join(rain_dir, f))])
    for idx, img_name in enumerate(rain_images[:single]):
        src_path = os.path.join(rain_src_dir, img_name)
        dst_path = os.path.join(rain_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        cv2.imwrite(dst_path, image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(rain_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # rain_fog场景: 在rainy_image基础上加fog
    start_idx = len([f for f in os.listdir(rain_fog_dir) if os.path.isfile(os.path.join(rain_fog_dir, f))])
    for idx, img_name in enumerate(rain_images[single:single+double]):
        src_path = os.path.join(rain_src_dir, img_name)
        dst_path = os.path.join(rain_fog_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        foggy_rainy_image = add_fog(
            original_image=image,
            beta_range=(0, 0.08),
            brightness_range=(0.6, 0.8)
        )
        cv2.imwrite(dst_path, foggy_rainy_image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(rain_fog_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # rain_snow场景: 在rainy_image基础上加snow
    start_idx = len([f for f in os.listdir(rain_snow_dir) if os.path.isfile(os.path.join(rain_snow_dir, f))])
    for idx, img_name in enumerate(rain_images[single+double:single+2*double]):
        src_path = os.path.join(rain_src_dir, img_name)
        dst_path = os.path.join(rain_snow_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        snowy_rainy_image = add_snow(
            original_image=image,
            snow_count=(200, 1500),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8
        )
        cv2.imwrite(dst_path, snowy_rainy_image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(rain_snow_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # rain_fog_snow场景: 在rainy_image基础上加fog再加snow
    start_idx = len([f for f in os.listdir(rain_fog_snow_dir) if os.path.isfile(os.path.join(rain_fog_snow_dir, f))])
    for idx, img_name in enumerate(rain_images[single+2*double:single+2*double+triple]):
        src_path = os.path.join(rain_src_dir, img_name)
        dst_path = os.path.join(rain_fog_snow_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        foggy_rainy_image = add_fog(
            original_image=image,
            beta_range=(0, 0.08),
            brightness_range=(0.6, 0.8)
        )
        snowy_foggy_rainy_image = add_snow(
            original_image=foggy_rainy_image,
            snow_count=(200, 1500),
            snow_size_range=((1, 2), (2, 3)),
            small_radio=(0.75, 0.95),
            alpha=(0.2, 0.3),
            wind_speed=((1, 2), (1, 2)),
            blur_angle_variance=20,
            snow_intensity=0.8
        )
        cv2.imwrite(dst_path, snowy_foggy_rainy_image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(rain_fog_snow_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

def process_snow(single=2000, double=500, triple=300):

    snow_src_dir = r"datasets\DesnowDataset\train\snowy_image"
    ground_truth_dir = r"datasets\DesnowDataset\train\ground_truth"
    dst_base_dir = r"datasets\MoEDataset"

    snow_dir = os.path.join(dst_base_dir, "1", "snow")
    snow_ground_truth_dir = os.path.join(dst_base_dir, "1", "snow_ground_truth")
    snow_fog_dir = os.path.join(dst_base_dir, "2", "fog_snow")
    snow_fog_ground_truth_dir = os.path.join(dst_base_dir, "2", "fog_snow_ground_truth")
    snow_rain_dir = os.path.join(dst_base_dir, "2", "rain_snow")
    snow_rain_ground_truth_dir = os.path.join(dst_base_dir, "2", "rain_snow_ground_truth")
    snow_fog_rain_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow")
    snow_fog_rain_ground_truth_dir = os.path.join(dst_base_dir, "3", "fog_rain_snow_ground_truth")

    os.makedirs(snow_dir, exist_ok=True)
    os.makedirs(snow_ground_truth_dir, exist_ok=True)
    os.makedirs(snow_fog_dir, exist_ok=True)
    os.makedirs(snow_fog_ground_truth_dir, exist_ok=True)
    os.makedirs(snow_rain_dir, exist_ok=True)
    os.makedirs(snow_rain_ground_truth_dir, exist_ok=True)
    os.makedirs(snow_fog_rain_dir, exist_ok=True)
    os.makedirs(snow_fog_rain_ground_truth_dir, exist_ok=True)

    snow_images = [f for f in os.listdir(snow_src_dir) if os.path.isfile(os.path.join(snow_src_dir, f)) and f.lower().endswith((".png", ".jpg"))]
    random.shuffle(snow_images)

    # snow单一场景直接复制
    start_idx = len([f for f in os.listdir(snow_dir) if os.path.isfile(os.path.join(snow_dir, f))])
    for idx, img_name in enumerate(snow_images[:single]):
        src_path = os.path.join(snow_src_dir, img_name)
        dst_path = os.path.join(snow_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        cv2.imwrite(dst_path, image)
        
        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(snow_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)
        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # snow_fog场景: 在snowy_image基础上加fog
    start_idx = len([f for f in os.listdir(snow_fog_dir) if os.path.isfile(os.path.join(snow_fog_dir, f))])
    for idx, img_name in enumerate(snow_images[single:single+double]):
        src_path = os.path.join(snow_src_dir, img_name)
        dst_path = os.path.join(snow_fog_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        foggy_snowy_image = add_fog(
            original_image=image,
            beta_range=(0, 0.08),
            brightness_range=(0.6, 0.8)
        )
        cv2.imwrite(dst_path, foggy_snowy_image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(snow_fog_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # snow_rain场景: 在snowy_image基础上加rain
    start_idx = len([f for f in os.listdir(snow_rain_dir) if os.path.isfile(os.path.join(snow_rain_dir, f))])
    for idx, img_name in enumerate(snow_images[single+double:single+2*double]):
        src_path = os.path.join(snow_src_dir, img_name)
        dst_path = os.path.join(snow_rain_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        rainy_snowy_image = add_rain(
            original_image=image,
            rain_count_range=(100, 800),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.6),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255)
        )
        cv2.imwrite(dst_path, rainy_snowy_image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(snow_rain_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

    # snow_fog_rain场景: 在snowy_image基础上加fog再加rain
    start_idx = len([f for f in os.listdir(snow_fog_rain_dir) if os.path.isfile(os.path.join(snow_fog_rain_dir, f))])
    for idx, img_name in enumerate(snow_images[single+2*double:single+2*double+triple]):
        src_path = os.path.join(snow_src_dir, img_name)
        dst_path = os.path.join(snow_fog_rain_dir, f"{idx+1+start_idx}.jpg")
        image = cv2.imread(src_path)
        foggy_snowy_image = add_fog(
            original_image=image,
            beta_range=(0, 0.08),
            brightness_range=(0.6, 0.8)
        )
        rainy_foggy_snowy_image = add_rain(
            original_image=foggy_snowy_image,
            rain_count_range=(100, 800),
            rain_length_range=(8, 30),
            rain_width_range=(1, 1),
            rain_alpha_range=(0.3, 0.6),
            blur_angle_range=(70, 110),
            blur_length_range=(3, 5),
            rain_brightness=240,
            rain_color=(255, 255, 255)
        )
        cv2.imwrite(dst_path, rainy_foggy_snowy_image)

        # 同步保存ground_truth
        gt_src_path = os.path.join(ground_truth_dir, '_'.join(img_name.split('_')[:-1]) + '.jpg')
        gt_dst_path = os.path.join(snow_fog_rain_ground_truth_dir, f"{idx+1+start_idx}.jpg")
        if os.path.exists(gt_src_path):
            gt_image = cv2.imread(gt_src_path)
            cv2.imwrite(gt_dst_path, gt_image)

        print(f"saved: {dst_path}, gt: {gt_dst_path}")

if __name__ == "__main__":
    process_fog(single=2000, double=500, triple=300)
    process_rain(single=2000, double=500, triple=300)
    process_snow(single=2000, double=500, triple=300)