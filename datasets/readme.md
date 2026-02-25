# 数据集说明

本目录存放用于训练和测试图像去雾、去雨、去雪模型的所有数据集。

## 目录结构

```
datasets/
├── DefogDataset/     # 去雾数据集(ground_truth共1000张)
│   ├── test/
│   │   ├── foggy_image/    # 测试用带雾图像
│   │   └── ground_truth/   # 测试用无雾清晰图像
│   └── train/
│       ├── foggy_image/    # 训练用带雾图像
│       └── ground_truth/   # 训练用无雾清晰图像
│
├── DerainDataset/    # 去雨数据集(ground_truth共1000张)
│   ├── test/
│   │   ├── rainy_image/    # 测试用带雨图像
│   │   └── ground_truth/   # 测试用无雨清晰图像
│   └── train/
│       ├── rainy_image/    # 训练用带雨图像
│       └── ground_truth/   # 训练用无雨清晰图像
│
├── DesnowDataset/    # 去雪数据集(ground_truth共4500张)
│   ├── test/
│   │   ├── snowy_image/    # (假设) 测试用带雪图像
│   │   └── ground_truth/   # (假设) 测试用无雪清晰图像
│   └── train/
│       ├── snowy_image/    # (假设) 训练用带雪图像
│       └── ground_truth/   # (假设) 训练用无雪清晰图像
│
└── MoEDataset/       # 混合专家模型数据集
    ├── test/
    │   ├── weather_image/  # 测试用天气图像
    │   └── ground_truth/   # 测试用清晰图像
    └── train/
        ├── 1/              # 单一天气 (fog, rain, snow)
        ├── 2/              # 双重天气 (fog_rain, ...)
        └── 3/              # 三重天气 (fog_rain_snow)
```

## 数据集详情

### 1. DefogDataset (去雾)

*   用于训练和评估去雾模型。
*   包含成对的带雾图像和对应的清晰图像。
*   共有训练集4500+测试集500

### 2. DerainDataset (去雨)

*   用于训练和评估去雨模型。
*   包含成对的带雨图像和对应的清晰图像。
*   共有训练集4500+测试集500

### 3. DesnowDataset (去雪)

*   用于训练和评估去雪模型。
*   包含成对的带雪图像和对应的清晰图像。
*   共有训练集4000+测试集1000

### 4. MoEDataset (混合专家数据集)

*   用于训练 MoE 模型的门控网络及微调专家。
*   **结构**:
    *   `train/1`: 单因素天气 (fog, rain, snow)，包含 mask 和 scores.txt。
    *   `train/2`: 双因素叠加 (fog_rain, fog_snow, rain_snow)。
    *   `train/3`: 三因素叠加 (fog_rain_snow)。
    *   `test`: 包含各类天气情况的测试图像。

### 5. PersonReIDDataset (行人重识别数据集)

*   用于训练和评估行人重识别模型。
*   由 Market-1501 子数据集制作而成。
*   每个子数据集包含训练集、查询集和图库。

#### 数据集结构:
- bounding_box_test 是测试集，包括 19732 张图片。
- bounding_box_train 是训练集，包括 12936 张图片。
- gt_bbox 是手工标注的训练集和测试集图片，包括 25259 张图片，用来区分 “good” “junk” 和 “distractors” 图片。
- query 是待查找的图片集，在 bounding_box_test 中实现查找。这些图片是手动绘制生成的。而 gallery 是通过 DPM 检测器生成的。
- gt_query 是一些 Matlab 格式的文件，里面记录了 “good” 和 “junk” 图片的索引，主要被用来评估模型。
#### 数据集命名规则
- 0012 是行人 ID，Market 1501 有 1501 个行人，故行人 ID 范围为 0001-1501
- c4 是摄像头编号(camera 4)，表明图片采集自第4个摄像头，一共有 6 个摄像头
- s1 是视频的第一个片段(sequece1)，一个视频包含若干个片段
- 000826 是视频的第 826 帧图片，表明行人出现在该帧图片中
- 01 代表第 826 帧图片上的第一个检测框，DPM 检测器可能在一帧图片上生成多个检测框