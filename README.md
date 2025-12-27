# 自适应行人重识别系统 (Adaptive Pedestrian Re-identification System)

## 1. 项目简介

本项目旨在开发一个自适应的行人重识别系统，能够有效应对雨、雾等恶劣天气条件对图像质量的影响。系统通过集成先进的去雨和去雾模型，在进行行人重识别之前对输入图像进行预处理，从而提高在复杂环境下的识别精度和鲁棒性。

## 2. 项目结构

```
.
├── README.md
├── datasets
│   ├── DefogDataset/     # 去雾数据集(HAZE4K)
│   ├── DerainDataset/    # 去雨数据集(Rain1400)
│   └── DesnowDataset/    # 去雾数据集(Snow100K)
├── fog_removing_model    # 去雾模型 (DEANet)
│   ├── train_DEANet.py   # 训练脚本
│   ├── test_DEANet.py    # 测试脚本
│   ├── model/            # 模型定义
│   ├── data/             # 数据加载器
│   └── weights/          # 预训练权重
├── rain_removing_model   # 去雨模型 (PReNet)
│   ├── train_PReNet.py   # 训练脚本
│   ├── test_PReNet.py    # 测试脚本
│   ├── model/            # 模型定义
│   ├── data/             # 数据加载器
│   └── weights/          # 预训练权重
└── snow_removing_model   # 去雪模型 (ConvIR)
    ├── train.py          # 训练脚本
    ├── test.py           # 测试脚本
    ├── model/            # 模型定义
    └── results/          # 结果保存目录
```

## 3. 数据集

本项目使用了以下两个数据集进行模型的训练和评估：

*   **DefogDataset**: 用于训练和测试去雾模型。
    *   `train/foggy_image`: 训练用带雾图像。
    *   `train/ground_truth`: 训练用无雾清晰图像。
    *   `test/foggy_image`: 测试用带雾图像。
    *   `test/ground_truth`: 测试用无雾清晰图像。
*   **DerainDataset**: 用于训练和测试去雨模型。
    *   `train/rainy_image`: 训练用带雨图像。
    *   `train/ground_truth`: 训练用无雨清晰图像。
    *   `test/rainy_image`: 测试用带雨图像。
    *   `test/ground_truth`: 测试用无雨清晰图像。

## 4. 模型

### 4.1 去雾模型: DEANet

*   **模型描述**: `DEANet` 是一个用于图像去雾的深度学习模型。
*   **相关文件**:
    *   模型结构: `fog_removing_model/model/`
    *   训练脚本: `fog_removing_model/train_DEANet.py`
    *   转换脚本: `fog_removing_model/reparam.py`
    *   测试脚本: `fog_removing_model/test_DEANet.py`

### 4.2 去雨模型: PReNet

*   **模型描述**: `PReNet` 是一个用于图像去雨的深度学习模型。
*   **相关文件**:
    *   模型结构: `rain_removing_model/model/PReNet.py`
    *   训练脚本: `rain_removing_model/train_PReNet.py`
    *   测试脚本: `rain_removing_model/test_PReNet.py`

### 4.3 去雪模型: ConvIR

*   **模型描述**: `ConvIR` 是一个用于图像去雪的深度学习模型。
*   **相关文件**:
    *   模型结构: `snow_removing_model/model/ConvIR.py`
    *   训练脚本: `snow_removing_model/train.py`
    *   测试脚本: `snow_removing_model/test.py`

## 5. 使用说明

### 5.1 环境配置

建议创建一个 Python 虚拟环境，并安装所需的依赖库。
```bash
pip install -r requirements.txt
```
*(注意: 项目中暂未提供 `requirements.txt` 文件，您需要根据代码中的 `import` 手动创建)*

### 5.2 模型训练

#### 训练去雾模型

```bash
python fog_removing_model/train_DEANet.py --data_dir ./datasets/DefogDataset
```

#### 训练去雨模型

```bash
python rain_removing_model/train_PReNet.py --data_dir ./datasets/DerainDataset
```

#### 训练去雪模型

```bash
python snow_removing_model/train.py --data_dir ./datasets/DesnowDataset
```

### 5.3 模型测试

#### 测试去雾模型

测试去雾模型之前，需要使用`reparam.py`文件对`train_DEANet.py`训练出来的模型进行重参数化。

```bash
python fog_removing_model/test_DEANet.py --data_dir ./datasets/DefogDataset/test --weights ./fog_removing_model/weights/best.pth
```

#### 测试去雨模型

```bash
python rain_removing_model/test_PReNet.py --data_dir ./datasets/DerainDataset/test --weights ./rain_removing_model/weights/best.pth
```

#### 测试去雪模型

```bash
python snow_removing_model/test.py --data_dir ./datasets/DesnowDataset/test --test_model ./snow_removing_model/weights/best.pkl
```

## 6. 结果

测试结果（包括去雾、去雨和去雪后的图像）将保存在相应模型的 `results` 文件夹中。

## 7. 未来工作

*   将去雾、去雨和去雪模块作为MOE的专家子网络训练混合专家模型。
*   将混合专家模型和行人重识别任务相结合。
*   

## 8. 困难与挑战

1. 在使用PReNet进行预测的时候，由于尺寸的问题，需要对图像进行resize操作，但是如果使用resize操作，会导致经过预测后的图像产生锯齿。
   1. 解决办法：使用分块训练的方法。

2. MoE架构的数据集如何生成？

3. 在加载预训练模型的时候，对于DEANet，使用`fog_removing_model/models/backbone_train.py`中的DEANet的网络结构无法正常加载`fog_removing_model\weights\best_before_reparam.pk`模型？
   1. 原因是在于，在训练DEANet.py的时候，保存网络结构之前，会使用`DataParallal`对网络进行处理:
   ```python
    if opt.device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
   ```
   导致在保存模型的参数的时候，会自动加上`module.`前缀，使得参数不能正常加载，因此需要使用专门的函数将该前缀删除:
   ```python
    def remove_module_prefix(state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
   ```
   这样才能正常加载网络模型。
   如果参数对应不上，可以通过以下方式对模型的参数进行检查，同时需要注意保存模型的时候是否直接保存的参数，如果不是，可能还需要通过访问字典['model']才可以加载参数。
   ```python
    convir_model = ConvIRExpert(return_features=False)
    missing_keys, unexpected_keys = convir_model.load_state_dict(torch.load('weather_removing_model/weights/ConvIR_pretrained.pkl')['model'], strict=False)
    print("=" * 50)
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)
   ```
4. 在训练MoE架构的时候，发现输出的专家权重与输入的分数相关性极差，说明“分数特征→门控”的通路没有发挥作用，且专家的预训练模型出现遗忘现象，而是重建头在进行兜底。同时在门控应用apply_mask_and_renorm函数之前，expert_weights的三个权重中，第一个的权重非常大（0.95左右）；而feature_weights的三个权重中，第二个的权重非常大（0.95左右）出现错误的可能原因有：
   1. 若训练集中“雾”场景占比或分数偏高，会把门控推向“去雾专家”常胜分布，可尝试对三个分数的尺度进行处理，使分布更均衡。
   2. 学习率过高导致专家预训练的知识跑偏，可尝试先冻结三个专家，仅训练 gate + fusion + reconstruction 5–10 epoch，但要注意不能冻结太久；再解冻专家，给专家更低学习率。参数分组学习率：gate/fusion/recon: 1e-4，experts: 1e-5 或更低。
   3. VGG 感知/对比损失在未做 ImageNet 归一化时，可能引入色彩偏移压力；或权重过大让模型优先“远离输入”，出现颜色异常。可尝试对 final_out/targets/inputs 做 ImageNet 归一化，并降低对比损失权重

5. 发现对于训练时候的数据集，对于所有的单一天气图像，moe_output都能有非常好的处理结果，但是只要是有雾在的双/多场景图像，只会除掉雾，而且雾处理的非常好，雪和雨完全无法去除。
   1. 可能的原因还是如4所说，雾的loss更低，导致只要存在雾，模型就会将更多的权重分给雾。

