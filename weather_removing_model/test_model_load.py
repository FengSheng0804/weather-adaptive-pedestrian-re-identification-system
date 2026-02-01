# 测试预训练模型是否能够正常加载

import torch
from models.experts.DEANetExpert import DEANetExpert
from models.experts.PReNetExpert import PReNetExpert
from models.experts.ConvIRExpert import ConvIRExpert

# 对于分布式训练保存的模型参数，去掉'module.'前缀，因为DataParallel会添加该前缀
def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def test_model_load():
    # 测试DEANetExpert模型加载
    deanet_model = DEANetExpert(return_features=False)
    cpkt = torch.load('weather_removing_model/weights/DEANet_pretrained.pk')
    missing_keys, unexpected_keys = deanet_model.load_state_dict(remove_module_prefix(cpkt['model']), strict=False)
    print("=" * 50)
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)

    # 测试PReNetExpert模型加载
    prenet_model = PReNetExpert(return_features=False)
    missing_keys, unexpected_keys = prenet_model.load_state_dict(torch.load('weather_removing_model/weights/PReNet_pretrained.pth'), strict=False)
    print("=" * 50)
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)

    # 测试ConvIRExpert模型加载
    convir_model = ConvIRExpert(return_features=False)
    missing_keys, unexpected_keys = convir_model.load_state_dict(torch.load('weather_removing_model/weights/ConvIR_pretrained.pkl')['model'], strict=False)
    print("=" * 50)
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)

if __name__ == "__main__":
    test_model_load()