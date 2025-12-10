from .ResNet18_2d import ResNet18 as ResNet2d
from .ResNet18_1d import ResNet18_1d
from .ResNet1d import ResNet_test as ResNet1d
from .MyCNN import MyCNN
from .DRSN_CW import DRSN_CW_test as DRSN_CW
from .TimeFreqAttention import create_model as TimeFreqAttention

# 模型注册表
MODEL_REGISTRY = {
    'ResNet18_2d': ResNet2d,
    'ResNet18_1d': ResNet18_1d,
    'ResNet1d': ResNet1d,
    'MyCNN': MyCNN,
    'DRSN-CW': DRSN_CW,
    'TimeFreqAttention': TimeFreqAttention,
    
    # 别名兼容
    'ResNet2d': ResNet2d, # 兼容旧配置
    'ResNet': ResNet1d, # 默认 ResNet 指向 1D，因为 RSBU 代码量更多？或者让用户明确指定。
                        # 这里先保留，防止某些配置没改完报错，但在配置中我会明确指定。
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_fn = MODEL_REGISTRY[model_name]
    return model_fn(**kwargs)
