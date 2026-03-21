from .ResNet18_2d import ResNet18 as ResNet2d
from .ResNet18_2d_Light import ResNet18_2d_Light
from .LaoDA import LaoDA

MODEL_REGISTRY = {
    'ResNet18_2d': ResNet2d,
    'ResNet18_2d_Light': ResNet18_2d_Light,
    'ResNet2d': ResNet2d, # 兼容
    'LaoDA': LaoDA,
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_fn = MODEL_REGISTRY[model_name]
    return model_fn(**kwargs)
