from models.ResNet import ResNet_test
from models.DRSN_CW import DRSN_CW_test
# 模型注册字典，映射模型名称到模型构造函数
MODEL_REGISTRY = {
    'ResNet': ResNet_test,
    'DRSN-CW': DRSN_CW_test,
}

def get_model(model_config):
    """
    根据配置自动选择和实例化模型。
    
    Args:
        model_config: 模型配置字典，应包含 'type' 和 'num_classes' 字段
        
    Returns:
        实例化的模型
        
    Raises:
        ValueError: 如果指定的模型类型不存在
    """
    model_type = model_config.get('type')
    if model_type is None:
        raise ValueError("模型配置中缺少 'type' 字段，请指定模型类型")
    
    if model_type not in MODEL_REGISTRY:
        available_models = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"未知的模型类型: '{model_type}'. "
            f"可用的模型类型: {available_models}"
        )
    
    model_fn = MODEL_REGISTRY[model_type]
    num_classes = model_config.get('num_classes', 10)
    
    return model_fn(num_classes=num_classes)

