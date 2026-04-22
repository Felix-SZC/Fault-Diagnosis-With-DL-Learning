from .LaoDA import LaoDA

MODEL_REGISTRY = {
    'LaoDA': LaoDA,
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_fn = MODEL_REGISTRY[model_name]
    return model_fn(**kwargs)
