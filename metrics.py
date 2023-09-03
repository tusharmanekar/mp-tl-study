import torch
import numpy as np
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def compute_layer_variances_dense(model, test_loader, device='cpu', cnn=True):
    # Define a hook to collect activations
    activations = {}

    def create_hook(name):
        def hook(module, input, output):
            activations[name] = output
        return hook


    for name, layer in model.named_modules():
        if name:  # this ensures we skip the top-level module (the entire model) which has an empty name
            layer.register_forward_hook(create_hook(name))

    # Run inference on the test set
    with torch.no_grad():
        for data, target in test_loader:
            if cnn:
                data = data.to(device)
            else:
                data, target = data.reshape([data.shape[0], -1]).to(device), target.to(device)
            outputs = model(data)

    # Calculate variance for each layer's activations
    variances = {key: torch.var(act, dim=0) for key, act in activations.items()}
    
    result = {}
    for layer_name, variance in variances.items():
        result[layer_name] = {
            'variance': variance.mean().item(),
            'variance_of_variance': torch.var(variance).item()
        }

    return result, variances