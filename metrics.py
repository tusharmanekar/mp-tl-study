import torch
import numpy as np
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

'''# ACTIVATION-BASED METRIC 1: ACTIVATION VARIANCES
def compute_layer_variances_dense(model, test_loader, device='cpu', cnn=True):
    activations = {}

    for name, layer in model.named_modules():
        if name:  # this ensures we skip the top-level module (the entire model) which has an empty name
            activations[name] = []

    def create_hook(name):
        def hook(module, input, output):
            activations[name].extend(list(torch.var(output, dim=1).detach().numpy()))
        return hook


    for name, layer in model.named_modules():
        if name:  # this ensures we skip the top-level module (the entire model) which has an empty name
            activations[name] = []
            layer.register_forward_hook(create_hook(name))

    # Run inference on the test set
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cnn:
                data = data.to(device)
            else:
                data, target = data.reshape([data.shape[0], -1]).to(device), target.to(device)
            outputs = model(data)
            i += 1
            
    # Calculate variance for each layer's activations
    result = {}
    for layer_name, variances in activations.items():
        result[layer_name] = {
            'variance': np.mean(variances),
            'variance_of_variance': np.std(variances)
        }
        
    return result, activations'''

class ActivationHook:
    def __init__(self, name, activations):
        self.name = name
        self.activations = activations

    def __call__(self, module, input, output):
        self.activations[self.name].extend(list(torch.var(output, dim=1).to('cpu').detach().numpy()))

def compute_layer_variances_dense(model, test_loader, device='cpu', cnn=True):
    activations = {}
    hooks = []

    for name, layer in model.named_modules():
        if name:  
            activations[name] = []
            hook = ActivationHook(name, activations)
            hooks.append(layer.register_forward_hook(hook))

    # Run inference on the test set
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cnn:
                data = data.to(device)
            else:
                data, target = data.reshape([data.shape[0], -1]).to(device), target.to(device)
            outputs = model(data)
            i += 1

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Calculate variance for each layer's activations
    result = {}
    for layer_name, variances in activations.items():
        result[layer_name] = {
            'variance': np.mean(variances),
            'variance_of_variance': np.std(variances)
        }
        
    return result, activations

def compute_weight_variances(model):
    weight_stats = {}
    
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            # Extract the weights as a numpy array
            weights = layer.weight.detach().cpu().numpy()
            
            # Calculate the variance of the weights for this layer
            weight_mean = np.mean(weights)
            weight_var = np.var(weights)
            
            # Calculate the variance of the variance for this layer
            weight_var_of_var = np.std(np.var(weights, axis=0))
            
            # Store these statistics
            weight_stats[name] = {
                'mean' : weight_mean,
                'variance': weight_var,
                'variance_of_variance': weight_var_of_var
            }
            
    return weight_stats

# Example usage:
# Assuming `model` is your PyTorch model
# weight_variances = compute_weight_variances_dense(model)
# print(weight_variances)
