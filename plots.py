import matplotlib.pyplot as plt
import torch

SEED = 42
import numpy as np
np.random.seed(SEED)

# this function needs some updating if we want to use it
# Plot activation variances of the activation layer and linear activations together, 
# one plot with range and one without (they both have variance of variances)
def plot_variances(variances, results):
    # Create a list of layer names and variances
    layer_names = list(variances.keys())
    variance_values = [variance.mean().item() for variance in variances.values()]
    variance_of_variance_values = [results[layer]['variance_of_variance'] for layer in layer_names]

    # Plot the variances
    plt.figure(figsize=(10, 5))
    plt.plot(layer_names, variance_values, label='Variance')
    plt.plot(layer_names, variance_of_variance_values, label='Variance of Variance', linestyle='--')
    plt.xticks(rotation=90)
    plt.xlabel('Layer Name')
    plt.ylabel('Value')
    plt.title('Activation Variances and Variance of Variances')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Create lists for mean, min, and max variance for each layer
    layer_names = list(variances.keys())
    mean_variances = [variance.mean().item() for variance in variances.values()]
    min_variances = [variance.min().item() for variance in variances.values()]
    max_variances = [variance.max().item() for variance in variances.values()]

    # Plot the variances
    plt.figure(figsize=(10, 5))
    plt.plot(layer_names, mean_variances, label='Mean Variance')
    plt.plot(layer_names, variance_of_variance_values, label='Variance of Variance', linestyle='--')

    # Shade the region between the min and max variance
    plt.fill_between(layer_names , min_variances , max_variances , color='gray', alpha=0.5, label='Variance Range')

    plt.xticks(rotation=90)
    plt.xlabel('Layer Name')
    plt.ylabel('Variance')
    plt.title('Activation Variances, Variance of Variances, and Variance Range')
    plt.tight_layout()
    plt.legend()
    plt.show()

# HELPER FUNCTION: metrics.compute_layer_variances_dense
def plot_variances_by_layer_type(variances, results, cnn=True, ignore_final_layer=False, std_of_variance=False, save_path=None):
    # Create a list of layer names and variances for fc layers
    if cnn:
        layer_type = 'conv'
    else:
        layer_type = 'lin'
        
    conv_layer_names = [name for name in variances.keys() if layer_type in name]
    conv_variance_values = [np.mean(variance) for name, variance in variances.items() if layer_type in name]

    # Create a list of layer names and variances for activation layers
    activation_layer_names = [name for name in variances.keys() if 'act' in name]
    activation_variance_values = [np.mean(variance) for name, variance in variances.items() if 'act' in name]

    # Extract variance of variance values
    conv_variance_of_variance_values = [results[name]['variance_of_variance'] for name in conv_layer_names]
    activation_variance_of_variance_values = [results[name]['variance_of_variance'] for name in activation_layer_names]

    # Ignore the final layer if specified
    if ignore_final_layer:
        conv_layer_names = conv_layer_names[:-1]
        conv_variance_values = conv_variance_values[:-1]
        conv_variance_of_variance_values = conv_variance_of_variance_values[:-1]

        activation_layer_names = activation_layer_names[:-1]
        activation_variance_values = activation_variance_values[:-1]
        activation_variance_of_variance_values = activation_variance_of_variance_values[:-1]

    # Create lists for mean, min, and max variance for each type of layer
    def extract_variances_from_names(names, variances):
        mean_vals = [np.mean(variances[name]) for name in names]
        return mean_vals

    conv_mean = extract_variances_from_names(conv_layer_names, variances)
    activation_mean = extract_variances_from_names(activation_layer_names, variances)

    def extract_variance_of_variance_from_names(names, results):
        return [results[name]['variance_of_variance'] for name in names]

    conv_vov = extract_variance_of_variance_from_names(conv_layer_names, results)
    activation_vov = extract_variance_of_variance_from_names(activation_layer_names, results)

    # Create a figure with 2 subplots
    if std_of_variance:
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the variances and variance of variances for conv layers
    axs[0].plot(conv_layer_names, conv_variance_values, label='Variance')
    if std_of_variance:
        axs[0].plot(conv_layer_names, conv_variance_of_variance_values, linestyle='--', label='Std. of Variance')
    axs[0].set_xticklabels(conv_layer_names, rotation=90)
    axs[0].set_xlabel('Layer Name')
    axs[0].set_ylabel('Value')
    axs[0].set_title('{} Layer Variances'.format(layer_type.upper()))
    axs[0].legend()

    # Plot the variances and variance of variances for activation layers
    axs[1].plot(activation_layer_names, activation_variance_values, label='Variance')
    if std_of_variance:
        axs[1].plot(activation_layer_names, activation_variance_of_variance_values, linestyle='--', label='Variance of Variance')
    axs[1].set_xticklabels(activation_layer_names, rotation=90)
    axs[1].set_xlabel('Layer Name')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Activation Layer Variances')
    axs[1].legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# HELPER FUNCTION: metrics.compute_layer_variances_dense
def plot_variances_ranges_by_layer_type(variances, results, cnn=True, ignore_final_layer=False, std_of_variance=True, save_path=None):
    # Create a list of layer names and variances for fc layers
    if cnn:
        layer_type = 'conv'
    else:
        layer_type = 'lin'
        
    conv_layer_names = [name for name in variances.keys() if layer_type in name]
    conv_variance_values = [np.mean(variance) for name, variance in variances.items() if layer_type in name]

    # Create a list of layer names and variances for activation layers
    activation_layer_names = [name for name in variances.keys() if 'act' in name]
    activation_variance_values = [np.mean(variance) for name, variance in variances.items() if 'act' in name]

    # Extract variance of variance values
    conv_variance_of_variance_values = [results[name]['variance_of_variance'] for name in conv_layer_names]
    activation_variance_of_variance_values = [results[name]['variance_of_variance'] for name in activation_layer_names]

    # Ignore the final layer if specified
    if ignore_final_layer:
        conv_layer_names = conv_layer_names[:-1]
        conv_variance_values = conv_variance_values[:-1]
        conv_variance_of_variance_values = conv_variance_of_variance_values[:-1]

        activation_layer_names = activation_layer_names[:-1]
        activation_variance_values = activation_variance_values[:-1]
        activation_variance_of_variance_values = activation_variance_of_variance_values[:-1]

    # Create lists for mean, min, and max variance for each type of layer
    def extract_variances_from_names(names, variances):
        mean_vals = [np.mean(variances[name]) for name in names]
        min_vals = [np.min(variances[name]) for name in names]
        max_vals = [np.max(variances[name]) for name in names]
        return mean_vals, min_vals, max_vals

    conv_mean, conv_min, conv_max = extract_variances_from_names(conv_layer_names, variances)
    activation_mean, activation_min, activation_max = extract_variances_from_names(activation_layer_names, variances)

    def extract_variance_of_variance_from_names(names, results):
        return [results[name]['variance_of_variance'] for name in names]

    conv_vov = extract_variance_of_variance_from_names(conv_layer_names, results)
    activation_vov = extract_variance_of_variance_from_names(activation_layer_names, results)

    # Create a figure with four subplots
    if std_of_variance:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the variances and variance of variances for conv layers
    axs[0, 0].plot(conv_layer_names, conv_variance_values, label='Variance')
    if std_of_variance:
        axs[0, 0].plot(conv_layer_names, conv_variance_of_variance_values, linestyle='--', label='Variance of Variance')
    axs[0, 0].set_xticklabels(conv_layer_names, rotation=90)
    axs[0, 0].set_xlabel('Layer Name')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].set_title('{} Layer Variances'.format(layer_type.upper()))
    axs[0, 0].legend()

    # Plot the variances and variance of variances for activation layers
    axs[0, 1].plot(activation_layer_names, activation_variance_values, label='Variance')
    if std_of_variance:
        axs[0, 1].plot(activation_layer_names, activation_variance_of_variance_values, linestyle='--', label='Variance of Variance')
    axs[0, 1].set_xticklabels(activation_layer_names, rotation=90)
    axs[0, 1].set_xlabel('Layer Name')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].set_title('Activation Layer Variances')
    axs[0, 1].legend()

    # Plot the variances for conv layers
    axs[1, 0].plot(conv_layer_names, conv_mean, label='Mean Variance')
    if std_of_variance:
        axs[1, 0].plot(conv_layer_names, conv_vov, linestyle='--', label='Variance of Variance')
    axs[1, 0].fill_between(conv_layer_names, conv_min, conv_max, color='gray', alpha=0.5, label='Variance Range')
    axs[1, 0].set_xticklabels(conv_layer_names, rotation=90)
    axs[1, 0].set_xlabel('Layer Name')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('{} Layer Variances'.format(layer_type.upper()))
    axs[1, 0].legend()

    # Plot the variances for activation layers
    axs[1, 1].plot(activation_layer_names, activation_mean, label='Mean Variance')
    if std_of_variance:
        axs[1, 1].plot(activation_layer_names, activation_vov, linestyle='--', label='Variance of Variance')
    axs[1, 1].fill_between(activation_layer_names, activation_min, activation_max, color='gray', alpha=0.5, label='Variance Range')
    axs[1, 1].set_xticklabels(activation_layer_names, rotation=90)
    axs[1, 1].set_xlabel('Layer Name')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('Activation Layer Variances')
    axs[1, 1].legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# HELPER FUNCTION: metrics.compute_weight_variances
def plot_weight_variances(weight_variances, save_path=None):
    # Create lists of layer names, variances, and variances of variances
    layer_names = list(weight_variances.keys())
    variance_values = [weight_variances[layer]['variance'] for layer in layer_names]
    variance_of_variance_values = [weight_variances[layer]['variance_of_variance'] for layer in layer_names]
    
    # Plot the variances
    plt.figure(figsize=(10, 5))
    plt.plot(layer_names, variance_values, label='Variance')
    plt.plot(layer_names, variance_of_variance_values, label='Std. of Variance', linestyle='--')
    plt.xticks(rotation=90)
    plt.xlabel('Layer Name')
    plt.ylabel('Value')
    plt.title('Weight Variances and Std. of Variances')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_acc_vs_cut(finetuned_accs, cuts, ylabel="Finetuned Training Accuracy", save_path=None):
    plt.plot(cuts, finetuned_accs)
    plt.xlabel('Cut')
    plt.ylabel(ylabel)
    plt.title('{} vs Cut'.format(ylabel))
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# HELPER FUNCTION: utils.multiple_fine_tuning_experiments
def box_plot_multiple_exp(experiments, cuts, save_path=None):
    # Initialize lists to store all accuracies for each cut point
    train_accuracies = [[] for _ in cuts]
    test_accuracies = [[] for _ in cuts]

    # Populate the lists with accuracy values
    for cut_models in experiments:
        for cut, model_data in enumerate(cut_models):
            train_accuracies[cut].append(model_data['finetuned_acc'])
            test_accuracies[cut].append(model_data['finetuned_test_acc'])

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot training accuracies
    axs[0].boxplot(train_accuracies)
    axs[0].set_title('Train Accuracies across Cuts')
    axs[0].set_xlabel('Cut Point')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xticks(range(1,len(cuts)+1))
    axs[0].set_xticklabels(cuts)
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot test accuracies
    axs[1].boxplot(test_accuracies)
    axs[1].set_title('Test Accuracies across Cuts')
    axs[1].set_xlabel('Cut Point')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xticks(range(1,len(cuts)+1))
    axs[1].set_xticklabels(cuts)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the plots
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# not sure if it works with torch Dataloader yet
def plot_decision_boundary(model, X, y, n_classes, loader=None):
    if loader:
        X = loader.dataset.dataset.data.numpy()
        y = loader.dataset.dataset.targets.numpy()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    with torch.no_grad():
        model.eval()
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).numpy()
        Z = np.argmax(Z, axis=1)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', linewidth=1)
    plt.title("Decision Boundary")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    for i in range(n_classes):
        plt.scatter([], [], c='k', alpha=0.8, s=20, label=f"Class {i}")
    plt.legend(loc='upper left')

    plt.show()