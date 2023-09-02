import matplotlib.pyplot as plt

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

def plot_variances_by_layer_type(variances, results, cnn = True, ignore_final_layer=False):
    # Create a list of layer names and variances for fc layers
    if cnn:
        layer_type = 'conv'
    else:
        layer_type = 'lin'
        
    conv_layer_names = [name for name in variances.keys() if layer_type in name]
    conv_variance_values = [variance.mean().item() for name, variance in variances.items() if layer_type in name]

    # Create a list of layer names and variances for activation layers
    activation_layer_names = [name for name in variances.keys() if 'act' in name]
    activation_variance_values = [variance.mean().item() for name, variance in variances.items() if 'act' in name]

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
        mean_vals = [variances[name].mean().item() for name in names]
        min_vals = [variances[name].min().item() for name in names]
        max_vals = [variances[name].max().item() for name in names]
        return mean_vals, min_vals, max_vals

    conv_mean, conv_min, conv_max = extract_variances_from_names(conv_layer_names, variances)
    activation_mean, activation_min, activation_max = extract_variances_from_names(activation_layer_names, variances)

    def extract_variance_of_variance_from_names(names, results):
        return [results[name]['variance_of_variance'] for name in names]

    conv_vov = extract_variance_of_variance_from_names(conv_layer_names, results)
    activation_vov = extract_variance_of_variance_from_names(activation_layer_names, results)

    # Create a figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot the variances and variance of variances for conv layers
    axs[0, 0].plot(conv_layer_names, conv_variance_values, label='Variance')
    axs[0, 0].plot(conv_layer_names, conv_variance_of_variance_values, linestyle='--', label='Variance of Variance')
    axs[0, 0].set_xticklabels(conv_layer_names, rotation=90)
    axs[0, 0].set_xlabel('Layer Name')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].set_title('{} Layer Variances'.format(layer_type.upper()))
    axs[0, 0].legend()

    # Plot the variances and variance of variances for activation layers
    axs[0, 1].plot(activation_layer_names, activation_variance_values, label='Variance')
    axs[0, 1].plot(activation_layer_names, activation_variance_of_variance_values, linestyle='--', label='Variance of Variance')
    axs[0, 1].set_xticklabels(activation_layer_names, rotation=90)
    axs[0, 1].set_xlabel('Layer Name')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].set_title('Activation Layer Variances')
    axs[0, 1].legend()

    # Plot the variances for conv layers
    axs[1, 0].plot(conv_layer_names, conv_mean, label='Mean Variance')
    axs[1, 0].plot(conv_layer_names, conv_vov, linestyle='--', label='Variance of Variance')
    axs[1, 0].fill_between(conv_layer_names, conv_min, conv_max, color='gray', alpha=0.5, label='Variance Range')
    axs[1, 0].set_xticklabels(conv_layer_names, rotation=90)
    axs[1, 0].set_xlabel('Layer Name')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('{} Layer Variances'.format(layer_type.upper()))
    axs[1, 0].legend()

    # Plot the variances for activation layers
    axs[1, 1].plot(activation_layer_names, activation_mean, label='Mean Variance')
    axs[1, 1].plot(activation_layer_names, activation_vov, linestyle='--', label='Variance of Variance')
    axs[1, 1].fill_between(activation_layer_names, activation_min, activation_max, color='gray', alpha=0.5, label='Variance Range')
    axs[1, 1].set_xticklabels(activation_layer_names, rotation=90)
    axs[1, 1].set_xlabel('Layer Name')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('Activation Layer Variances')
    axs[1, 1].legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()