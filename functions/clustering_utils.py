from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, params, output_dim, input_shape=(1, 28, 28)):
        super(CNNFeatureExtractor, self).__init__()
        self.params = params
        self.input_shape = input_shape
        in_channels = input_shape[0]

        # Dynamically add convolutional and activation layers based on the specified depth
        for i in range(params["depth"]):
            # Create a convolutional layer and add it to the model
            setattr(self, f"conv{i}", nn.Conv2d(in_channels, params["num_channels"], kernel_size=params["kernel_size"], padding=math.floor(params["kernel_size"]/2)))

            # Create an activation layer (e.g., ReLU) and add it to the model
            setattr(self, f"act{i}", params["activation_function"]())

            # Optionally add pooling layers to reduce spatial dimensions
            if params["use_pooling"] and (i+1) % params["pooling_every_n_layers"] == 0:
                setattr(self, f"pool{i}", nn.AvgPool2d(2, stride=params['pooling_stride']))

            # Update the input channels for the next convolutional layer
            in_channels = params["num_channels"]

        # Compute the size of the flattened features for the fully connected layer
        self.calculate_to_linear_size()

        # Add one fully connected layers for classification
        self.fc = nn.Linear(self._to_linear, output_dim)

    # calculate the input dimensions to the fully-connecting layer by forwarding a dummy input
    def calculate_to_linear_size(self):
        x = torch.zeros((1,) + self.input_shape)
        for layer_name, layer in self.named_children():
            # Process the input tensor through convolutional and activation layers
            if "conv" in layer_name or "act" in layer_name:
                x = layer(x)
            # Process the input tensor through pooling layers if they exist
            elif "pool" in layer_name:
                x = layer(x)
            # If reached fully connected layers, break the loop
            elif isinstance(layer, nn.Linear):
                break
        self._to_linear = x.view(x.size(0), -1).size(1)

    def forward(self, x):
        outputs = {}
        # outputs["input"] = x.cpu()
        # Iterate over each module in the CustomCNN class
        for layer_name, layer in self.named_children():
            # Process the input tensor through convolutional and activation layers
            if "conv" in layer_name or "act" in layer_name:
                x = layer(x)
            # Process the input tensor through pooling layers if they exist
            elif "pool" in layer_name:
                x = layer(x)
            # If reached fully connected layers, break the loop
            elif isinstance(layer, nn.Linear):
                break
            if "conv" in layer_name or "fc" in layer_name:
                outputs[layer_name] = x.cpu()

        x = x.view(-1, self._to_linear) # Flatten
        x = self.fc(x)
        # outputs["fc"] = F.log_softmax(x, dim=1).cpu()
        return outputs

    def get_features_layers(self):
        names = []
        for layer_name, _ in self.named_children():
            if "conv" in layer_name:
                names.append(layer_name) 
        return names         
    
def extract_features_and_labels(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    features_from_layers = {layer: [] for layer in model.get_features_layers()}
    features_from_layers["input"] = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            labels_list.extend(labels.numpy())

            images = images.to(device)

            # Extract features
            out_features = model(images)

            # Store features from each layer
            for name, feature in out_features.items():
                features_from_layers[name].append(feature)
    
    # Concatenate the features from all batches
    for layer in features_from_layers:
        features_from_layers[layer] = torch.cat(features_from_layers[layer], 0)

    return features_from_layers, torch.Tensor(labels_list)

def extract_features_and_labels_gap(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    features_from_layers = {layer: [] for layer in model.get_features_layers()}
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            labels_list.extend(labels.numpy())

            images = images.to(device)

            # Extract features
            out_features = model(images)

            # Perform global average pooling over spatial dimensions for each layer
            for name, feature in out_features.items():
                if name != "fc":
                    # Determine the spatial dimensions
                    spatial_dims = tuple(range(2, len(feature.shape)))
                    # Perform global average pooling over spatial dimensions
                    pooled_feature = torch.mean(feature, dim=spatial_dims)
                    # print(pooled_feature.shape)
                    features_from_layers[name].append(pooled_feature)
                    # print(name, pooled_feature.shape)
    
    # Stack the features from all batches
    for layer in features_from_layers:
        if layer != "fc":
            # print(features_from_layers[layer])
            features_from_layers[layer] = torch.cat(features_from_layers[layer], dim=0)

    return features_from_layers, torch.Tensor(labels_list)

def visualize_tsne(features, labels, layer_name, title):
    # Reshape features to 2D array
    features_2d = features.view(features.size(0), -1).numpy()

    # Apply t-SNE to reduce the dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features_2d)

    # Plot t-SNE visualization for the layer
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    plt.title(f't-SNE Visualization of Extracted Features ({layer_name} - {title})')
    plt.colorbar()
    plt.show()

def calculate_ARI_score(features, labels, apply_pca=None):
    # Reshape features to 2D array
    features_2d = features.view(features.size(0), -1).numpy()
    if apply_pca:
        pca = PCA(n_components=apply_pca)
        features_2d = pca.fit_transform(features_2d)

    # KMeans clustering labels
    kmeans = KMeans(n_clusters=np.unique(labels).shape[0], random_state=42)
    cluster_labels = kmeans.fit_predict(features_2d)

    # ARI 
    ari = metrics.adjusted_rand_score(labels, cluster_labels)
    return ari

# Dataloader can be already subsampled using our subsampling function (reduce_dataset)
# If yes, we can still use num_samples to randomly sample from that subset
# or if num_samples = 0: take all the samples
def get_ARI_scores(model, dataloader, num_samples, channel_ids, device, apply_pca=None):
    # Extract features and labels using the feature extractor model and the test_loader
    extracted_features, labels = extract_features_and_labels(model, dataloader, device)

    dim_size = labels.size(0)

    if num_samples > 0:
        random_indices = torch.randperm(dim_size)[:num_samples]
        sampled_labels = torch.index_select(labels, 0, random_indices)
    else:
        sampled_labels = labels

    layer_names = extracted_features.keys()
    results = {}

    for layer_name in layer_names:
        results_channels = {}
        for channel_id in channel_ids:
            if channel_id < extracted_features[layer_name].shape[1]:
                if num_samples > 0:
                    sampled_features = torch.index_select(extracted_features[layer_name], 0, random_indices)
                    layer_features = sampled_features[:,channel_id]
                else:
                    sampled_features = extracted_features[layer_name]
                    layer_features = sampled_features[:,channel_id]
                ari = calculate_ARI_score(layer_features, sampled_labels, apply_pca=apply_pca)*100
                results_channels[channel_id] = ari
        results[layer_name] = results_channels
    return results

def calculate_pairwise_positioning_recall(labels, cluster_labels):
    n = len(labels)
    labels = np.array(labels)
    cluster_labels = np.array(cluster_labels)
    
    true_pairs = np.sum((labels[:, None] == labels) & (np.arange(n)[:, None] != np.arange(n)))
    correct_pairs = np.sum((labels[:, None] == labels) & (cluster_labels[:, None] == cluster_labels) & (np.arange(n)[:, None] != np.arange(n)))

    if true_pairs == 0:
        return 0
    
    return correct_pairs / true_pairs

def remove_zero_vectors(features, labels):
    non_zero_mask = np.linalg.norm(features, axis=1) != 0
    # print(non_zero_mask.shape, features.shape, labels.shape)
    return features[non_zero_mask], labels[non_zero_mask]

def calculate_ppr_score(train_features, train_labels, test_features, test_labels, apply_tsne=None):
    # Reshape features to 2D array
    train_features_2d = train_features.view(train_features.size(0), -1).numpy()
    test_features_2d = test_features.view(test_features.size(0), -1).numpy()
    # Remove zero vectors from train and test features
    train_features_2d, train_labels = remove_zero_vectors(train_features_2d, train_labels)
    test_features_2d, test_labels = remove_zero_vectors(test_features_2d, test_labels)

    if apply_tsne:
        tsne = TSNE(n_components=apply_tsne, random_state=42)
        train_features_2d = tsne.fit_transform(train_features_2d)
        test_features_2d = tsne.fit_transform(test_features_2d)

    # Agglomerative Clustering with cosine distance on test features
    clustering = AgglomerativeClustering(n_clusters=np.unique(train_labels).shape[0], affinity='cosine', linkage='average')
    cluster_labels = clustering.fit_predict(test_features_2d)

    # Calculate pairwise positioning recall
    ppr = calculate_pairwise_positioning_recall(test_labels, cluster_labels)
    return ppr

def get_PPR_scores(model, dataloader_train, dataloader_test, num_samples, device, apply_tsne=None):
    # Extract features and labels using the feature extractor model and the test_loader
    extracted_features_train, train_labels = extract_features_and_labels_gap(model, dataloader_train, device)
    extracted_features_test, test_labels = extract_features_and_labels_gap(model, dataloader_test, device)
    # print(extracted_features_train["conv0"].shape, train_labels.shape)
    dim_size = train_labels.size(0)

    if num_samples > 0:
        random_indices = torch.randperm(dim_size)[:num_samples]
        sampled_labels_train = torch.index_select(train_labels, 0, random_indices)
        random_indices = torch.randperm(dim_size)[:num_samples]
        sampled_labels_test = torch.index_select(train_labels, 0, random_indices)
    else:
        sampled_labels_train = train_labels
        sampled_labels_test = test_labels

    layer_names = extracted_features_train.keys()
    # print(layer_names)
    results = {}

    for layer_name in layer_names:
        if  layer_name != "fc":
            if num_samples > 0:
                random_indices = torch.randperm(num_samples)[:num_samples]

                sampled_features_train = extracted_features_train[layer_name][random_indices]
                sampled_features_test = extracted_features_test[layer_name][random_indices]
            else:
                # If num_samples is 0, use all features without sampling
                sampled_features_train = extracted_features_train[layer_name]
                sampled_features_test = extracted_features_test[layer_name]
            ppr = calculate_ppr_score(sampled_features_train, sampled_labels_train, 
                                        sampled_features_test, sampled_labels_test, 
                                        apply_tsne=apply_tsne)*100
            results[layer_name] = ppr
    return results
