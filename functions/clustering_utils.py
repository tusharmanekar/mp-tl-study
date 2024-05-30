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
from scipy.optimize import linear_sum_assignment

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

        # x = x.view(-1, self._to_linear) # Flatten
        # x = self.fc(x)
        # outputs["fc"] = F.log_softmax(x, dim=1).cpu()
        return outputs

    def get_features_layers(self):
        names = []
        for layer_name, _ in self.named_children():
            if "conv" in layer_name:
                names.append(layer_name) 
        return names         

def extract_features_and_labels_gap(model, data_loader, device):
    model.to(device).eval() 
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
                    features_from_layers[name].append(pooled_feature)
    
    # Stack the features from all batches
    for layer in features_from_layers:
        if layer != "fc":
            features_from_layers[layer] = torch.cat(features_from_layers[layer], dim=0)

    return features_from_layers, torch.Tensor(labels_list)

def remove_zero_vectors(features, labels):
    non_zero_mask = np.linalg.norm(features, axis=1) != 0
    return features[non_zero_mask], labels[non_zero_mask]

def pairwise_precision_recall(labels, cluster_labels):
    true_same_cluster = (labels[:, None] == labels[None, :])
    pred_same_cluster = (cluster_labels[:, None] == cluster_labels[None, :])

    # Calculate true positives, false positives, false negatives, and true negatives
    true_positives = np.sum(true_same_cluster & pred_same_cluster) // 2
    false_positives = np.sum(~true_same_cluster & pred_same_cluster) // 2
    false_negatives = np.sum(true_same_cluster & ~pred_same_cluster) // 2
    true_negatives = np.sum(~true_same_cluster & ~pred_same_cluster) // 2

    # Calculate pairwise precision and recall
    pairwise_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    pairwise_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return pairwise_precision, pairwise_recall

def clustering_metrics(labels, cluster_labels):
    accuracy = metrics.accuracy_score(labels, cluster_labels)
    
    # Calculate pairwise precision and recall
    pairwise_precision, pairwise_recall = pairwise_precision_recall(labels, cluster_labels)
    
    return accuracy, pairwise_precision, pairwise_recall

def calculate_ppr_score(train_features, train_labels, apply_tsne=None):
    # Reshape features to 2D array
    train_features_2d = train_features.view(train_features.size(0), -1).numpy()
    # Remove zero vectors from train and test features
    train_features_2d, train_labels = remove_zero_vectors(train_features_2d, train_labels)

    if apply_tsne:
        tsne = TSNE(n_components=apply_tsne, random_state=42)
        train_features_2d = tsne.fit_transform(train_features_2d)

    # Agglomerative Clustering with cosine distance on test features
    clustering = AgglomerativeClustering(n_clusters=np.unique(train_labels).shape[0], affinity='cosine', linkage='average')
    cluster_labels = clustering.fit_predict(train_features_2d)

    # Calculate pairwise positioning recall
    # ppr = calculate_pairwise_precision(test_labels, cluster_labels)
    # return ppr
    scores = optimal_mapping_metrics(train_labels.cpu().numpy(), np.array(cluster_labels))
    return scores, metrics.adjusted_rand_score(train_labels.cpu().numpy(), np.array(cluster_labels))

def optimal_mapping_metrics(true_labels, cluster_labels):
    unique_true_labels = np.unique(true_labels)
    unique_cluster_labels = np.unique(cluster_labels)

    cost_matrix = np.zeros((len(unique_true_labels), len(unique_cluster_labels)))

    for i, true_label in enumerate(unique_true_labels):
        for j, cluster_label in enumerate(unique_cluster_labels):
            cost_matrix[i, j] = np.sum((true_labels == true_label) & (cluster_labels == cluster_label))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    mapped_cluster_labels = np.zeros_like(cluster_labels)
    for i, cluster_label in enumerate(unique_cluster_labels):
        mapped_cluster_labels[cluster_labels == cluster_label] = unique_true_labels[col_ind[i]]

    return clustering_metrics(true_labels, mapped_cluster_labels)

def get_scores(model, dataloader_train, device, apply_tsne=None):
    # Extract features and labels using the feature extractor model and the test_loader
    extracted_features_train, train_labels = extract_features_and_labels_gap(model, dataloader_train, device)

    sampled_labels_train = train_labels

    layer_names = extracted_features_train.keys()
    results = {}

    for layer_name in layer_names:
        if  layer_name.startswith("conv"):
            sampled_features_train = extracted_features_train[layer_name]
            # ppr = calculate_ppr_score(sampled_features_train, sampled_labels_train, 
            #                             sampled_features_test, sampled_labels_test, 
            #                             apply_tsne=apply_tsne)*100
            (acc, precision, recall), ari = calculate_ppr_score(sampled_features_train, sampled_labels_train, 
                                        apply_tsne=apply_tsne)
            results[layer_name] = {'accuracy':acc, 'pairwise_precision':precision,'pairwise_recall':recall, 'ari':ari}
    return results
