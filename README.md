# mp-tl-study

"Transfer Learning Study" Masters Project repository of team- Arnisa, David and Tushar. See our report (link) for the details of experiment setup and the results.

The supervisors

## Introduction

Put our abstract here?

## Datasets

The experiments in this repository use the following datasets:

- **MNIST Handwritten Digits Dataset**
  - **Description:** This dataset contains 28x28 pixel images of handwritten digits (0-9).
  - **Source:** [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
  - **Citation:** L. Deng, “The mnist database of handwritten digit images for machine learning research,” IEEE Signal Processing Magazine 29, 141–142 (2012)
  - **License:** This dataset is released under the [Modified BSD License](http://yann.lecun.com/exdb/mnist/).

- **FashionMNIST Dataset**
  - **Description:** FashionMNIST is a dataset of Zalando's article images, consisting of 28x28 grayscale images of 10 fashion categories.
  - **Source:** [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
  - **Citation:** H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms,” (2017).
  - **License:** This dataset is released under the [MIT License](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

- **CIFAR-10 Dataset**
  - **Description:** This dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
  - **Source:** [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
  - **Citation:** A. Krizhevsky, “Learning multiple layers of features from tiny images,” (2009).
  - **License:** This dataset is freely available for research purposes.

MNIST                      |  FashionMNIST            |  CIFAR-10
:-------------------------:|:------------------------:|:-------------------------:
![mnist_dataset](/images/mnist_dataset.png) | ![Fashion_dataset](/images/Fashion_dataset.png) | ![cifar_dataset](/images/cifar_dataset.png)

## Experimental Details {#exp-details}

Reinit, freeze etc

Baselines vs. cuts

## Empirical Experiments

box plot, explain the plot

![cifar_regular_classes](/images/cifar_regular_classes.jpeg)

some other visualizations.

!! See example_visualizations.ipynb for a preview of all possible visualizations.

## Clustering Experiments

In this experiment we extract hidden representations for the fine-tuning dataset samples from the pretrained model, and we use the hidden representations for training a clustering algorithm (K-means) and calculating the performance of the clustering (using ARI score).

Steps of the method:

1. We start with a pretrained model
2. Feed in a subset of the fine-tuning dataset into the pretrained model
3. Extract the hidden activations after each convolutional layer (note that we have multiple channels per each layer), in addition to inputs (the flattened input images themselves), and the output of the final dense layer.
4. Train K-means clustering algorithms for clustering the fine-tuning samples, for each channel of each layer, and for each percentage of data (similar to our default empirical experiment). 
5. For each of the clustering, measure the performance of quality of the clusters  using Adjusted Rand Index (ARI) metric.
6. We repeat above steps many times (20 times for less data and 5 for more data, same as the default empirical experiment) and report the ARI scores in box plots.

image for the clustering visualization

Evaluated on Pretrain Data |  Evaluted on Fine-tune Data
:-------------------------:|:------------------------:
![mnist_clustering_conv2_pretrain_test_data](/images/mnist_clustering_conv2_pretrain_test_data.png) | ![mnist_clustering_conv2_finetune_test_data](/images/mnist_clustering_conv2_finetune_test_data.png)

We also repeat the clustering experiments multiple times (on multiple subsets of the pretrain/fine-tuning data) and report the results in a box-plot format similar to the empirical experiments, so that it is easier to compare the two. Here is an example results of a clustering experiment:

![fashion_clustering](/images/fashion_clustering.png)

## Output file structure

The outputs of the empirical experiments represent the performance metrics of model trained in different settings, and they are saved in json files. Note that we save the results of the baseline experiments (end-to-end models trained on subsets of fine-tuning dataset) and the regular fine-tuning experiments are saved in separate json files, whose names start with _baselines_ and _results_ respectively. See our see the [experimental details](#exp-details) for more details into the experiment setting. \

The json files contain a list of dictionaries, where the first dictionary is a copy of the params dictionary: 

```
{"depth": 3, "num_channels": 10, "hidden_dim_lin": 128, "activation_function": "<class 'torch.nn.modules.activation.ReLU'>", "kernel_size": 5, "lr_pretrain": 0.001, "lr_fine_tune": 0.001, "num_train": 40, "early_stop_patience": 6, "save_best": false, "save_checkpoints": false, "is_cnn": true, "is_debug": false, "classification_report_flag": false, "batch_size": 4096, "pre_train_classes": [0, 1, 2, 3, 4], "fine_tune_classes": [5, 6, 7, 8, 9], "val_split": 0.1, "num_workers": 0, "generate_dataset_seed": 42, "percentages": [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1], "use_pooling": false, "freeze": true, "reinit": true, "reinit_both_dense": true}
```

See the [Experiment Hyperparameters](#hyperparameters) section for more details about all possible hyperparameters. \

The rest of the elements are the results for each single experiment, specifying the specific experiment parameters and the performance metrics of the trained model on the fine-tune datasets. Here is an example output: 

```
{"lr": 0.001, "sampled_percentage": 0.001, "sampled_cut_point": 2, "repeat": 2, "train_acc": 1.0, "test_acc": 0.7683604196667353}
```

More detailed explanation of the keys:

- `lr`: Learning rate used for fine-tuning or end-to-end training of the baseline model
- `sampled_percentage`: The percentage of the fine-tuning dataset used for fine-tuning or end-to-end training of the baseline model
- `sampled_cut_point`: The cut point for the fine-tuning, see the [experimental details](#exp-details) for more details.
- `repeat`: We repeat the same experiment setting multiple times, to be used in statistical significance tests later. This key specifies the repeat index.
- `train_acc`: Accuracy of the fine-tuned or the baseline model on the train set of the fine-tune dataset.
- `test_acc`: Accuracy of the fine-tuned or the baseline model on the test set of the fine-tune dataset.


## Overview directories

```
mp-tl-study
    ├── functions
    |       ├── utils.py 
    |       ├── visualization_utils.py
    |       └── clustering_utils.py
    ├── MNIST
    |       ├── data (where the chosen dataset is downloaded)
    |       ├── pretrained_models
    |       |           └── example_pretrained_model
    |       |                   ├── params.json
    |       |                   └── pretrained_model.pth
    |       ├── results
    |       |           ├── baselines_freeze_True_pool_False.json (example baseline results)
    |       |           ├── results_freeze_True_pool_False.json (example fine-tuning results with various cuts)
    |       |           └── ari_scores (this folder contains the results of the clustering experiments in json format)
    |       ├── empirical_experiment.ipynb
    |       ├── truncation_experiment.ipynb
    |       ├── clustering_experiment.ipynb
    |       └── visualizations.ipynb
    |
    ├── FashionMNIST 
    |       ├── data
    |       ├── pretrained_models
    |       ├── results
    |       ├── empirical_experiment.ipynb
    |       ├── truncation_experiment.ipynb
    |       ├── clustering_experiment.ipynb
    |       └── visualizations.ipynb
    ├── CIFAR, FashionMNIST_to_MNIST and MNIST_to_FashionMNIST folders 
    └── example_visualizations.ipynb
```

## Setup Instructions

1. Clone the repository:

    ```
    git clone https://github.com/tusharmanekar/mp-tl-study.git
    cd mp-tl-study
    ```

2. Using conda environment:

    ```
    conda env create -f environment.yml
    conda activate mp-env
    ```

    Select mp-env as the kernel for running the notebooks

3. Install project dependencies from the requirements file:

    ```
    pip install -r requirements.txt
    ```

4. Don't forget to change the path to the Github repo in each notebook
    ```
    sys.path.append('<PATH-TO-THE-REPO>/mp-tl-study')
    ```

## Experiment Hyperparameters {#hyperparameters}

Here are the parameters used in our experiments:

### Model Architecture Parameters

- `depth`: The depth of the model architecture.
- `num_channels`: The number of channels for the Convolutional Neural Network (CNN).
- `two_linear_layers`: Whether to include two linear layers in the model. Default: None
- `hidden_dim_lin`: The hidden dimension size if `two_linear_layers` is True.
- `activation_function`: The activation function used in the model.
- `kernel_size`: The kernel size used in the CNN.

### Training Parameters

- `device`: The device (CPU or GPU) used for training.
- `lr_pretrain`: The learning rate for pre-training.
- `lr_fine_tune`: The learning rate for fine-tuning.
- `num_train`: The number of training iterations.
- `early_stop_patience`: The patience for early stopping.
- `save_best`: Whether to save the best model. When it is False the training still returns the best model, but does not save it. Default: None
- `save_checkpoints`: Whether to save checkpoints during training. Default: False
- `is_cnn`: Whether the model is a CNN. Default: True
- `is_debug`: Whether to enable debug mode. Default: False
- `classification_report_flag`: Whether to enable classification report generation. Default: False
- `batch_size`: The batch size used for training.

### Dataset Parameters

- `pre_train_classes`: The classes used for pre-training.
- `fine_tune_classes`: The classes used for fine-tuning.
- `val_split`: The validation split ratio.
- `num_workers`: The number of workers for data loading. Default: 0
- `generate_dataset_seed`: The seed used for dataset generation. Default: 42

### Experiment Setting Parameters

- `use_pooling`: Whether to use pooling in the model.
- `pooling_every_n_layers`: The frequency of pooling layers insertion. Default: 1
- `pooling_stride`: The stride used in pooling. Default: 2
- `freeze`: Whether to freeze convolutional layers before the cut.
- `reinit`: Whether to reinitialize convolutional layers after the cut.
- `reinit_both_dense`: Whether to reinitialize both dense layers or only the last dense layer. Default: True
- `truncate`: Whether to remove the convolutional layers after the cut instead of reinitializing. Default: None
