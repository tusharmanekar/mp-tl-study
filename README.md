# mp-tl-study

"Transfer Learning Study" Masters Project repository of team- Arnisa, David and Tushar. See our report (link) for the details of experiment setup and the results.

The supervisors

## Introduction

Put our abstract here?

### Datasets

MNIST                      |  FashionMNIST            |  CIFAR10
:-------------------------:|:-------------------------:-------------------------:
![mnist_dataset](/images/mnist_dataset.png) | ![Fashion_dataset](/images/Fashion_dataset.png) | ![cifar_dataset](/images/cifar_dataset.png)


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

## Output file structure


## Empirical Results

box plot, explain the plot

some other visualizations.

!! See example_visualizations.ipynb for a preview of all possible visualizations.


## Clustering Results

image for the clustering visualization

box plot



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

## Experiment Parameters

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
