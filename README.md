# Exploring Layer Freezing in Transfer Learning

Masters Project repository of team- Arnisa, David and Tushar. See our report (link) for the details of experiment setup and the results.

## Project Supervision

**Project Supervisor:**  
- Prof. Dr. Manuel Günther  

**Co-Supervisors:**  
- Prof. Dr. Benjamin Grewe  
- Pau Vilimelis Aceituno  

## Abstract

This project investigates the effectiveness of transfer learning by exploring various fine-tuning design choices, including freezing earlier layers, reinitializing or truncating later layers, and determining the subset of layers of the pre-trained model to retain during fine-tuning. The study explores optimal strategies for adapting pre-trained models to new tasks across the MNIST, FashionMNIST and CIFAR10 datasets. We conduct systematic experiments across these datasets in which we vary the amount of target data and number of layers to transfer. Experiments reveal a pattern consistent across mutiple source and target dataset combinations, where the optimal number of pre-trained layers to retain is reduced with increasing target data, indicating a reduced reliance on pre-trained features as more task-specific data becomes available. Key findings also highlight that completely removing some of the final layers offers a viable, more parameter-efficient alternative. Additionally, we propose a clustering based approach to identify layers that contain more transferable features. This study offers a framework for making informed fine-tuning decisions, emphasizing the importance of design choices in transfer learning.


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

## The Experiments

The experiments uses a pre-trained model to fine-tune multiple models on the target data, per 
1. different target data percentages: [0.1%, 1%, 10%, 50%, 100%]
2. different cut points: it freezes the layers until the cut point, and re-initializes/truncates the rest

| Cuts               | CNN layer 1                | CNN layer 2                | CNN layer 3                | Final dense layer             |
|--------------------|----------------------------|----------------------------|----------------------------|--------------------------------|
| cut=-1 (baseline) | reinitialized, trainable  | reinitialized, trainable  | reinitialized, trainable  | reinitialized, trainable     |
| cut=0              | reinitialized, trainable  | reinitialized, trainable  | reinitialized, trainable  | reinitialized, trainable     |
| cut=1              | frozen                     | reinitialized, trainable  | reinitialized, trainable  | reinitialized, trainable     |
| cut=2              | frozen                     | frozen                     | reinitialized, trainable  | reinitialized, trainable     |
| cut=3              | frozen                     | frozen                     | frozen                     | reinitialized, trainable     |

We use a very specific terminology to name the experiments, pre-trained models and results files. Here is some terminology to understand all the file and folder names:

| Experiment Setting                                     | Layers before the cut               | Layers after the cut               | Final dense layer(s)           | Short name |
|--------------------------------------------------------|-------------------------------------|------------------------------------|--------------------------------|------------|
| *freeze=True* and *reinit=True*                        | not reinitialized, not trainable  | reinitialized, trainable         | replaced and reinitialized    | *FR*       |
| *freeze=True* and *reinit=False*                       | not reinitialized, not trainable  | not reinitialized, trainable      | replaced and reinitialized    | *F*        |
| *freeze=False* and *reinit=True*                       | not reinitialized, trainable      | reinitialized, trainable         | replaced and reinitialized    | *R*        |
| *freeze=False* and *reinit=False*                      | not reinitialized, trainable      | not reinitialized, trainable      | replaced and reinitialized    | --         |
| *freeze=True* and *truncate=True*                      | not reinitialized, not trainable  | removed                            | replaced and reinitialized    | *FT*       |

For example, *FashionMNIST regular classes FR* setting stands for: 
1. regular classes: We pre-train on a specific subset of the FashionMNIST (all the classes other than footwear or bag), and fine-tune on a specific subset (footwear-related classes or bag)
2. FR: we freeze the layers before the cut and re-initialize the ones after the cut. We always replace the final fully-connected layer to match the output shape of the target dataset

And then, we also have the clustering experiments, where we use the feature maps from each convolutional layer to train some clustering algorithms! We call them *MNIST regular classes cluster*: Training cluster experiments on the regular splits of MNIST.

### How to control the parameters:
This is the model architecture, but modify it in the params dictionary:

![architecture](/images/architecture.png)

Below are the default parameters, change them in the notebooks!

    params = {
      # MODEL ARCHITECTURE PARAMS
      'depth': 6,
      'num_channels': 64, # num channels for CNN
      'activation_function': nn.ReLU,
      'kernel_size': 3,
      'use_pooling': True,  
      'pooling_every_n_layers': 2, # add pooling after every n layers specified here. For only one pooling after all the CNN layers, this equals params['depth']
      'pooling_stride': 2,

      # TRAINING PARAMS
      'device': device,
      'lr_pretrain': 0.001,   
      'lr_fine_tune': 0.001,
      'num_train': 40,
      'early_stop_patience': 6,
      'batch_size':64,

      # DATASET PARAMS: Set the pre-training and fine-tuning classes
      'pre_train_classes': [0, 1, 2, 3, 4],
      'fine_tune_classes': [5, 6, 7, 8, 9],
      
      # EXPERIMENT SETTING PARAMS
      'freeze': True,         # Freeze the layers before the cut!
      'reinit': True,         # Re-initialize the layers after the cut!
      'truncate': False,      # Remove the layers after the cut (except the fully-connected layer)
    }

Also see how we define the source and target datasets:

```python
    root_dir = './data' 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataloader_wrapped = TransferLearningWrapper(params, "datasets.MNIST", "datasets.CIFAR", root_dir, transform=transform)
```

## Results

For example, when we compare the FR and F experiments, these settings do not really affect the pre-trained model and baselines, so we re-use both.

FashionMNIST regular classes FR                      |  FashionMNIST regular classes F
:---------------------------------------------------:|:-----------------------------------------:
![fashion-regular-fr](/images/fashion-regular-fr.png) | ![fashion-regular-f](/images/fashion-regular-f.png)


## Output file structure

The outputs of the empirical experiments include the performance metrics (train, test accuracies on the target dataset) of model trained in different settings, and they are saved in json files. Note that we save the results of the baseline experiments (end-to-end models trained on subsets of fine-tuning dataset) and the regular fine-tuning experiments are saved in separate json files, whose names start with _baselines_ and _cuts_ respectively. We save the clustering experiment results in _regular\_cluster_.

The json files contain a list of dictionaries, where the first dictionary is a copy of the params dictionary: 

```
{"depth": 6, "num_channels": 64, "activation_function": "<class 'torch.nn.modules.activation.ReLU'>", "kernel_size": 5, "lr_pretrain": 0.001, "lr_fine_tune": 0.001, "num_train": 40, "early_stop_patience": 6, "save_best": false, "save_checkpoints": false, "is_cnn": true, "is_debug": false, "classification_report_flag": false, "batch_size": 4096, "pre_train_classes": [0, 1, 2, 3, 4], "fine_tune_classes": [5, 6, 7, 8, 9], "val_split": 0.1, "num_workers": 0, "generate_dataset_seed": 42, "percentages": [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1], "use_pooling": false, "freeze": true, "reinit": true, "reinit_both_dense": true}
```

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


## Directories overview

```
mp-tl-study
    ├── functions                               # the code-base
    |       ├── utils.py 
    |       ├── visualization_utils.py
    |       └── clustering_utils.py
    ├── MNIST
    |       ├── data (where the chosen dataset is downloaded)
    |       ├── pretrained_models
    |       |           └── regular_classes.pth
    |       ├── results
    |       |           ├── baselines_regular_classes.json
    |       |           └── cuts_regular_classes_FR.json
    |       ├── MNIST_regular_classes_FR.ipynb
    |       └── visualizations.ipynb
    |
    ├── FashionMNIST 
    |       ├── data
    |       ├── pretrained_models
    |       |           ├── random_classes.pth    # pre-trained on some other split of classes
    |       |           └── regular_classes.pth
    |       ├── results
    |       ├── FashionMNIST_regular_classes_FR.ipynb
    |       ├── FashionMNIST_regular_classes_FT.ipynb
    |       ├── FashionMNIST_regular_classes_cluster.ipynb
    |       ├── cluster_visualization.ipynb
    |       └── visualizations.ipynb
    |
    └── CIFAR, FashionMNIST_to_MNIST and MNIST_to_FashionMNIST folders 
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


3. Don't forget to change the path to the Github repo in each notebook
    ```
    sys.path.append('<PATH-TO-THE-REPO>/mp-tl-study')
    ```

## Experiment Hyperparameters {#hyperparameters}

Here are the parameters used in our experiments:

### Model Architecture Parameters

- `depth`: The depth of the model architecture.
- `num_channels`: The number of channels for the Convolutional Neural Network (CNN).
- `activation_function`: The activation function used in the model.
- `kernel_size`: The kernel size used in the CNN.
- `use_pooling`: Whether to use (average) pooling in the model.
- `pooling_every_n_layers`: The frequency of pooling layers insertion. Default: 1
- `pooling_stride`: The stride used in pooling. Default: 2

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

- `freeze`: Whether to freeze convolutional layers before the cut.
- `reinit`: Whether to reinitialize convolutional layers after the cut.
- `reinit_both_dense`: Whether to reinitialize both dense layers or only the last dense layer. Default: True
- `truncate`: Whether to remove the convolutional layers after the cut instead of reinitializing. Default: None
