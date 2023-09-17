#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# EXAMPLE USAGE:
# 
#           python pipelined_experiment.py -d 10 -w 5 -a relu -f -o ../outputs/exp3
#
# This command means: depth=10, width=5, activation_function=relu, freeze_layers=True

# CHECK THE ARGPARSE ARGUMENTS!! most of them have defaults
# For bool arguments: for example -f means freeze_layer = True.
# -o is the output path, path to an empty folder (preferably not inside our github repo)

# ALSO set other hyperparmas for training such as epochs, learning rate etc. in the SETUP part

import argparse, os, sys, json, logging
from tqdm import tqdm
import pandas as pd
# logger = logging.getLogger('logger')
# logging.basicConfig(level=logging.INFO)

from utils import *
from data_utils import *
from metrics import *
from plots import *

import matplotlib
matplotlib.use('Agg')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ---------------------------------------------- HYPERPARAMS --------------------------------------------

# plot and save the variances for any model and given dataset propagated through the model
def plot_variances(model, loader, save_path=None):
    results, variances = compute_layer_variances_dense(model, loader, device=params['device'], cnn=args.cnn)
    plot_variances_by_layer_type(variances, results, cnn=args.cnn, ignore_final_layer=True, 
                                 std_of_variance=args.std_of_variances, save_path=save_path)

# if we can want we can do this later after saving and reloading the checkpoints
# otherwise need to change the training function
def plot_variances_epochs(model, loader, save_path):
    pass
        
if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--depth", default=20, type=int, help="network depth")
    parser.add_argument("-w", "--width", default=100, type=int, help="network width")
    parser.add_argument("-sw", "--sigma_w", default=1.7346938775510203, type=float, help="variance of weights")
    parser.add_argument("-sb", "--sigma_b", default=0.05, type=float, help="variance for weights")
    parser.add_argument("-a", "--activation_function", default='tanh', help="tanh or relu")
    parser.add_argument("-c", "--cnn", default=False, action='store_true', help="True for CNN and False for linear")
    parser.add_argument("-g", "--gaussian_init", default=False, action='store_true', help="True for Gaussian init and False for torch init")
    parser.add_argument("-std", "--std_of_variances", default=False, action='store_true', help="True for printing the std of variances too in the same graph")
    parser.add_argument("-num", "--num_experiments", default=1, type=int, help="Number of experiments to fine-tune different cuts to generate the box plots")
    parser.add_argument("-cuts", "--cuts_skip", default=1, type=int, help="For example if this is 10 then we fine-tune every 10th cut (0,10,20,..)")
    parser.add_argument("-f", "--freeze", default=False, action='store_true', help="Freeze the layers before the cut")
    parser.add_argument("-r", "--reinitialize", default=False, action='store_true', help="if true, we reinitialize the layers after the cut")
    parser.add_argument("-o", "--out_path", required=True, help="path to the output folder")
    
    args = parser.parse_args()
    
    if os.path.exists(args.out_path):
        print("ERROR: Folder already exists")
        exit()
    else:
        os.mkdir(args.out_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file = logging.FileHandler(os.path.join(args.out_path, "outputs.log"))
    file.setLevel(logging.INFO)
    fileformat = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s",datefmt="%H:%M:%S")
    file.setFormatter(fileformat)
    logger.addHandler(file)
        
    logger.info("ARGS: {}".format(args))
    # -------------------------------------------- SETUP -------------------------------------------
    print("SETUP")
    logger.info("\n\n Pretraining the model:")
    batch_size = 128
    lr=0.01
    num_train=10
    early_stop_patience = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    params = dict(device=device, batch_size=batch_size,
                    width=args.width, lr=lr, num_train=num_train,
                    sb=args.sigma_b, depth=args.depth, sw=args.sigma_w, 
                    early_stop_patience=early_stop_patience, activation_function=args.activation_function,
                    cnn=args.cnn)
    
    if params['activation_function'] == 'relu':
        activation_function = nn.ReLU
    elif params['activation_function'] == 'tanh':
        activation_function = nn.Tanh
    else:
        activation_function = nn.Tanh
    
    # --------------------------------------- 1. UNTRAINED MODEL AND PLOT ----------------------------------------
    print("1. UNTRAINED MODEL AND PLOT")
    # create the folder:
    folder = os.path.join(args.out_path, 'before_pretraining')
    os.mkdir(folder)
    
    print("save the untrained model")
    del params['device']
    with open(os.path.join(folder, "model_params.json"), "w+") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)  # encode dict into JSON
    params['device'] = device
    
    print("loading the dataset")
    dataset = TransferLearningMNIST(batch_size)
    dataset_wrapped = TransferLearningMNISTWrapper(dataset, 'pretrain')
    
    print("generate the model")
    if not args.cnn:
        model = generate_fc_dnn(dataset.input_dim, dataset.output_dim,
                params, activation_function=activation_function, gaussian_init=args.gaussian_init).to(device)
    else:
        model = generate_cnn(dataset.input_dim, dataset.output_dim, params['depth'], params['width'], act_fn=activation_function, use_pooling=False)
    
    print("saving the variances")
    plot_variances(model, dataset_wrapped.train_loader, os.path.join(folder, 'variances_on_pretraining_set.png'))
    # plot_variances(model, dataset_wrapped.train_loader)
    
    # --------------------------------- 2. PRETRAINED MODEL AND PLOTS ---------------------------------------
    print("2. PRETRAINED MODEL AND PLOTS")
    folder = os.path.join(args.out_path, 'after_pretraining')
    os.mkdir(folder)
    
    if not args.cnn:
        pretrain_train_acc, pretrain_test_acc, pre_trained_model, pretraining_checkpoints = compute_training_acc_epochs(model, dataset_wrapped, params, debug=True, save_checkpoints=False, return_checkpoints=True, logger=logger.info)
        test_acc = eval(pre_trained_model, device, dataset.pretrain_test_loader, debug=False, logger=logger.info)
        logger.info(f"Test Accuracy on training classes: {test_acc:.2f}")
    else:
        pass
    
    # Save model state dictionary
    torch.save(pre_trained_model.state_dict(), os.path.join(folder, "pretrained_model.pth"))
    torch.save(pretraining_checkpoints, os.path.join(folder, 'pretrained_model_checkpoints.pth'))

    # Save plots
    plot_variances(pre_trained_model, dataset_wrapped.train_loader, os.path.join(folder, 'variances_on_pretraining_set.png'))
    plot_variances(pre_trained_model, dataset.finetune_train_loader, os.path.join(folder, 'variances_on_finetuning_set.png'))

    # ----------------------------------- 3. FINE-TUNING ----------------------------------------------------------
    print("3. FINE-TUNING")
    folder = os.path.join(args.out_path, 'after_fine_tuning')
    os.mkdir(folder)
    
    dataset_wrapped.update_phase('finetune')

    num_experiments = args.num_experiments
    experiments = []

    for i in tqdm(range(num_experiments)):
        print('experiment number: ', i)
        cut_models = []
        for cut in tqdm(range(0, args.depth, args.cuts_skip)):
            temp = {}
            temp['cut_model'] = cut_model(pre_trained_model, cut_point=cut, freeze=args.freeze, reinitialize=args.reinitialize)
            logger.info("\n\nCut: {}".format(cut))
            finetuned_acc, finetuned_test_acc, finetuned_model, checkpoints_temp = compute_training_acc_epochs(temp['cut_model'], dataset_wrapped, params, debug=True, save_checkpoints=False, return_checkpoints=True, logger=logger.info)
            temp['finetuned_acc'] = finetuned_acc
            temp['finetuned_test_acc'] = finetuned_test_acc
            temp['finetuned_model'] = finetuned_model
            temp['checkpoints'] = checkpoints_temp
            cut_models.append(temp)  
        experiments.append(cut_models)
       
    print("Saving everything and finishing up")
    # Save all the fine-tuned models and their variance graphs
    if len(experiments) == 1:
        finetuned_train_accs = [model['finetuned_acc'] for model in cut_models]
        finetuned_test_accs = [model['finetuned_test_acc'] for model in cut_models]
        plot_acc_vs_cut(finetuned_train_accs, cuts=range(0, args.depth, args.cuts_skip), ylabel="Finetuned Train Accuracy", save_path=os.path.join(folder, 'train_acc_vs_cut.png'))
        plot_acc_vs_cut(finetuned_test_accs, cuts=range(0, args.depth, args.cuts_skip), ylabel="Finetuned Test Accuracy", save_path=os.path.join(folder, 'test_acc_vs_cut.png'))

        cut_models = experiments[0]
        for i,cut_model in enumerate(cut_models):
            model_folder = os.path.join(folder, 'cut_{}'.format(i*args.cuts_skip))
            os.mkdir(model_folder)
            
            # Save model state dictionary
            torch.save(cut_model['finetuned_model'].state_dict(), os.path.join(model_folder, "fine_tuned_model.pth"))
            torch.save(cut_model['checkpoints'], os.path.join(model_folder, 'fine_tuned_model_checkpoints.pth'))
            
            # Save plots
            plot_variances(cut_model['finetuned_model'], dataset_wrapped.train_loader, os.path.join(model_folder, 'variances_on_finetuning_set.png'))

            # save fine_tuning results
            del cut_model['cut_model']
            del cut_model['finetuned_model']
            del cut_model['checkpoints']
            
            with open(os.path.join(model_folder, "acc.json"), "w+") as f:
                json.dump(cut_model, f, indent=2, ensure_ascii=False)  # encode dict into JSON

    elif len(experiments) > 1:
        box_plot_multiple_exp(experiments, range(0, args.depth, args.cuts_skip), save_path=os.path.join(folder, 'box_plot.png'))
