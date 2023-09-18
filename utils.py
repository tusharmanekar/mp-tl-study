'''
params = dict(device=device,
                width=width, lr=1e-3, num_train=60,
                sb=0.05, depth=depth, sw=2.0, early_stop_patience=10, activation_function='relu')
if params['activation_function'] == 'relu':
    activation_function = nn.ReLU
elif params['activation_function'] == 'tanh':
    activation_function = nn.Tanh
else:
    activation_function = nn.Tanh
'''

'''        LOGGING IN ALL THE FUNCTIONS:
    logger can be either print function or logger.info function : this way we can write the outputs to a file as well.
    Example usage to write to a file: (given out_path is a path to a folder)
    """
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        file = logging.FileHandler(os.path.join(out_path, "outputs.log"))
        file.setLevel(logging.INFO)
        fileformat = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s",datefmt="%H:%M:%S")
        file.setFormatter(fileformat)
        logger.addHandler(file)
    """
    # then you pass the logger to the function (also setting debug=True): multiple_fine_tuning_experiments(num_experiments, cuts, pre_trained_model, dataset_wrapped, params, debug=True, logger=logger)

    To just use the print function, no need to specify the logger:  multiple_fine_tuning_experiments(num_experiments, cuts, pre_trained_model, dataset_wrapped, params, debug=True)
'''
__version__ = '1.1'

import torch
import numpy as np
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
from data_utils import *

# --------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- FEED-FORWARD MODEL FUNCTIONS -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
# init given linear layer m with given sw and sb
def init_weights(m, sw, sb, default_init=False):
    if type(m) == nn.Linear:
        if default_init:
            m.reset_parameters()
        else:
            nn.init.normal_(m.weight, mean=0.0, std=(np.sqrt(sw / m.out_features)))
            nn.init.normal_(m.bias, mean=0.0, std=np.sqrt(sb))

# init new model
def generate_fc_dnn(input_dim, output_dim, params, activation_function=nn.Tanh, gaussian_init=True):
    depth, width = params['depth'], params['width']
    
    def gen_linear_layer_dim(layer_index):
        return {
            0: (input_dim, width),
            depth - 1: (width, output_dim),
        }.get(layer_index, (width, width))
    
    model = nn.Sequential()
    
    for i in range(depth):
        linear_layer = nn.Linear(*gen_linear_layer_dim(i))
        activation_layer = nn.LogSoftmax(dim=1) if (depth - 1 == i) else activation_function()
        
        # Give descriptive names to layers
        setattr(model, f"linear{i}", linear_layer)
        setattr(model, f"activation{i}", activation_layer)
        
    if gaussian_init:
        model.apply(lambda m: init_weights(m, params['sw'], params['sb']))
    return model

# dataset_loader is fine-tuning dataset
def eval(model, device, dataset_loader, debug=False, logger=print):
    model.eval()
    test_loss, correct = 0., 0.
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = data.reshape([data.shape[0], -1]).to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    num_data = len(dataset_loader.dataset)
    test_loss /= num_data
    acc = correct / num_data
    if debug:
        logger('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss, correct, num_data, 100. * acc))

    return acc

def compute_training_acc(model, dataset, params, debug=False, logger=print):
    device = torch.device(params['device'])
    optimizer = optim.SGD(model.parameters(), lr=params['lr'])
    # if debug: print(model, optimizer)

    # run training for few steps and return the accuracy
    train_acc = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(dataset.train_loader):
        data, target = data.reshape([data.shape[0],-1]).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Evaluate after every 10 steps epoch
        if debug and batch_idx % 10 == 0:
            train_acc = eval(model, device, dataset.train_loader, debug=False)
            logger('Step: {} \tTraining Accuracy: {:.2f}%'.format(batch_idx, train_acc*100))
            # if debug and (epoch+1) % 1 == 0:
            val_acc = eval(model, device, dataset.val_loader, debug=False)
            logger('\t\tValidation Accuracy: {:.2f}%'.format(val_acc*100))

    train_acc = eval(model, device, dataset.train_loader, debug=False)
    test_acc = eval(model, device, dataset.test_loader, debug=False)
    return train_acc, test_acc, model

# like previous function, but run for given number of epochs determined by params['num_train']
def compute_training_acc_epochs(model, dataset, params, debug=False, return_checkpoints=False, save_checkpoints=False, logger=print):
    device = torch.device(params['device'])
    optimizer = optim.SGD(model.parameters(), lr=params['lr'])   
    # if debug: print(model, optimizer)

    if params['early_stop_patience']:
        no_improve_epochs = 0
        max_val_acc = 0.

    train_acc = 0.0
    model.train()
    checkpoints = []

    # Loop over epochs
    for epoch in range(params['num_train']):
        for batch_idx, (data, target) in enumerate(dataset.train_loader):
            data, target = data.reshape([data.shape[0], -1]).to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if debug and batch_idx % 20 == 0:
                #logger('Epoch: {} Train step: {} \tLoss: {:.6f}'.format(epoch, batch_idx, loss.item()))
                pass

        # Evaluate after each epoch
        train_acc = eval(model, device, dataset.train_loader, debug=False)
        val_acc = eval(model, device, dataset.val_loader, debug=False)
        if debug: 
            logger('Epoch: {} \tTraining Accuracy: {:.2f}%'.format(epoch, train_acc*100))
            # if debug and (epoch+1) % 1 == 0:  
            logger('Validation Accuracy: {:.2f}%'.format(val_acc*100))

        if params['early_stop_patience']:
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                logger("val_acc: {}, max_val_acc: {}, no_improve_epochs: {}".format(val_acc, max_val_acc, no_improve_epochs))
                if no_improve_epochs >= params['early_stop_patience']:
                    logger("Early stopping invoked.")
                    break
        
        if return_checkpoints or save_checkpoints:
            # Save checkpoint after each epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc
            }
            checkpoints.append(checkpoint)
        if save_checkpoints:
            torch.save(checkpoint, 'checkpoint_epoch_{}.pth'.format(epoch))

    # Final evaluation after all epochs are completed
    train_acc = eval(model, device, dataset.train_loader, debug=False)
    test_acc = eval(model, device, dataset.test_loader, debug=False)
    if save_checkpoints or return_checkpoints:
        return train_acc, test_acc, model, checkpoints
    else:
        return train_acc, test_acc, model, []

# cut_point: no. of layers to keep in the model
# reinitialize after cutting point using init_weights function
def cut_model(model, sw = 1, sb = 1, gaussian_init=True, cut_point=1, freeze=True, reinitialize=True):
    #deepcopy to avoid changing the original model
    model = copy.deepcopy(model)
    # Convert sequential model to list of layers
    layers = list(model.children())

    # Check if cut_point is out of range
    if cut_point < 0 or cut_point >= len(layers) // 2:
        raise ValueError("cut_point should be in range [0, number of layers - 1]")

    # If freeze is True, set requires_grad to False for layers before cut_point
    if freeze:
        for i in range(cut_point):
            for param in layers[2*i].parameters():
                param.requires_grad = False

    # Cut layers
    new_layers = layers[:2*cut_point]

    # Reinitialize layers after cut point
    for i in range(cut_point, len(layers) // 2):
        linear_layer = layers[2*i]
        activation = layers[2*i + 1]

        if reinitialize:
            # Apply initialization
            init_weights(linear_layer, sw, sb, default_init=not gaussian_init)

        # Append to new layers
        new_layers.extend([linear_layer, activation])

    # Return new model
    return nn.Sequential(*new_layers)

# If num_experiments == 1: after getting the outputs  cut_models = experiments[0] and then this is a simple fine-tuning experiment
# Don't forget to switch the dataset to the fine-tuning:    dataset_wrapped.update_phase('finetune')  before calling this function
def multiple_fine_tuning_experiments(num_experiments, cuts, pre_trained_model, dataset_wrapped, params, freeze=True, reinitialize=False, debug=True, logger=print):
    experiments = []
    for i in range(num_experiments):
        if debug:
            logger('\n\nExperiment number: {}'.format(i))
        cut_models = []
        for cut in cuts:
            temp = {}
            temp['cut_model'] = cut_model(pre_trained_model, cut_point=cut, freeze=freeze, reinitialize=reinitialize)
            logger("\n----> Cut: {}".format(cut))
            finetuned_acc, finetuned_test_acc, finetuned_model, checkpoints_temp = compute_training_acc_epochs(temp['cut_model'], dataset_wrapped, params, debug=debug, save_checkpoints=False, return_checkpoints=True, logger=logger)
            temp['finetuned_acc'] = finetuned_acc
            temp['finetuned_test_acc'] = finetuned_test_acc
            temp['finetuned_model'] = finetuned_model
            temp['checkpoints'] = checkpoints_temp
            cut_models.append(temp)  
        experiments.append(cut_models)
        return experiments
# -----------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- CNNS (will need some updating) --------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

class CustomCNN(nn.Module):
    def __init__(self, input_dim, output_dim, depth, num_channels, act_fn, use_pooling=True):
        super(CustomCNN, self).__init__()
        
        in_channels = 1  # Assuming grayscale input images

        for i in range(depth):
            # Add convolutional layer
            setattr(self, f"conv{i}", nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
            
            # Add activation layer
            setattr(self, f"act{i}", act_fn())

            # Add MaxPool2d layer every 2 convolutional layers if use_pooling is set
            if use_pooling and i % 2 == 1:
                setattr(self, f"pool{i}", nn.MaxPool2d(2))
                input_dim = input_dim // 2

            in_channels = num_channels

        flattened_size = in_channels * input_dim * input_dim
        self.fc = nn.Linear(flattened_size, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        for layer_name in list(self._modules.keys())[:-2]:  # excluding fc and logsoftmax
            layer = getattr(self, layer_name)
            x = layer(x)
        
        #print(x.size())  # Print the tensor size before flattening
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.logsoftmax(x)


def generate_cnn(input_dim, output_dim, depth, num_channels, act_fn=nn.ReLU, use_pooling=True):
    model = CustomCNN(input_dim, output_dim, depth, num_channels, act_fn, use_pooling)
    return model

def compute_training_acc_epochs_cnn(model, dataset, params, debug=False, logger=print):
    device = torch.device(params['device'])
    optimizer = optim.SGD(model.parameters(), lr=params['lr'])
    
    if debug: 
        logger(model, optimizer)

    train_acc = 0.0
    model.train()

    # Loop over epochs
    for epoch in range(params['num_train']):
        for batch_idx, (data, target) in enumerate(dataset.train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if debug and batch_idx % 20 == 0:
                pass

        # Evaluate after each epoch
        if debug:
            train_acc = eval_cnn(model, device, dataset.train_loader, debug=False)
            logger('Epoch: {} \tTraining Accuracy: {:.2f}%'.format(epoch, train_acc*100))
            if debug and (epoch+1) % 1 == 0:
                val_acc = eval_cnn(model, device, dataset.val_loader, debug=False)
                logger('Validation Accuracy: {:.2f}%'.format(val_acc*100))
                
    # Final evaluation after all epochs are completed
    train_acc = eval_cnn(model, device, dataset.train_loader, debug=False)
    test_acc = eval_cnn(model, device, dataset.test_loader, debug=False)
    return train_acc, test_acc, model

def eval_cnn(model, device, dataset_loader, debug=False, logger=print):
    model.eval()
    test_loss, correct = 0., 0.
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    num_data = len(dataset_loader.dataset)
    test_loss /= num_data
    acc = correct / num_data
    if debug:
        logger('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, num_data, 100. * acc))

    return acc

def cut_cnn_model(model, cut_point=1, freeze=True):
    """
    Cut the CNN model at a specific layer and reinitialize the weights for layers after cut_point.

    Parameters:
    - model (nn.Module): Original model.
    - cut_point (int): Layer index at which to cut the model.
    - freeze (bool): If True, layers before cut_point will have their weights frozen.

    Returns:
    - new_model (nn.Sequential): Cut and potentially modified model.
    """
    
    # Deep copy to avoid changing the original model
    model = copy.deepcopy(model)

    # Convert sequential model to list of layers
    layers = list(model.children())

    # Check if cut_point is out of range
    if cut_point < 0 or cut_point >= len(layers):
        raise ValueError("cut_point should be in range [0, number of layers - 1]")

    # If freeze is True, set requires_grad to False for layers before cut_point
    if freeze:
        for i in range(cut_point):
            for param in layers[i].parameters():
                param.requires_grad = False

    # Cut layers
    new_layers = layers[:cut_point]

    # Reinitialize layers after cut point
    for i in range(cut_point, len(layers)):
        layer = layers[i]
        
        # Reinitialize weights if layer has parameters (like Conv2d)
        if hasattr(layer, 'weight'):
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        
        # Append to new layers
        new_layers.append(layer)

    # Return new model
    return nn.Sequential(*new_layers)

def cut_cnn_model_orthogonal(model, cut_point=1, freeze=True):
    """
    Cut the CustomCNN model at a specific layer and reinitialize the weights for layers after cut_point.

    Parameters:
    - model (CustomCNN): Original model.
    - cut_point (int): Layer index at which to cut the model.
    - freeze (bool): If True, layers before cut_point will have their weights frozen.

    Returns:
    - new_model (CustomCNN): Cut and potentially modified model.
    """
    
    # Deep copy to avoid changing the original model
    model = copy.deepcopy(model)

    # Convert modules to a list for ease of access
    layers = list(model.named_children())
    
    current_layer = 0
    for name, layer in layers:
        # Check the type of layer and decide on actions
        if isinstance(layer, nn.Conv2d):
            if current_layer < cut_point:
                if freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                # Reinitialize weights for layers after the cut point
                weights_init(layer)
            current_layer += 1

    return model

# ------------------------------------------------------------- HELPER FUNCTIONS FOR CNN ORTHOGONAL WEIGHT INIT (from the CNN paper) -------------------------
import torch.nn.init as init
def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if m.weight.shape[0] > m.weight.shape[1]:
                _orthogonal_kernel(m.weight.data)
                m.bias.data.zero_()
            else:
                init.orthogonal(m.weight.data)
                m.bias.data.zero_()

        elif isinstance(m, nn.ConvTranspose2d):
            if m.weight.shape[1] > m.weight.shape[0]:
                ConvT_orth_kernel2D(m.weight.data)
               # m.bias.data.zero_()
            else:
                init.orthogonal(m.weight.data)
               # m.bias.data.zero_()

           # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            #m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()
'''
Algorithm requires The number of input channels cannot exceed the number of output channels.
 However, some questions may be in_channels>out_channels. 
 For example, the final dense layer in GAN. If counters this case, Orthogonal_kernel is replaced by the common orthogonal init'''
'''
for example,
net=nn.Conv2d(3,64,3,2,1)
net.apply(Conv2d_weights_orth_init)
'''

######################################Generating 2D orthogonal initialization kernel####################################
#generating uniform orthogonal matrix
def _orthogonal_matrix(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q

#generating orthogonal projection matrix,i.e. the P,Q of Algorithm1 in the original
def _symmetric_projection(n):
    """Compute a n x n symmetric projection matrix.
    Args:
      n: Dimension.
    Returns:
      A n x n orthogonal projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
    """
    q = _orthogonal_matrix(n)
    # randomly zeroing out some columns
    # mask = math.cast(random_ops.random_normal([n], seed=self.seed) > 0,
    # #                      self.dtype)
    mask = torch.randn(n)

    c = torch.mul(mask,q)
    U,_,_= torch.svd(c)
    U1 = U[:,0].view(len(U[:,0]),1)
    P = torch.mm(U1,U1.t())
    P_orth_pro_mat = torch.eye(n)-P
    return P_orth_pro_mat

#generating block matrix the step2 of the Algorithm1 in the original
def _block_orth(p1, p2):
    """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
    Args:
      p1: A symmetric projection matrix (Square).
      p2: A symmetric projection matrix (Square).
    Returns:
      A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                      [(1-p1)p2, (1-p1)(1-p2)]].
    Raises:
      ValueError: If the dimensions of p1 and p2 are different.
    """
    if p1.shape != p2.shape:
        raise ValueError("The dimension of the matrices must be the same.")
    kernel2x2 = {}#Block matrices are contained by a dictionary
    eye = torch.eye(p1.shape[0])
    kernel2x2[0, 0] = torch.mm(p1, p2)
    kernel2x2[0, 1] = torch.mm(p1, (eye - p2))
    kernel2x2[1, 0] = torch.mm((eye - p1), p2)
    kernel2x2[1, 1] = torch.mm((eye - p1), (eye - p2))

    return kernel2x2

#compute convolution operator of equation2.17 in the original
def _matrix_conv(m1, m2):
    """Matrix convolution.
    Args:
      m1: A k x k dictionary, each element is a n x n matrix.
      m2: A l x l dictionary, each element is a n x n matrix.
    Returns:
      (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
    Raises:
      ValueError: if the entries of m1 and m2 are of different dimensions.
    """

    n = m1[0, 0].shape[0]
    if n != m2[0, 0].shape[0]:
        raise ValueError("The entries in matrices m1 and m2 "
                         "must have the same dimensions!")
    k = int(np.sqrt(len(m1)))
    l = int(np.sqrt(len(m2)))
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
        for j in range(size):
            result[i, j] = torch.zeros(n,n)
            for index1 in range(min(k, i + 1)):
                for index2 in range(min(k, j + 1)):
                    if (i - index1) < l and (j - index2) < l:
                        result[i, j] += torch.mm(m1[index1, index2],
                                                        m2[i - index1, j - index2])
    return result

def _dict_to_tensor(x, k1, k2):
    """Convert a dictionary to a tensor.
    Args:
      x: A k1 * k2 dictionary.
      k1: First dimension of x.
      k2: Second dimension of x.
    Returns:
      A k1 * k2 tensor.
    """
    return torch.stack([torch.stack([x[i, j] for j in range(k2)])
                            for i in range(k1)])

#generating a random 2D orthogonal Convolution kernel
def _orthogonal_kernel(tensor):
    """Construct orthogonal kernel for convolution.
    Args:
      ksize: Kernel size.
      cin: Number of input channels.
      cout: Number of output channels.
    Returns:
      An [ksize, ksize, cin, cout] orthogonal kernel.
    Raises:
      ValueError: If cin > cout.
    """
    ksize = tensor.shape[2]
    cin = tensor.shape[1]
    cout = tensor.shape[0]
    if cin > cout:
        raise ValueError("The number of input channels cannot exceed "
                         "the number of output channels.")
    orth = _orthogonal_matrix(cout)[0:cin, :]#这就是算法1中的H
    if ksize == 1:
        return torch.unsqueeze(torch.unsqueeze(orth,0),0)

    p = _block_orth(_symmetric_projection(cout),
                         _symmetric_projection(cout))
    for _ in range(ksize - 2):
        temp = _block_orth(_symmetric_projection(cout),
                                _symmetric_projection(cout))
        p = _matrix_conv(p, temp)
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = torch.mm(orth, p[i, j])
    tensor.copy_(_dict_to_tensor(p, ksize, ksize).permute(3,2,1,0))
    return tensor

#defining 2DConvT orthogonal initialization kernel
def ConvT_orth_kernel2D(tensor):
    ksize = tensor.shape[2]
    cin = tensor.shape[0]
    cout = tensor.shape[1]
    if cin > cout:
        raise ValueError("The number of input channels cannot exceed "
                         "the number of output channels.")
    orth = _orthogonal_matrix(cout)[0:cin, :]  # 这就是算法1中的H
    if ksize == 1:
        return torch.unsqueeze(torch.unsqueeze(orth, 0), 0)

    p = _block_orth(_symmetric_projection(cout),
                    _symmetric_projection(cout))
    for _ in range(ksize - 2):
        temp = _block_orth(_symmetric_projection(cout),
                           _symmetric_projection(cout))
        p = _matrix_conv(p, temp)
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = torch.mm(orth, p[i, j])
    tensor.copy_(_dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0))
    return tensor
#Call method
def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if m.weight.shape[0] > m.weight.shape[1]:
                _orthogonal_kernel(m.weight.data)
                m.bias.data.zero_()
            else:
                init.orthogonal(m.weight.data)
                m.bias.data.zero_()

        elif isinstance(m, nn.ConvTranspose2d):
            if m.weight.shape[1] > m.weight.shape[0]:
                ConvT_orth_kernel2D(m.weight.data)
               # m.bias.data.zero_()
            else:
                init.orthogonal(m.weight.data)
               # m.bias.data.zero_()

           # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            #m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()
'''
Algorithm requires The number of input channels cannot exceed the number of output channels.
 However, some questions may be in_channels>out_channels. 
 For example, the final dense layer in GAN. If counters this case, Orthogonal_kernel is replaced by the common orthogonal init'''
'''
for example,
net=nn.Conv2d(3,64,3,2,1)
net.apply(Conv2d_weights_orth_init)
'''

def makeDeltaOrthogonal(in_channels=3, out_channels=64, kernel_size=3, gain=torch.Tensor([1])):
    weights = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
    out_channels = weights.size(0)
    in_channels = weights.size(1)
    if weights.size(1) > weights.size(0):
        raise ValueError("In_filters cannot be greater than out_filters.")
    q = _orthogonal_matrix(out_channels)
    q = q[:in_channels, :]
    q *= torch.sqrt(gain)
    beta1 = weights.size(2) // 2
    beta2 = weights.size(3) // 2
    weights[:, :, beta1, beta2] = q
    return weights
#Calling method is the same as the above _orthogonal_kernel
######################################################END###############################################################
