# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle
import wandb

# Custom Libraries
import utils
from utils import count_zeros, percentage_pruned, get_topk

# Tensorboard initialization
#writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

# Main
def main(args, ITE=0):
    
    wandb.init(entity="67Samuel", project='Lottery Ticket', name=args.run_name, config={'batch size':args.batch_size, 'lr':args.lr, 'epochs':args.end_iter})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type=="reinit" else False

    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet 

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  
    
    # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    #train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)
    
    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
       model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(device)  
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)   
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)   
    # If you want to add extra model paste here
    else:
        print("\nWrong Model choice\n")
        exit()

    # Weight Initialization
    model.apply(weight_init)

    # Copying and Saving Initial State
    net = copy.deepcopy(model)
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    make_mask(model)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())
        
    # multi GPU
    if args.multi_gpu:
        if torch.cuda.device_count() > 1:
            try:
                ls = []
                for gpu_idx in args.multi_gpu_selection:
                    ls.append(int(gpu_idx))
                gpu_ids = ls
                print("--info--: there are ", torch.cuda.device_count(), "GPUs. Activate GPUs: ", gpu_ids)
                model = nn.DataParallel(model, device_ids=gpu_ids)
                print('data parallel initiated')
            except Exception as e:
                print(e)

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    best_topk_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)
    topk_name = f'top{args.topk} acc (%)'
    
    for _ite in range(args.start_iter, ITERATION):
        if args.early_stopping:
            early_stopper = EarlyStopping(patience=args.esp)
            #late_early_stopping=False
        if not _ite == 0:
            prune_by_percentile(args.prune_percent, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                #if args.arch_type == "fc1":
                #    model = fc1.fc1().to(device)
                #elif args.arch_type == "lenet5":
                #    model = LeNet5.LeNet5().to(device)
                #elif args.arch_type == "alexnet":
                #    model = AlexNet.AlexNet().to(device)
                #elif args.arch_type == "vgg16":
                #    model = vgg.vgg16().to(device)  
                #elif args.arch_type == "resnet18":
                #    model = resnet.resnet18().to(device)   
                #elif args.arch_type == "densenet121":
                #    model = densenet.densenet121().to(device)   
                #else:
                #    print("\nWrong Model choice\n")
                #    exit()
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            if args.schedule_lr:
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                               optimizer, patience=args.lr_patience, factor=0.7)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        if args.tqdm:
            from tqdm import tqdm
            pbar = tqdm(range(args.end_iter))
        else:
            pbar = range(args.end_iter)

        wandb.log({'prune percent':args.prune_percent, 'prune iterations':args.prune_iterations})
        for iter_ in pbar:
            if args.early_stopping:
                if (early_stopper.early_stop == True):
                    break
                #if late_early_stopping:
                #    if (late_early_stopper.early_stop == True):
                #        break

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy, val_loss, topk_accuracy = test(model, test_loader, criterion)
                wandb.log({'top1 acc (%)':accuracy, topk_name:topk_accuracy, 'val loss':val_loss})
                
                # Update lr scheduler
                if args.schedule_lr:
                    lr_scheduler.step(val_loss)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")
                    
                # Record best topk accuracy
                if topk_accuracy > best_topk_accuracy:
                    best_topk_accuracy = topk_accuracy
                    print(f'New best top{args.topk} accuracy: {topk_accuracy}%')
                    
                        #if late_early_stopping:
                        #    late_early_stopper(val_loss=val_loss, model=model)
                        #    if late_early_stopper.early_stop == True:
                        #        break
                    #if (val_loss < args.lesv) and (late_early_stopping==False):
                    #    late_early_stopper = EarlyStopping(patience=args.late_early_stop)
                    #    late_early_stopping = True
                    #    print('late early stopper activated')

            # Training
            loss = train(model, train_loader, optimizer, criterion)
            wandb.log({'loss':loss, 'epochs':iter_})
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy
            
            # Call early stopper
            if best_accuracy > 30:
                if args.early_stopping:
                    early_stopper(val_loss=loss, model=model)
                    if early_stopper.early_stop == True:
                        break
            
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')   
                
        percentage_pruned_dict, total_percentage_pruned = percentage_pruned(net, model)
        print(f'Total percentage pruned: {total_percentage_pruned}%')
        wandb.log(percentage_pruned_dict)
        
        try:
            #writer.add_scalar('Accuracy/test', best_accuracy, comp1)
            bestacc[_ite]=best_accuracy
            wandb.log({'best accuracy':best_accuracy})

            # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
            #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
            #NOTE Normalized the accuracy to [0,100] for ease of plotting.
            plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
            plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
            plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})") 
            plt.xlabel("Iterations") 
            plt.ylabel("Loss and Accuracy") 
            plt.legend() 
            plt.grid(color="gray") 
            utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
            plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200) 
            plt.close()

            # Dump Plot values
            utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
            all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
            all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")

            # Dumping mask
            utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
            with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
                pickle.dump(mask, fp)

            # Making variables into 0
            best_accuracy = 0
            all_loss = np.zeros(args.end_iter,float)
            all_accuracy = np.zeros(args.end_iter,float)
        except Exception as e:
            print(f'post training error: {e}')

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
    comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()    
    
    
# early stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, save_address=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_address = save_address

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.save_address is None:
            torch.save(model.state_dict(), 'checkpoint.pt')
        else:
            torch.save(model.state_dict(), self.save_address)
        self.val_loss_min = val_loss
    
# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    num_correct_k = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            num_correct_k += get_topk(output, target, k=args.topk)
            #test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        topk_acc = 100. * num_correct_k / len(test_loader.dataset)
    return accuracy, test_loss, topk_acc

# Prune by Percentile module
def prune_by_percentile(percent, resample=False, reinit=False,**kwargs):
    global step
    global mask
    global model

    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
                
            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0

# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__=="__main__":
    
    #from gooey import Gooey
    #@Gooey      
    
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=35, type=int, help="Pruning iterations count")
    parser.add_argument("--topk", default=5, type=int, help="Top k accuracy")
    parser.add_argument('--early_stopping', action='store_true',  default=False, help='use early stopping (default: False)') 
    parser.add_argument('--lesv', default=1.0, type=float, help='late early stopping value; the value below which to add the late early stopper (default: 1.0)')
    parser.add_argument('--late_early_stop', default=3, type=int, help='patience of early stopper that activates when loss<args.lesv (default: 3)')
    parser.add_argument('--esp', default=5, type=int, help='patience for early stopping (default: 5)')  
    parser.add_argument('--run_name', default='test', type=str, help='name of the run, recorded in wandb (default: test)')  
    parser.add_argument('--multi_gpu_selection', default='02', type=str, help='indicate which gpus to use. 02 means 0 and 2. (default: 02)')
    parser.add_argument('--multi_gpu', action='store_true', default=False, help='use multiple GPUs to train (default: False)')
    parser.add_argument('--tqdm', action='store_true', default=False, help='use tqdm (default: False)')
    parser.add_argument('--schedule_lr', action='store_true', default=False, help='use lr scheduler (default: False)')
    parser.add_argument('--lr_patience', default=3, type=int, help='how many epochs before decreasing lr (default: 3)')

    
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
