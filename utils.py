#ANCHOR Libraries
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy

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

def count_zeros(model):
    total_item_count = 0
    total_zero_count = 0
    layer_zero_count_dict = {}
    layer_item_count_dict = {}
    for num_params,_ in enumerate(model.parameters()):
        layer_zero_count_dict[num_params] = 0
        layer_item_count_dict[num_params] = 0
    
    for n, layer in enumerate(model.parameters()):
        try:
            for channel in layer:
                for kernal_2d in channel:
                    for kernal_1d in kernal_2d:
                        for item in kernal_1d:
                            if item.detach() == 0:
                                layer_zero_count_dict[n] += 1
                                total_zero_count += 1
                            layer_item_count_dict[n] += 1
                            total_item_count += 1
        except TypeError:
            # case where channel is a single valued tensor (bias)
            try:
                if channel.detach() == 0:
                    layer_zero_count_dict[n] += 1
                    total_zero_count += 1
                layer_item_count_dict[n] += 1
                total_item_count += 1
            except RuntimeError:
                # case where channel is a multi-valued tensor (bias)
                for item in channel:
                    if item.detach() == 0:
                        layer_zero_count_dict[n] += 1
                        total_zero_count += 1
                    layer_item_count_dict[n] += 1
                    total_item_count += 1
    return total_zero_count, total_item_count, layer_zero_count_dict, layer_item_count_dict

def percentage_pruned(original_model, pruned_model):
    _, _, original_model_perlayer_zero_count_dict, layer_item_count_dict = count_zeros(original_model)
    total_zero_count, total_item_count, pruned_model_perlayer_zero_count_dict, _ = count_zeros(pruned_model)
    total_percentage_pruned = (total_zero_count*100)/total_item_count
    pruned_perlayer_zero_count_dict = {}
    percentage_pruned_perlayer_dict = {}
    for n, layer in enumerate(original_model_perlayer_zero_count_dict):
        key_name = f"layer {n}"
        pruned_perlayer_zero_count_dict[n] = pruned_model_perlayer_zero_count_dict[n] - original_model_perlayer_zero_count_dict[n] 
        percentage_pruned_perlayer_dict[key_name] = (pruned_perlayer_zero_count_dict[n]/layer_item_count_dict[n])*100
    return percentage_pruned_perlayer_dict, total_percentage_pruned

def get_topk(pred_batch, label_batch, k=1):
    num_correct=0
    batch_size = label_batch.shape[0]
    for datapoint in range(batch_size):
        pred = pred_batch[datapoint]
        _, topk_idx = torch.topk(pred, k)
        label = label_batch[datapoint]
        for idx in topk_idx:
            if int(idx) == int(label):
                num_correct+=1
                break
    return num_correct

#ANCHOR Print table of zeros and non-zeros count
def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round((nonzero/total)*100,1))

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

        


#ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)

#FIXME 
def plot_train_test_stats(stats,
                          epoch_num,
                          key1='train',
                          key2='test',
                          key1_label=None,
                          key2_label=None,
                          xlabel=None,
                          ylabel=None,
                          title=None,
                          yscale=None,
                          ylim_bottom=None,
                          ylim_top=None,
                          savefig=None,
                          sns_style='darkgrid'
                          ):

    assert len(stats[key1]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key1, len(stats[key1]), epoch_num)
    assert len(stats[key2]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key2, len(stats[key2]), epoch_num)

    plt.clf()
    sns.set_style(sns_style)
    x_ticks = np.arange(epoch_num)

    plt.plot(x_ticks, stats[key1], label=key1_label)
    plt.plot(x_ticks, stats[key2], label=key2_label)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    if yscale is not None:
        plt.yscale(yscale)

    if ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        plt.ylim(top=ylim_top)

    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fancybox=True)

    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    else:
        plt.show()
