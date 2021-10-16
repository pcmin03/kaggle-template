import numpy as np
import torch
import operator

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path + '/best_checkpoint.pth'
        self.trace_func = trace_func
    def __call__(self, val_loss, model,increase:bool):

        score = val_loss
        compare = operator.lt if increase==True else operator.gt
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif compare(score,self.best_score + self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# class EarlyStopping(object):
#     def __init__(self, mode='min', min_delta=0, patience=10, percentage=False,path):
#         self.mode = mode
#         self.min_delta = min_delta
#         self.patience = patience
#         self.best = None
#         self.num_bad_epochs = 0
#         self.is_better = None
#         self._init_is_better(mode, min_delta, percentage)

#         if patience == 0:
#             self.is_better = lambda a, b: True
#             self.step = lambda a: False

#     def __call__(self, metrics,model):
#         if self.best is None:
#             self.best = metrics
#             return False

#         if torch.isnan(metrics):
#             return True

#         if self.is_better(metrics, self.best):
#             self.num_bad_epochs = 0
#             self.best = metrics
#         else:
#             self.num_bad_epochs += 1

#         if self.num_bad_epochs >= self.patience:
#             return True

#         self.save_checkpoint(self.best, model)

#         return False

#     def _init_is_better(self, mode, min_delta, percentage):
#         if mode not in {'min', 'max'}:
#             raise ValueError('mode ' + mode + ' is unknown!')
#         if not percentage:
#             if mode == 'min':
#                 self.is_better = lambda a, best: a < best - min_delta
#             if mode == 'max':
#                 self.is_better = lambda a, best: a > best + min_delta
#         else:
#             if mode == 'min':
#                 self.is_better = lambda a, best: a < best - (
#                             best * min_delta / 100)
#             if mode == 'max':
#                 self.is_better = lambda a, best: a > best + (
#                             best * min_delta / 100)
        


#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             self.trace_func(f'Validation loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), self.path)
#         self.val_loss_min = val_loss
