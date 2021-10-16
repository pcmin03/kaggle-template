from datetime import date
from genericpath import exists
import glob
import operator
import logging
import os

import torch
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)

class CheckpointSaver:
    def __init__(
            self,
            conf,
            model,
            optimizer,
            scaler=None):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.conf = conf

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.best_loss = None

        # config
        self.checkpoint_dir = conf['checkpoint_save_path']
        self.top_save_path = conf['top_save_path']
        self.date_path = '/'.join(list(os.getcwd().split('/'))[0:-1])
        self.load_dir = os.path.join(self.date_path, 'load')
        self.save_prefix = conf['checkpoint_save_prefix']
        self.extension = '.pth'
        self.increasing = False if conf['standard'] == 'metric' else True  # a lower metric is better if True
        self.cmp = operator.lt if self.increasing else operator.gt  # True if lhs better than rhs
        self.max_history = conf['top_k']

    def save_checkpoint(self, model, epoch, loss, rank, metric=None):
        print(f"Running DDP checkpoint example on rank {rank}.")
        if rank == 0:
            self.model = model
            assert epoch >= 0
            # path check
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            if not os.path.exists(self.top_save_path):
                os.makedirs(self.top_save_path)
            if not os.path.exists(self.load_dir):
                os.makedirs(self.load_dir)

            # save last checkpoint
            previous_save_path = os.path.join(self.checkpoint_dir, 'last_'+ self.save_prefix + '_epoch_' 
            + str(epoch-1) + self.extension)
            last_save_path = os.path.join(self.checkpoint_dir, 'last_' + self.save_prefix + '_epoch_' 
            + str(epoch) + self.extension)
            if os.path.exists(previous_save_path):
                os.unlink(previous_save_path)
            self._save(last_save_path, epoch, loss, metric)
            
            # save_interval case
            if epoch % self.conf['save_interval'] == 0:
                # path initialize
                last_save_path = os.path.join(self.checkpoint_dir, self.save_prefix + 
                '_epoch_'+ str(epoch) + self.extension)

                # Save checkpoint
                if os.path.exists(last_save_path):
                    os.unlink(last_save_path)
                self._save(last_save_path, epoch, loss, metric)

            # load worst_file info from the checkpoint files
            worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None

            print(len(self.checkpoint_files),metric,worst_file,self.cmp)
            # Top-k
            if (worst_file is None or self.cmp(metric, worst_file[1])):
                # save path check
                if worst_file is not None : 
                    print(self.cmp(metric, worst_file[1]))
                # if epoch % self.conf['save_interval'] != 0:
                last_save_path = os.path.join(self.checkpoint_dir, self.save_prefix + 
                '_epoch_'+ str(epoch) + self.extension)

                # append checkpoint_files to list
                self.checkpoint_files.append((last_save_path, metric))
                self.checkpoint_files = sorted(
                    self.checkpoint_files, key=lambda x: x[1],
                    reverse=not self.increasing)  # sort in descending order if a lower metric is not better

                # delete last checkpoint
                
                if len(self.checkpoint_files) > self.max_history:
                    
                    for file in os.listdir(self.top_save_path):
                        if self.checkpoint_files[self.max_history][0].split('/')[2] in file:
                            os.remove(os.path.join(self.top_save_path, file))
                    self.checkpoint_files.pop()


                # check the existness of the top checkpoint
                for i in range(len(self.checkpoint_files)):
                    # file exist flag
                    exist_flag = False
                    for file in os.listdir(self.top_save_path):
                        if self.checkpoint_files[i][0].split('/')[2] in file:
                            os.rename(os.path.join(self.top_save_path, file), 
                            os.path.join(self.top_save_path, str(i+1).zfill(3)+'st_'+self.checkpoint_files[i][0].split('/')[2])+self.extension)
                            exist_flag=True
                            break

                    # if file doesn't exist in top_folder
                    if not exist_flag:
                        self._save(os.path.join(self.top_save_path, str(i+1).zfill(3)+'st_'+self.checkpoint_files[i][0].split('/')[2])+self.extension,
                        epoch, loss,metric)

                if self.best_metric is None or self.cmp(metric, self.best_metric):
                    self.best_epoch = epoch
                    self.best_metric = metric
                    self.best_loss = loss

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, epoch, loss, metric=None):
        # save_state = {
        #     'epoch': epoch,
        #     'arch': type(self.model).__name__.lower(),
        #     'model': self.model.module.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     'loss': loss
        # }
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': loss
        }
        if self.scaler is not None:
            save_state['scaler'] = self.scaler.state_dict()
        if metric is not None:
            save_state['metric'] = metric
        torch.save(self.model.module.state_dict(), save_path)

    def load_for_training(self, model, optimizer, rank, scaler=None, metric=None):
        # For using DDP
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

        # path check
        if not os.path.exists(self.load_dir):
            os.makedirs(self.load_dir)

        # define the loading path
        if len(os.listdir(self.load_dir)) > 0:
            top_dir = os.path.join(self.load_dir,self.top_save_path[2:])
            top_save_path = os.path.join(top_dir, sorted(os.listdir(top_dir))[0])
        else:
            tmp_dir = os.listdir(self.date_path)
            date_dir=[]
            for dir in tmp_dir:
                if '-' in dir:
                    date_dir.append(dir)
            dir_num = len(date_dir)
            date_offset = 0
            while len(os.listdir(self.date_path))>1:
                # if the last date folder doesn't have checkpoint
                try:
                    last_top_dir = os.path.join(self.date_path,sorted(date_dir)[-2 + date_offset], self.top_save_path[2:])
                    top_save_path = os.path.join(last_top_dir,sorted(os.listdir(last_top_dir))[0])
                    break
                except:
                    if date_offset < -1*(dir_num - 2):
                        raise AttributeError('You dont have pre-trained checkpoint')
                        break
                    else:
                        date_offset -= 1

        # Load state_dict
        checkpoint = torch.load(top_save_path, map_location=map_location)
        model.module.load_state_dict(checkpoint['model'])
        model.train()
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        if checkpoint['scaler'] is not None and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        if checkpoint['metric'] is not None:
            metric = checkpoint['metric']
        return epoch

    def load_for_inference(self, model, rank):
        # For using DDP
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

        # path check
        if not os.path.exists(self.load_dir):
            os.makedirs(self.load_dir)

        # define the loading path
        if len(os.listdir(self.load_dir)) > 0:
            top_dir = os.path.join(self.load_dir,self.top_save_path[2:])
            top_save_path = os.path.join(top_dir, sorted(os.listdir(top_dir))[0])
        else:
            tmp_dir = os.listdir(self.date_path)
            date_dir=[]
            for dir in tmp_dir:
                if '-' in dir:
                    date_dir.append(dir)
            dir_num = len(date_dir)
            date_offset = 0
            while 1:
                # if the last date folder doesn't have checkpoint
                try:
                    last_top_dir = os.path.join(self.date_path,sorted(date_dir)[-2 + date_offset], self.top_save_path[2:])
                    top_save_path = os.path.join(last_top_dir,sorted(os.listdir(last_top_dir))[0])
                    break
                except:
                    if date_offset < -1*(dir_num - 2):
                        raise AttributeError('You dont have pre-trained checkpoint')
                        break
                    else:
                        date_offset -= 1

        # Load state_dict
        checkpoint = torch.load(top_save_path, map_location=map_location)
        print(checkpoint['metric'])
        model.module.load_state_dict(checkpoint['model'])
        model.eval()