   
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class CheckpointManager:
    def __init__(self, save_root, mode: str, top_k: int = 1, save_last: bool = False):
        if save_last:
            raise NotImplementedError

        assert isinstance(top_k, int) and top_k > 0, 'Invalid top-k.'

        if mode == 'max':
            self.prev_best = -float('inf')
            self.max = True
        elif mode == 'min':
            self.prev_best = float('inf')
            self.max = False
        else:
            raise ValueError(f'Invalid mode: {mode}')

        self.save_root = Path(save_root)
        self.save_root.mkdir(exist_ok=True)
        self.top_k = top_k
        self.save_last = save_last
        self.history = dict()
        self.top_ks = list()

    def step(self, metric, save_dict: dict, global_step: int):
        if isinstance(metric, (Tensor, np.ndarray)):
            metric = metric.item()
        elif not isinstance(metric, (int, float)):
            raise ValueError(f'Invalid scalar metric value '
                             f'{metric} of type {type(metric)}.')
        if global_step in self.history:
            raise KeyError(f'Repeated logging at global step {global_step}.')
        self.history[global_step] = metric
        self.top_ks.append(metric)
        self.top_ks.sort(reverse=self.max)
        if len(self.top_ks) <= self.top_k:
            save = True
            remove = False
            rm_val = None
        elif self.top_ks[-1] != metric:  # Current input is not worst.
            save = True
            remove = True
            rm_val = self.top_ks.pop()
        else:
            save = False
            remove = False
            rm_val = None
            self.top_ks.pop()

        if save:
            if remove:
                assert rm_val is not None
                # Imperfect as it implicitly relies on metric values not being the same too often.
                for k, v in sorted(self.history.items()):  # In order of save.
                    if v == rm_val:
                        remove_path = self.save_root / f'{k:07d}.pt'
                        # May hide problems but prevents bugs.
                        remove_path.unlink()
                        break  # Remove earliest if same.
            save_key = 'checkpoint_save_metric'
            save_path = self.save_root / f'{global_step:07d}.pt'
            assert save_key not in save_dict
            save_dict.update({save_key: metric})
            torch.save(save_dict, save_path)