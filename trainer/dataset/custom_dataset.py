import torch.utils.data as torch_data
import sklearn.model_selection as sk_model_selection
import pandas as pd
from torchvision import transforms
from glob import glob
import numpy as np
from .data_utils import mri_png2array, random_stack, sequential_stack


class custom_dataset(torch_data.Dataset):
    def __init__(self,
                 input_paths,
                 targets=None,
                 mode="train",
                 transform=transforms.ToTensor(),
                 conf=None):
        train['last_value_u_in'] = train.groupby('breath_id')['u_in'].transform('last')
        train['u_in_lag1'] = train.groupby('breath_id')['u_in'].shift(1)
        train['u_out_lag1'] = train.groupby('breath_id')['u_out'].shift(1)
        train['u_in_lag_back1'] = train.groupby('breath_id')['u_in'].shift(-1)
        train['u_out_lag_back1'] = train.groupby('breath_id')['u_out'].shift(-1)
        train['u_in_lag2'] = train.groupby('breath_id')['u_in'].shift(2)
        train['u_out_lag2'] = train.groupby('breath_id')['u_out'].shift(2)
        train['u_in_lag_back2'] = train.groupby('breath_id')['u_in'].shift(-2)
        train['u_out_lag_back2'] = train.groupby('breath_id')['u_out'].shift(-2)
        train['u_in_lag3'] = train.groupby('breath_id')['u_in'].shift(3)
        train['u_out_lag3'] = train.groupby('breath_id')['u_out'].shift(3)
        train['u_in_lag_back3'] = train.groupby('breath_id')['u_in'].shift(-3)
        train['u_out_lag_back3'] = train.groupby('breath_id')['u_out'].shift(-3)
        train = train.fillna(0)


        train['R__C'] = train["R"].astype(str) + '__' + train["C"].astype(str)

        # max value of u_in and u_out for each breath
        train['breath_id__u_in__max'] = train.groupby(['breath_id'])['u_in'].transform('max')
        train['breath_id__u_out__max'] = train.groupby(['breath_id'])['u_out'].transform('max')

        # difference between consequitive values
        train['u_in_diff1'] = train['u_in'] - train['u_in_lag1']
        train['u_out_diff1'] = train['u_out'] - train['u_out_lag1']
        train['u_in_diff2'] = train['u_in'] - train['u_in_lag2']
        train['u_out_diff2'] = train['u_out'] - train['u_out_lag2']
        # from here: https://www.kaggle.com/yasufuminakama/ventilator-pressure-lstm-starter
        train.loc[train['time_step'] == 0, 'u_in_diff'] = 0
        train.loc[train['time_step'] == 0, 'u_out_diff'] = 0

        # difference between the current value of u_in and the max value within the breath
        train['breath_id__u_in__diffmax'] = train.groupby(['breath_id'])['u_in'].transform('max') - train['u_in']
        train['breath_id__u_in__diffmean'] = train.groupby(['breath_id'])['u_in'].transform('mean') - train['u_in']

        # OHE
        train = train.merge(pd.get_dummies(train['R'], prefix='R'), left_index=True, right_index=True).drop(['R'], axis=1)
        train = train.merge(pd.get_dummies(train['C'], prefix='C'), left_index=True, right_index=True).drop(['C'], axis=1)
        train = train.merge(pd.get_dummies(train['R__C'], prefix='R__C'), left_index=True, right_index=True).drop(['R__C'], axis=1)

        # https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/273974
        train['u_in_cumsum'] = train.groupby(['breath_id'])['u_in'].cumsum()
        train['time_step_cumsum'] = train.groupby(['breath_id'])['time_step'].cumsum()

    def __len__(self):
        return len(self.input_paths)

    def preload_img_paths(self):
        mri_types = self.conf["mri_types"]
        sampled_image_paths = [{} for _ in range(len(self.input_paths))]

        for i, each_patient_dir in enumerate(self.input_paths):
            for each_mri in mri_types:
                img_paths = sorted(glob(f"{each_patient_dir}/{each_mri}/*"))
                sampled_image_paths[i][each_mri] = self.sampling_scheme(img_paths,
                                                                        N_samples=self.conf["N_samples"])

        return sampled_image_paths

    def load_input(self, index):
        sampled_imgs = self.sampled_image_paths[index]
        inputs = mri_png2array(sampled_imgs,
                               output_type=self.conf["output_type"])

        return inputs

    def load_target(self, index):  # For classification task
        return self.targets[index]

    def __getitem__(self, index):
        inputs = self.load_input(index)
        inputs = self.transform(inputs)
        if self.mode != "test":
            targets = self.load_target(index)
            return inputs, targets
        else:
            return inputs


