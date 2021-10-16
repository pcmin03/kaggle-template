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
        '''
        targets: list of values for classification
        or list of paths to segmentation mask for segmentation task.
        augment: list of keywords for augmentations.
        '''

        self.input_paths = input_paths  # paths to patients
        self.targets = targets
        self.mode = mode
        self.transform = transform
        self.conf = conf

        self.sampling_scheme = {"random": random_stack,
                                "sequential": sequential_stack}[conf["sampling_scheme"]]
        self.sampled_image_paths = self.preload_img_paths()

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


