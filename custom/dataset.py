import os

import numpy as np
from torch.utils.data import Dataset
from torchdistill.datasets.registry import register_dataset


@register_dataset
class MuMIMODataset(Dataset):
    @staticmethod
    def _convert(mat_file_paths):
        real_mat_list, imag_mat_list = list(), list()
        org_shape = None
        for mat_file_path in mat_file_paths:
            mat = np.load(mat_file_path)
            if org_shape is None:
                org_shape = mat.shape

            num_samples = len(mat)
            real_mat_list.append(mat.real.reshape(num_samples, -1))
            imag_mat_list.append(mat.imag.reshape(num_samples, -1))
        return np.hstack(real_mat_list) + 1j * np.hstack(imag_mat_list).astype(float), org_shape

    def __init__(self, h_mat_file_paths, v_mat_file_paths, ch_seeds_file_path=None):
        super().__init__()
        assert len(h_mat_file_paths) == len(v_mat_file_paths), \
            f'len(h_mat_file_paths) should be equal to len(v_mat_file_paths)'
        self.h_mat_file_paths = h_mat_file_paths
        self.v_mat_file_paths = v_mat_file_paths
        self.ch_seeds_file_path = os.path.expanduser(ch_seeds_file_path) if ch_seeds_file_path is not None else None
        self.num_users = len(h_mat_file_paths)
        self.samples, _ = self._convert(h_mat_file_paths)
        self.targets, self.org_shape = self._convert(v_mat_file_paths)
        self.ch_seeds = None if self.ch_seeds_file_path is None else np.load(self.ch_seeds_file_path)

    def __getitem__(self, index):
        sample = tuple(np.hstack([sub_sample.real, sub_sample.imag]) for sub_sample in
                       np.hsplit(self.samples[index], self.num_users))
        target = tuple(np.hstack([sub_target.real, sub_target.imag]) for sub_target in
                       np.hsplit(self.targets[index], self.num_users))
        misc_data = sample if self.ch_seeds is None else self.ch_seeds[index]
        return sample, target, misc_data

    def __len__(self):
        return len(self.samples)

    def get_num_users(self):
        return self.num_users

    def get_org_shape(self):
        return self.org_shape

    def reshape_output(self, model_output):
        num_samples = model_output.shape[0]
        model_output = model_output.detach().cpu().numpy()
        real_mat, imag_mat = np.hsplit(model_output, 2)
        real_mat = real_mat.reshape(num_samples, *self.org_shape[1:])
        imag_mat = imag_mat.reshape(num_samples, *self.org_shape[1:])
        return real_mat, imag_mat
