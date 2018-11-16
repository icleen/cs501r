import torchvision
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import os
from functools import reduce

class RLDataset(Dataset):
    def __init__(self, experience):
        super(RLDataset, self).__init__()
        self._exp = experience
        self._num_runs = len(experience)
        self._length = reduce(lambda acc, x: acc + len(x), experience, 0)

    def __getitem__(self, index):
        idx = 0
        seen_data = 0
        current_exp = self._exp[0]
        while seen_data + len(current_exp) - 1 < index:
            seen_data += len(current_exp)
            idx += 1
            current_exp = self._exp[idx]
        chosen_exp = current_exp[index - seen_data]
        return chosen_exp

    def __len__(self):
        return self._length


# class PolicyDataset(Dataset):
#     def __init__(self, experience):
#         super(PolicyDataset, self).__init__()
#         self._exp = experience
#         self._num_runs = len(experience)
#         self._length = reduce(lambda acc, x: acc + len(x), experience, 0)
#
#     def __getitem__(self, index):
#         idx = 0
#         seen_data = 0
#         current_exp = self._exp[0]
#         while seen_data + len(current_exp) - 1 < index:
#             seen_data += len(current_exp)
#             idx += 1
#             current_exp = self._exp[idx]
#         chosen_exp = current_exp[index - seen_data]
#         return chosen_exp
#
#     def __len__(self):
#         return self._length
