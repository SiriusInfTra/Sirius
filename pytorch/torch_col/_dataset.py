from enum import IntEnum
import sys
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Iterator, Optional
import dataclasses

import torch_col
from torch_col.hook import HookABC, HookMode
from torch_col.util import TrainMode, MemoryPool, EventManager
import torch_col.xsched


_BATCH_FINISH_TAG = 'finish'
_BATCH_CANCEL_TAG = 'cancel'


class DatasetType(IntEnum):
    VISION = 0
    TEXT_GEN = 1
    TEXT_CLS = 2
    DYNAMIC_BATCH = 3


@dataclasses.dataclass
class VisionDatasetConfig:
    input_shape: tuple
    num_class: int

@dataclasses.dataclass
class TextDatasetConfig:
    seq_len: int    


class DynamicBatchDataset(IterableDataset):
    def __init__(self, 
                 dataset_type: int, 
                 size: int,
                 global_batch_size: int,
                 vision_dataset_config: Optional[VisionDatasetConfig] = None,
                 text_dataset_config: Optional[TextDatasetConfig] = None,
                 fake_data=False):
        self.dataset_type = dataset_type
        self.size = size
        self.global_batch_size = global_batch_size
        self.vision_dataset_config = vision_dataset_config
        self.text_dataset_config = text_dataset_config

        if not dataset_type == DatasetType.VISION:
            fake_data = True
            print('Warning: fake_data is forced to be True for non-vision dataset', 
                  file=sys.stderr)

        if not fake_data:
            if dataset_type == DatasetType.VISION:
                self.all_inputs = {
                    'images': torch.from_numpy(
                            np.load('workload_data/cifiar10/cifiar10_inputs.npy')
                        ).pin_memory(),
                    'labels': torch.from_numpy(
                            np.load('workload_data/cifiar10/cifiar10_targets.npy')
                        ).pin_memory()
                }
                assert (self.vision_dataset_config.num_class 
                        == torch.max(self.all_inputs['labels']).item() + 1), \
                    f"expect num of class: {torch.max(self.all_inputs['labels']).item() + 1}."
                assert size == len(self.all_inputs['images']), \
                    f"expect size {len(self.all_inputs['image'])}."
                assert (self.vision_dataset_config.input_shape 
                        == self.all_inputs['images'].shape[1:]), \
                    f"expect input shape: {self.all_inputs['images'].shape[1:]}"
            else:
                raise Exception(f"Unsupported dataset type: {dataset_type}")
        else:
            if dataset_type == DatasetType.VISION:
                self.all_inputs = {
                    'images': torch.randn(
                            size, *vision_dataset_config.input_shape
                        ).pin_memory(),
                    'labels': torch.randint(
                            0, vision_dataset_config.num_class, (size,)
                        ).pin_memory()
                }
            elif dataset_type == DatasetType.TEXT_GEN:
                self.all_inputs = {
                    "input_ids": torch.from_numpy(
                            np.random.randint(100, 30000, 
                                              (size, self.text_dataset_config.seq_len))
                        ).pin_memory(),
                }
                self.all_inputs['labels'] = self.all_inputs['input_ids']
                
    def get_batch_size(self, batch):
        if self.dataset_type == DatasetType.VISION:
            return len(batch['images'])
        elif self.dataset_type == DatasetType.TEXT_GEN:
            return len(batch['input_ids'])
        else:
            raise Exception(f"Unsupported dataset type: {self.dataset_type}")

    