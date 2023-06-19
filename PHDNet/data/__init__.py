'''dataset loader'''
import torch.utils.data
from data.base_dataset import BaseDataset
from data.cocoart_dataset import COCOARTDataset
from data.phd_dataset import PHDDataset
import numpy as np
import random

'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)
'''

class CustomDataset(object):
    """User-defined dataset
    
    Example usage:
        >>> from data import CustomDataset
        >>> dataset = CustomDataset(opt, is_for_train)
        >>> dataloader = dataset.load_data()
    """
    def __init__(self, opt, is_for_train):
        self.opt = opt
        if opt.dataset_mode.lower() == 'cocoart':
            self.dataset = COCOARTDataset(opt,is_for_train)
            print("dataset [%s] was created" % type(self.dataset).__name__)
        elif opt.dataset_mode.lower() == 'phd':
            self.dataset = PHDDataset(opt,is_for_train)
            print("dataset [%s] was created" % type(self.dataset).__name__)
        else:
            raise ValueError(opt.dataset_mode, "not implmented.")
        
        print(len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=is_for_train,
            num_workers=int(opt.num_threads),
            drop_last=False)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)