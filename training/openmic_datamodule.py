from multiprocessing import Pool
import av
import h5py
import io
import pytorch_lightning as L
import logging
import numpy as np
#from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from training.utils import decode_mp3, pad_or_truncate, roll, gain_adjust
    
_logger = logging.getLogger()
torch.set_float32_matmul_precision('medium')
    
def mixup(dataset, x, y, mask, beta=2, rate=0.5):
    """Masked Mixup adapted from Koutini et. al. (PaSST)"""
    if torch.rand(1) < rate:
        idx2 = torch.randint(len(dataset), (1,)).item()
        x2, y2, mask2 = dataset.wavs[idx2], dataset.labels[idx2], dataset.masks[idx2] # Kinda hacky but avoids recursion
        l = np.random.beta(beta, beta)
        l = max(l, 1. - l)
        x1 = x-x.mean()
        x2 = x2-x2.mean()
        x = (x1 * l + x2 * (1. - l))
        x = x - x.mean()
        y = y * l + y2 * (1. - l)
        mask = (mask.bool() | mask2.bool()).float()
    return x, y, mask

class OpenMICDataset(Dataset):
    def __init__(self, wavs, labels, masks,
                 sample_rate:int=32000, 
                 audio_length:float=10, 
                 apply_mixup:bool=False,
                 mixup_kwargs:dict={},
                 apply_roll:bool=False,
                 roll_kwargs:dict={},
                 apply_random_gain:bool=False,
                 gain_kwargs:dict={}):
        self.wavs = wavs
        self.labels = labels
        self.masks = masks
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.apply_mixup = apply_mixup
        self.mixup_kwargs = mixup_kwargs
        self.apply_roll = apply_roll
        self.roll_kwargs = roll_kwargs
        self.apply_random_gain = apply_random_gain
        self.gain_kwargs = gain_kwargs

    def __len__(self):       
        return len(self.wavs)
    
    def __getitem__(self, index):
        x = self.wavs[index]
        y = self.labels[index]
        mask = self.masks[index]
        if self.apply_mixup:
            x, y, mask = mixup(self, x, y, mask, **self.mixup_kwargs)
        if self.apply_random_gain:
            x = gain_adjust(x, **self.gain_kwargs)
        if self.apply_roll:
            x = roll(x, **self.roll_kwargs)
        x = pad_or_truncate(x, self.sample_rate * self.audio_length)
        x = x - x.mean() # Normalisation
        return torch.Tensor(x).float(), y, mask
    
class OpenMICDataModule(L.LightningDataModule):
    def __init__(self, 
                 test_hdf: str = "/hpcwork/rwth1754/openmic/openmic_test.hdf5", 
                 train_hdf: str = "/hpcwork/rwth1754/openmic/openmic_train.hdf5",
                 batch_size_train: int = 6, 
                 batch_size_test: int = 20,
                 num_workers: int = 0,
                 sample_rate: int = 32000,
                 debug: bool = False):
        super().__init__()
        self.test_hdf = test_hdf
        self.train_hdf = train_hdf
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.debug = debug
        self.test_dataset = None
        self.train_dataset = None

        
    def setup(self, stage:str):
        if stage == "test" or stage == "predict":
            if self.test_dataset is not None:
                _logger.info(f"Called datamodule.setup '{stage}' but setup was already done. Skipping.")
                return
            f = h5py.File(self.test_hdf)
        elif stage == "fit" or stage == "validate":
            if self.train_dataset is not None:
                _logger.info(f"Called datamodule.setup '{stage}' but setup was already done. Skipping.")
                return
            f = h5py.File(self.train_hdf)
        else: 
            return
            
        mp3s = f['mp3']
        targets = f['target']
        
        if self.debug:
            mp3s = mp3s[:128]
            targets = targets[:128]
    
        labels = []
        masks = []
        with Pool(max(self.num_workers, 1)) as p:
            decoded_mp3s = p.map(decode_mp3, tqdm(mp3s, "decoding mp3s"))
        wavs = np.array([dec[:320000] for dec in decoded_mp3s])
        
        for i in tqdm(range(len(mp3s)), "reading labels and masks"):
            label = targets[i][:20]
            mask = targets[i][20:]
            labels.append(label)
            masks.append(mask)
        labels = np.array(labels)
        masks = np.array(masks)
        
        if stage == "test" or stage == "predict":    
            if self.test_dataset is None: # Only load it once
                self.test_dataset = OpenMICDataset(
                    torch.tensor(wavs), 
                    torch.tensor(labels), 
                    torch.tensor(masks))
        elif stage == "fit" or stage == "validate":
            """full_dataset = OpenMICDataset(
                torch.tensor(np.array(wavs)), 
                torch.tensor(np.array(labels)), 
                torch.tensor(np.array(masks)),
                apply_mixup=True, apply_roll=True, apply_random_gain=True)
            train_indices, _, val_indices, __ = iterative_train_test_split(
                range(len(full_dataset)),
                full_dataset.labels,
                test_size=0.2) # Always using the same train/val split
            self.train_dataset = Subset(full_dataset, train_indices)
            self.val_dataset = Subset(full_dataset, val_indices)"""
            train_idx, train_y, val_idx, val_y = iterative_train_test_split(
                np.arange(len(wavs)).reshape(len(wavs), 1),
                labels,
                test_size=0.2)
            train_x = wavs[train_idx[:, 0]]
            train_masks = masks[train_idx[:, 0]]
            self.train_dataset = OpenMICDataset(
                torch.tensor(train_x), 
                torch.tensor(train_y), 
                torch.tensor(train_masks),
                apply_mixup=True, apply_roll=True, apply_random_gain=True)
            val_x = wavs[val_idx[:, 0]]
            val_masks = masks[val_idx[:, 0]]
            self.val_dataset = OpenMICDataset(
                torch.tensor(val_x), 
                torch.tensor(val_y), 
                torch.tensor(val_masks),
                apply_mixup=False, apply_roll=False, apply_random_gain=False)
            
            #self.setup("validate")
        
        f.close()
        
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_train, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_test, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size_test, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, num_workers=self.num_workers)

if __name__ == "__main__":
    dm = OpenMICDataModule(num_workers=20, debug=True)
    dm.setup("fit")
    loader = dm.train_dataloader()
    loader_val = dm.val_dataloader()
    _logger.info(len(loader))
    _logger.info(len(loader_val))
    for x, y, mask in loader:
        _logger.info(x.shape)
        _logger.info(y.shape)
        _logger.info(mask.shape)
        _logger.info(y)
        _logger.info(mask)
        break
    
    for x, y, mask in loader_val:
        _logger.info(x.shape)
        _logger.info(y.shape)
        _logger.info(mask.shape)
        _logger.info(y)
        _logger.info(mask)
        break
    