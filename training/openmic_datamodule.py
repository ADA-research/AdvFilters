from multiprocessing import Pool
import av
import h5py
import io
import pytorch_lightning as L
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from training.utils import decode_mp3, pad_or_truncate, roll
    
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
                 roll_kwargs:dict={}):
        self.wavs = wavs
        self.labels = labels
        self.masks = masks
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.apply_mixup = apply_mixup
        self.mixup_kwargs = mixup_kwargs
        self.apply_roll = apply_roll
        self.roll_kwargs = roll_kwargs

    def __len__(self):       
        return len(self.wavs)
    
    def __getitem__(self, index):
        x = self.wavs[index]
        y = self.labels[index]
        mask = self.masks[index]
        if self.apply_mixup:
            x, y, mask = mixup(self, x, y, mask, **self.mixup_kwargs)
        if self.apply_roll:
            x = roll(x, **self.roll_kwargs)
        x = pad_or_truncate(x, self.sample_rate * self.audio_length)
        x = x - x.mean() # Normalisation
        return x, y, mask
    
class OpenMICDataModule(L.LightningDataModule):
    def __init__(self, 
                 test_hdf: str = "/hpcwork/rwth1754/openmic/openmic_test.hdf5", 
                 train_hdf: str = "/hpcwork/rwth1754/openmic/openmic_train.hdf5",
                 batch_size_train: int = 64, 
                 batch_size_test: int = 64,
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
        if stage == "test" or stage == "predict" or stage == "validate":
            f = h5py.File(self.test_hdf)
        elif stage == "fit":
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
        wavs = [dec[:320000] for dec in decoded_mp3s]
        
        for i in tqdm(range(len(mp3s)), "reading labels and masks"):
            label = targets[i][:20]
            mask = targets[i][20:]
            labels.append(label)
            masks.append(mask)
        
        if stage == "test" or stage == "predict" or stage == "validate":    
            if self.test_dataset is None: # Only load it once
                self.test_dataset = OpenMICDataset(
                    torch.tensor(np.array(wavs)), 
                    torch.tensor(np.array(labels)), 
                    torch.tensor(np.array(masks)))
        elif stage == "fit":
            self.train_dataset = OpenMICDataset(
                torch.tensor(np.array(wavs)), 
                torch.tensor(np.array(labels)), 
                torch.tensor(np.array(masks)),
                apply_mixup=True, apply_roll=True)
            self.setup("validate")
        
        f.close()
        
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_train, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        print("Only train and test set supported for now.")
        if self.test_dataset is None:
            self.setup("validate")
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size_test, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, num_workers=self.num_workers)

if __name__ == "__main__":
    dm = OpenMICDataModule(num_workers=28)
    dm.setup("fit")
    loader = dm.train_dataloader()
    for x, y, mask in loader:
        print(x.shape)
        print(y.shape)
        print(mask.shape)
        print(y)
        print(mask)
        break