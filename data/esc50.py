from multiprocessing import Pool
import pytorch_lightning as L
import numpy as np
import librosa
import pandas as pd
import pathlib
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data.utils import pad_or_truncate, roll, gain_adjust, mixup

class ESC50Dataset(Dataset):
    def __init__(self, wavs, labels,
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
        if self.apply_mixup:
            x, y = mixup(self, x, y, **self.mixup_kwargs)
        if self.apply_random_gain:
            x = gain_adjust(x, **self.gain_kwargs)
        if self.apply_roll:
            x = roll(x, **self.roll_kwargs)
        x = pad_or_truncate(x, self.sample_rate * self.audio_length)
        x = x - x.mean() # Normalisation
        return torch.Tensor(x).float(), y
    
def _load_wav(filename):
    y, sr = librosa.load(filename, sr=32000)
    return y

class ESC50DataModule(L.LightningDataModule):
    def __init__(self, 
                 dir: str,
                 batch_size: int = 32, 
                 augment: bool = True, 
                 num_workers: int = 0,
                 test_fold: int = 5,
                 val_fold: int = 4,
                 mixup_kwargs: dict = {},
                 roll_kwargs: dict = {},
                 gain_kwargs: dict = {}):
        """
        DataModule for ESC-50 dataset.
        Args:
            dir (str): Path to the ESC-50 dataset directory. This directory should contain the 'audio' and 'meta' subdirectories.
            batch_size (int, optional): Batch size. Defaults to 32.
            augment (bool, optional): Whether to apply data augmentation. Defaults to True (for training).
            num_workers (int, optional): Number of workers for data loading, see Lightning documentation. Defaults to 0.
            test_fold (int, optional): Fold to use for testing. Defaults to 5.
            val_fold (int, optional): Fold to use for validation. Defaults to 4.
            mixup_kwargs (dict, optional): Arguments for mixup augmentation. See utils.py. Defaults to {}.
            roll_kwargs (dict, optional): Arguments for roll augmentation. See utils.py. Defaults to {}.
            gain_kwargs (dict, optional): Arguments for random gain augmentation. See utils.py. Defaults to {}.
        """
        super().__init__()
        dir = pathlib.Path(dir)
        self.wav_dir = dir / "audio"
        self.labels_csv = dir / "meta" / "esc50.csv"
        self.batch_size = batch_size
        self.augment = augment
        self.num_workers = num_workers
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.mixup_kwargs = mixup_kwargs
        self.roll_kwargs = roll_kwargs
        self.gain_kwargs = gain_kwargs
            
    def setup(self, stage:str):
        # Load label data
        df = pd.read_csv(self.labels_csv)
        if stage == "test" or stage == "predict":
            test_wavs, test_labels = [], []
            test_df = df.loc[df.fold == self.test_fold]
            filepaths = [self.wav_dir / file for file in test_df.filename.to_list()]
            with Pool(max(self.num_workers, 1)) as p:
                test_wavs = p.map(_load_wav, tqdm(filepaths, "Loading test folds"))
            for index, row in test_df.iterrows():
                labels = np.zeros(50)
                labels[row.target] = 1
                test_labels.append(labels)
            self.test_dataset = ESC50Dataset(
                torch.tensor(test_wavs, dtype=torch.float32),
                torch.tensor(test_labels, dtype=torch.float32)
            )
            
        if stage == "fit" or stage == "validate":
            train_wavs, train_labels = [], []
            train_df = df.loc[df.fold != self.test_fold]
            train_df = train_df.loc[train_df.fold != self.val_fold]
            filepaths = [self.wav_dir / file for file in train_df.filename.to_list()]
            with Pool(max(self.num_workers, 1)) as p:
                train_wavs = p.map(_load_wav, tqdm(filepaths, "Loading training folds"))
            for index, row in train_df.iterrows():
                labels = np.zeros(50)
                labels[row.target] = 1
                train_labels.append(labels)
            self.train_dataset = ESC50Dataset(
                torch.tensor(train_wavs, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.float32,),
                apply_mixup=True, apply_roll=True, apply_random_gain=True,
                mixup_kwargs=self.mixup_kwargs,
                roll_kwargs=self.roll_kwargs,
                gain_kwargs=self.gain_kwargs
            )
            val_wavs, val_labels = [], []
            val_df = df.loc[df.fold == self.val_fold]
            filepaths = [self.wav_dir / file for file in val_df.filename.to_list()]
            with Pool(max(self.num_workers, 1)) as p:
                val_wavs = p.map(_load_wav, tqdm(filepaths, "Loading validation folds"))
            for index, row in val_df.iterrows():
                labels = np.zeros(50)
                labels[row.target] = 1
                val_labels.append(labels)
            self.val_dataset = ESC50Dataset(
                torch.tensor(val_wavs, dtype=torch.float32),
                torch.tensor(val_labels, dtype=torch.float32)
            )
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
