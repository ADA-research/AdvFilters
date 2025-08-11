from multiprocessing import Pool
import pytorch_lightning as L
from glob import glob
import numpy as np
import librosa
import pandas as pd
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
    
    
def _load_fold(fold:str):
    features = np.load(fold)
    labels = np.load(fold.replace("features", "labels"))
    return features, labels
    
def _load_wav(filename):
    y, sr = librosa.load(filename, sr=32000)
    return y


class ESC50DataModule(L.LightningDataModule):
    def __init__(self, 
                 batch_size: int = 32, 
                 augment: bool = True, 
                 wav_dir: str = "/hpcwork/wq656653/adv-filters/ESC-50/audio/", 
                 labels_csv: str = "/hpcwork/wq656653/adv-filters/ESC-50/meta/esc50.csv",
                 num_workers: int = 0,
                 test_fold: int = 5,
                 val_fold: int = 4,
                 mixup_kwargs: dict = {},
                 roll_kwargs: dict = {},
                 gain_kwargs: dict = {}):
        super().__init__()
        self.batch_size = batch_size
        self.augment = augment
        self.wav_dir = wav_dir
        self.labels_csv = labels_csv
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
            filepaths = [self.wav_dir + file for file in test_df.filename.to_list()]
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
            filepaths = [self.wav_dir + file for file in train_df.filename.to_list()]
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
            filepaths = [self.wav_dir + file for file in val_df.filename.to_list()]
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

if __name__ == "__main__":
    module = ESC50DataModule(num_workers=20)
    module.setup("fit")
    print(len(module.train_dataloader()))