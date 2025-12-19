from multiprocessing import Pool
import pytorch_lightning as L
from glob import glob
import json
import numpy as np
import librosa
import pandas as pd
import pathlib
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
import torchaudio
from tqdm import tqdm
from data.utils import pad_or_truncate, roll, gain_adjust, mixup

class NSynthDataset(Dataset):
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
        x = torch.Tensor(self.wavs[index]).float()
        y = self.labels[index]
        if self.apply_mixup:
            x, y = mixup(self, x, y, **self.mixup_kwargs)
        if self.apply_random_gain:
            x = gain_adjust(x, **self.gain_kwargs)
        if self.apply_roll:
            x = roll(x, **self.roll_kwargs)
        x = pad_or_truncate(x, self.sample_rate * self.audio_length)
        x = x - x.mean() # Normalisation
        return torch.Tensor(x).float(), y.float()
    
def _load_wav(filename):
    """y, sr = torchaudio.load(filename)
    return y"""
    y, sr = librosa.load(filename, sr=32000)
    return np.float16(y)  # Convert to float16 for memory efficiency

class NSynthDataModule(L.LightningDataModule):
    def __init__(self, 
                 dir:str,
                 batch_size: int = 32, 
                 num_workers: int = 0,
                 mixup_kwargs: dict = {},
                 roll_kwargs: dict = {},
                 gain_kwargs: dict = {}):
        """
        Args:
            dir (str, optional): Directory where the NSynth dataset is located.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers for data loading, see Lightning documentation. Defaults to 0.
            mixup_kwargs (dict, optional): Arguments for mixup augmentation. See utils.py. Defaults to {}.
            roll_kwargs (dict, optional): Arguments for roll augmentation. See utils.py. Defaults to {}.
            gain_kwargs (dict, optional): Arguments for random gain augmentation. See utils.py. Defaults to {}.
        """
        super().__init__()
        self.dir = pathlib.Path(dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup_kwargs = mixup_kwargs
        self.roll_kwargs = roll_kwargs
        self.gain_kwargs = gain_kwargs
        self.class_map = {
            0: "bass",
            1: "keyboard",
            2: "guitar",
            3: "reed",
            4: "flute",
            5: "string",
            6: "vocal",
            7: "brass",
            8: "mallet",
            9: "organ",
            10: "synth_lead"
        }
            
    def setup(self, stage:str):
        class_map_inv = {v: k for k, v in self.class_map.items()}
        if stage == "test" or stage == "predict":
            with open(self.dir / "nsynth-test" / "examples.json", "r") as f:
                test_data = json.load(f)
            label_dict = {name: data["instrument_family_str"] for name, data in test_data.items()}
            filepaths = [self.dir / "nsynth-test" / "audio/" / (file + ".wav") for file in label_dict.keys()]
            with Pool(max(self.num_workers, 1)) as p:
                test_wavs = p.map(_load_wav, tqdm(filepaths, "Loading test folds"))
            test_labels = []
            for _, label_str in label_dict.items():
                label = class_map_inv[label_str]
                test_labels.append(label)
            test_labels = one_hot(torch.tensor(test_labels), num_classes=11).float()
            self.test_dataset = NSynthDataset(
                test_wavs,
                test_labels
            )
        if stage == "fit" or stage == "validate":
            with open(self.dir / "nsynth-train" / "examples.json", "r") as f:
                train_data = json.load(f)
            label_dict = {name: data["instrument_family_str"] for name, data in train_data.items()}
            filepaths = [self.dir / "nsynth-train" / "audio" / (file + ".wav") for file in label_dict.keys()]
            with Pool(max(self.num_workers, 1)) as p:
                train_wavs = list(tqdm(p.imap(_load_wav, filepaths), "Loading train folds", total=len(filepaths)))
            train_labels = []
            for _, label_str in tqdm(label_dict.items(), "Mapping labels"):
                label = class_map_inv[label_str]
                train_labels.append(label)
            train_labels = one_hot(torch.tensor(train_labels), num_classes=11).float()
            print("Creating dataset object")
            self.train_dataset = NSynthDataset(
                train_wavs,
                train_labels,
                apply_mixup=True, apply_roll=True, apply_random_gain=True,
                mixup_kwargs=self.mixup_kwargs,
                roll_kwargs=self.roll_kwargs,
                gain_kwargs=self.gain_kwargs
            )
            print("Done. Creating validation dataset.")
            with open(self.dir / "nsynth-valid" / "examples.json", "r") as f:
                val_data = json.load(f)
            label_dict = {name: data["instrument_family_str"] for name, data in val_data.items()}
            filepaths = [self.dir / "nsynth-valid" / "audio" / (file + ".wav") for file in label_dict.keys()]
            with Pool(max(self.num_workers, 1)) as p:
                val_wavs = p.map(_load_wav, tqdm(filepaths, "Loading validation folds"))
            val_labels = []
            for _, label_str in label_dict.items():
                label = class_map_inv[label_str]
                val_labels.append(label)
            val_labels = one_hot(torch.tensor(val_labels), num_classes=11).float()
            self.val_dataset = NSynthDataset(
                val_wavs,
                val_labels
            )
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
