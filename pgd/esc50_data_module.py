import lightning as L
from glob import glob
import numpy as np
import librosa
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class ESC50Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):       
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
def _load_fold(fold:str):
    features = np.load(fold)
    labels = np.load(fold.replace("features", "labels"))
    return features, labels
    
class ESC50DataModule(L.LightningDataModule):
    def __init__(self, 
                 batch_size: int = 64, 
                 augment: bool = True, 
                 wav_dir: str = "./ESC-50/audio/", 
                 labels_csv: str = "./ESC-50/meta/esc50.csv",
                 num_workers: int = 0,
                 test_fold: int = 5):
        super().__init__()
        self.batch_size = batch_size
        self.augment = augment
        self.wav_dir = wav_dir
        self.labels_csv = labels_csv
        self.num_workers = num_workers
        self.test_fold = test_fold
        
    def setup(self, stage:str):
        # Load wav data
        df = pd.read_csv(self.labels_csv)

        if stage == "test":
            test_wavs, test_labels = [], []
            test_df = df.loc[df.fold == self.test_fold]
            for index, row in test_df.iterrows():
                y, sr = librosa.load(self.wav_dir + row.filename, sr=32000)
                labels = np.zeros(50)
                labels[row.target] = 1
                test_wavs.append(y)
                test_labels.append(labels)
            self.test_dataset = ESC50Dataset(
                torch.tensor(test_wavs, dtype=torch.float32),
                torch.tensor(test_labels, dtype=torch.float32)
            )
        else:
            raise NotImplementedError("Only stage=test supported for now")
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda")
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__":
    module = ESC50DataModule()
    module.setup("fit")