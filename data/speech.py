from multiprocessing import Pool
import pytorch_lightning as L
from glob import glob
import librosa
import os
import pathlib
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data.utils import pad_or_truncate, roll, gain_adjust, mixup

WORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]

class SpeechCommandsDataset(Dataset):
    def __init__(self, wavs, labels,
                 sample_rate:int=16000, 
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
    y, sr = librosa.load(filename, sr=16000)
    y = pad_or_truncate(y, 16000) # Have to pad here or tensor creation breaks
    return y

class SpeechCommandsDataModule(L.LightningDataModule):
    def __init__(self, 
                 dir: str = '/hpcwork/wq656653/adv-filters/SpeechCommands/speech_commands_v0.02/',
                 batch_size: int = 32, 
                 num_workers: int = 0,
                 test_fold: int = 5,
                 val_fold: int = 4,
                 mixup_kwargs: dict = {},
                 roll_kwargs: dict = {},
                 gain_kwargs: dict = {},
                 debug: bool = False):
        """
        Args:
            dir (str, optional): Directory where the Speech Commands dataset is located.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers for data loading, see Lightning documentation. Defaults to 0.
            test_fold (int, optional): Fold to use as test set. Defaults to 5.
            val_fold (int, optional): Fold to use as validation set. Defaults to 4.
            mixup_kwargs (dict, optional): Arguments for mixup augmentation. See utils.py. Defaults to {}.
            roll_kwargs (dict, optional): Arguments for roll augmentation. See utils.py. Defaults to {}.
            gain_kwargs (dict, optional): Arguments for random gain augmentation. See utils.py. Defaults to {}.
            debug (bool, optional): If True, only use 100 samples for train/val/test. Defaults to False.
        """
        super().__init__()
        self.batch_size = batch_size
        self.dir = pathlib.Path(dir)
        self.num_workers = num_workers
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.mixup_kwargs = mixup_kwargs
        self.roll_kwargs = roll_kwargs
        self.gain_kwargs = gain_kwargs
        self.debug = debug
        self.class_map = {i: word for i, word in enumerate(WORDS)}
            
    def setup(self, stage:str):
        if not os.path.exists(self.dir / "train_list.txt"):
            with open(self.dir / 'validation_list.txt', 'r') as f:
                content = f.read()
                val_files = content.split('\n')
            with open(self.dir / 'testing_list.txt', 'r') as f:
                content = f.read()
                test_files = content.split('\n')
            all_files = glob(self.dir / '*/**/*.wav', recursive=True)
            all_files = [file.replace(self.dir, "") for file in all_files]
            train_files = [file for file in all_files if file not in val_files and 
                           file not in test_files and 
                           "_background_noise_" not in file]
            with open(self.dir / 'train_list.txt', 'w') as f:
                for file in train_files:
                    f.write(str(file) + '\n')
        class_map_inv = {v: k for k, v in self.class_map.items()}   
        if stage == "fit" or stage == "validate": 
            # Train set
            with open(self.dir / 'train_list.txt', 'r') as f:
                content = f.read()
                train_list = content.split('\n')[:-1]
                train_files = [self.dir / file for file in train_list]
            if self.debug:
                train_files = train_files[:100]
            with Pool(max(self.num_workers, 1)) as p:
                train_wavs = p.map(_load_wav, tqdm(train_files, "Loading training files"))
            labels = [file.parent.name for file in train_files]
            labels = [class_map_inv[label] for label in labels]
            train_labels_enc = one_hot(torch.tensor(labels), num_classes=len(self.class_map))
            self.train_dataset = SpeechCommandsDataset(
                torch.tensor(train_wavs, dtype=torch.float32),
                torch.tensor(train_labels_enc, dtype=torch.float32),
                apply_mixup=True, apply_roll=True, apply_random_gain=True,
                mixup_kwargs=self.mixup_kwargs,
                roll_kwargs=self.roll_kwargs,
                gain_kwargs=self.gain_kwargs
            )
            # Validation set
            with open(self.dir / 'validation_list.txt', 'r') as f:
                content = f.read()
                val_list = content.split('\n')[:-1]
                val_files = [self.dir / file for file in val_list]
            if self.debug:
                val_files = val_files[:100]
            with Pool(max(self.num_workers, 1)) as p:
                val_wavs = p.map(_load_wav, tqdm(val_files, "Loading validation files"))
            labels = [file.parent.name for file in val_files]
            labels = [class_map_inv[label] for label in labels]
            val_labels_enc = one_hot(torch.tensor(labels), num_classes=len(self.class_map))
            self.val_dataset = SpeechCommandsDataset(
                torch.tensor(val_wavs, dtype=torch.float32),
                torch.tensor(val_labels_enc, dtype=torch.float32)
            )
        if stage == "test" or stage == "predict":
            with open(self.dir / 'testing_list.txt', 'r') as f:
                content = f.read()
                test_list = content.split('\n')[:-1]
                test_files = [self.dir / file for file in test_list]
            if self.debug:
                test_files = test_files[:100]
            with Pool(max(self.num_workers, 1)) as p:
                test_wavs = p.map(_load_wav, tqdm(test_files, "Loading test files"))
            labels = [file.parent.name for file in test_files]
            labels = [class_map_inv[label] for label in labels]
            test_labels_enc = one_hot(torch.tensor(labels), num_classes=len(self.class_map))
            self.test_dataset = SpeechCommandsDataset(
                torch.tensor(test_wavs, dtype=torch.float32),
                torch.tensor(test_labels_enc, dtype=torch.float32)
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
