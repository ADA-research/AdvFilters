from training.PaSST.passt import PaSST, PatchEmbed, get_model
from training.PaSST.mel_configurable import AugmentMelSTFT

import numpy as np
import pytorch_lightning as L
from pytorch_lightning.cli import LightningCLI
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Passt(L.LightningModule):
    """This is the base class for PaSST. 
    You will almost always want to use one of the subclasses like PaSSTAdv or PaSSTRandom.
    See the docstrings for those classes for more information."""
    def __init__(self, 
                 pretrained_arch:str = "passt_s_swa_p16_128_ap476",
                 num_classes:int = 527,
                 lr:float = 0.00002,
                 s_patchout_t:int = 40,
                 s_patchout_f:int = 4,
                 n_mels:int = 128,
                 sr:int = 32000,
                 win_length:int = 800,
                 hop_size:int = 320,
                 n_fft:int = 1024,
                 freqm:int = 48,
                 timem:int = 192,
                 htk:bool = True,
                 f_min:float = 0.0,
                 f_max:float = None,
                 window_fn:str = "hann",
                 power:int = 2,
                 normalized:bool = False,
                 center:bool = True,
                 pad_mode:str = "reflect",
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Load Pretrained
        if pretrained_arch is not None:
            self.passt = get_model(arch=pretrained_arch, pretrained=True, n_classes=num_classes)
        else:
            self.passt = PaSST(num_classes=num_classes, 
                            s_patchout_t=s_patchout_t, 
                            s_patchout_f=s_patchout_f)
            
        self.mel = AugmentMelSTFT(n_mels=n_mels, sr=sr, win_length=win_length, hopsize=hop_size, n_fft=n_fft, freqm=freqm,
                                 timem=timem, htk=htk, f_min=f_min, f_max=f_max, window_fn=window_fn, power=power, 
                                 normalized=normalized, center=center, pad_mode=pad_mode) #fmin_aug_range=10,
                                 #fmax_aug_range=2000) #TODO: Extend to allow different window fn
        self.lr = lr
        self.loss = torch.nn.functional.cross_entropy
        self.y_hats = []
        self.y_trues = []
        
        self.save_hyperparameters()
        
    def forward(self, x):
        x = self.mel(x)
        x = self.passt(x.unsqueeze(1))
        return x[0] # Passt returns a tuple (logits, embeddings)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("test/loss", loss)
        self.y_hats.append(torch.softmax(y_hat, 1).cpu())
        self.y_trues.append(y.cpu())
        return {"y_hat": y_hat, "y": y, "loss": loss}

    def on_test_epoch_end(self):
        y_hats = np.vstack(self.y_hats).argmax(axis=1)
        y_trues = np.vstack(self.y_trues).argmax(axis=1)
        self.log(f"test/accuracy", accuracy_score(y_trues, y_hats))
        self.log(f"test/precision", precision_score(y_trues, y_hats, average="macro"))
        self.log(f"test/recall", recall_score(y_trues, y_hats, average="macro"))
        self.log(f"test/f1_score", f1_score(y_trues, y_hats, average="macro"))
        self.y_hats = []
        self.y_trues = []
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val/loss", loss)
        self.y_hats.append(torch.softmax(y_hat, 1).cpu())
        self.y_trues.append(y.cpu())
        return {"y_hat": y_hat, "y": y, "loss": loss}

    def on_validation_epoch_end(self):
        y_hats = np.vstack(self.y_hats).argmax(axis=1)
        y_trues = np.vstack(self.y_trues).argmax(axis=1)
        self.log(f"val/accuracy", accuracy_score(y_trues, y_hats))
        self.log(f"val/precision", precision_score(y_trues, y_hats, average="macro"))
        self.log(f"val/recall", recall_score(y_trues, y_hats, average="macro"))
        self.log(f"val/f1_score", f1_score(y_trues, y_hats, average="macro"))
        self.y_hats = []
        self.y_trues = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
