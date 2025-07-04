from training.PaSST.passt import PaSST, PatchEmbed, get_model
from training.PaSST.mel_configurable import AugmentMelSTFT
#from audioset_datamodule import AudioSetDataModule

import numpy as np
import pytorch_lightning as L
from pytorch_lightning.cli import LightningCLI
import torch
from sklearn.metrics import average_precision_score

class Passt(L.LightningModule):
    def __init__(self, 
                 pretrained_arch:str = None,
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
        
        #TODO Read HPO config
        
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
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits
        
    def forward(self, x):
        x = self.mel(x)
        x = self.passt(x.unsqueeze(1))
        return x[0] # Passt returns a tuple (logits, embeddings)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return {"y_hat": y_hat, "y": y}

    def on_test_epoch_end(self):
        y_hats = []
        y_trues = []
        for res in outputs:
            y_hats.append(torch.sigmoid(res["y_hat"]).cpu().numpy())
            y_trues.append(res["y"].cpu().numpy())
        y_hats = np.vstack(y_hats)
        y_trues = np.vstack(y_trues)
        mAP = np.array([
            average_precision_score(y_trues[:, i],
                                    y_hats[:, i]) for i in range(y_trues.shape[1])
        ]).mean()
        self.log("mAP", mAP)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        self.on_test_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

   
"""def cli_main():
    cli = LightningCLI(Passt, AudioSetDataModule) 
    #cli = LightningCLI(PasstHPO, OpenMICDataModule) 
    
if __name__ == "__main__":
    cli_main()
    """