#from training.PaSST.passt import PaSST, PatchEmbed, get_model
from training.PANNs.cnn14 import Transfer_Cnn14
from training.PaSST.mel_configurable import AugmentMelSTFT
from attacks.filter_pgd import run_pgd_batched

import numpy as np
import pytorch_lightning as L
from pytorch_lightning.cli import LightningCLI
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import wandb
from lightning.pytorch import loggers as pl_loggers

class CNN14Adv(L.LightningModule):
    def __init__(self, 
                 pretrained_ckpt:str = '/hpcwork/wq656653/adv-filters/nnv_music/training/PANNs/Cnn14_mAP=0.431.pth',
                 num_classes:int = 527,
                 lr:float = 0.00002,
                 n_mels:int = 64,
                 sr:int = 32000,
                 win_length:int = 1024,
                 hop_size:int = 320,
                 n_fft:int = 1024,
                 freqm:int = 48,
                 timem:int = 192,
                 htk:bool = True,
                 f_min:float = 50.0,
                 f_max:float = 14000,
                 window_fn:str = "hann",
                 power:int = 2,
                 normalized:bool = False,
                 center:bool = True,
                 pad_mode:str = "reflect",
                 pgd_alpha = 0, #NOTE: PGD disabled for training in CNN14
                 pgd_eps = 0,
                 pgd_steps = 0,
                 pgd_restarts = 0,
                 pgd_restarts_val = 0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.cnn14 = Transfer_Cnn14(classes_num=num_classes, freeze_base=False)
        if pretrained_ckpt is not None:
            self.cnn14.load_from_pretrain(pretrained_ckpt)
            
        self.mel = AugmentMelSTFT(n_mels=n_mels, sr=sr, win_length=win_length, hopsize=hop_size, n_fft=n_fft, freqm=freqm,
                                 timem=timem, htk=htk, f_min=f_min, f_max=f_max, window_fn=window_fn, power=power, 
                                 normalized=normalized, center=center, pad_mode=pad_mode) #fmin_aug_range=10,
                                 #fmax_aug_range=2000) #TODO: Extend to allow different window fn
        self.lr = lr
        self.loss = torch.nn.functional.cross_entropy
        self.pgd_alpha = pgd_alpha
        self.pgd_eps = pgd_eps
        self.pgd_steps = pgd_steps
        self.pgd_restarts = pgd_restarts
        self.pgd_restarts_val = pgd_restarts_val
        
        self.y_hats = []
        self.y_trues = []
        self.y_hats_perturbed = []
        if self.training:
            self.save_hyperparameters()
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.cnn14(x)
        return x['clipwise_output']

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.mel(x)
        if self.pgd_restarts > 0:
            pgd_res = run_pgd_batched(self,
                                    samples=x,
                                    labels=y,
                                    alpha=self.pgd_alpha,
                                    eps=self.pgd_eps,
                                    max_iters=self.pgd_steps,
                                    restarts=self.pgd_restarts,
                                    verbose=False)
            x_adv = pgd_res["perturbed_inputs"]
        else:
            x_adv = x
        y_hat = self.forward(x_adv)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.mel(x)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("test/clean/loss", loss)
        self.y_hats.append(torch.softmax(y_hat, 1).cpu())
        self.y_trues.append(y.cpu())
        with torch.inference_mode(False):
            pgd_res = run_pgd_batched(self,
                                    samples=x.clone(),
                                    labels=y.clone(),
                                    alpha=self.pgd_alpha,
                                    eps=self.pgd_eps,
                                    max_iters=self.pgd_steps,
                                    restarts=self.pgd_restarts,
                                    verbose=False)
            x_adv = pgd_res["perturbed_inputs"]
            y_hat_adv = self.forward(x_adv)
            loss_adv = self.loss(y_hat_adv, y.clone())
            self.log("test/adv/loss", loss_adv)
            self.y_hats_perturbed.append(y_hat_adv.detach().cpu().numpy())
        return {"y_hat": y_hat, "y": y, "loss": loss}

    def on_test_epoch_end(self):
        y_hats = np.vstack(self.y_hats).argmax(axis=1)
        y_trues = np.vstack(self.y_trues).argmax(axis=1)
        y_hats_perturbed = np.vstack(self.y_hats_perturbed).argmax(axis=1)
        self.log(f"test/clean/accuracy", accuracy_score(y_trues, y_hats), on_epoch=True)
        self.log(f"test/clean/precision", precision_score(y_trues, y_hats, average="micro"), on_epoch=True)
        self.log(f"test/clean/recall", recall_score(y_trues, y_hats, average="micro"), on_epoch=True)
        self.log(f"test/clean/f1_score", f1_score(y_trues, y_hats, average="micro"), on_epoch=True)
        
        # Slightly hacky way to log epsilon as step because 
        # loading and updating wandb tables hurt my feelings
        self.logger.log_metrics({f"test/adv/accuracy": accuracy_score(y_trues, y_hats_perturbed)}, step=self.pgd_eps)
        self.logger.log_metrics({f"test/adv/precision": precision_score(y_trues, y_hats_perturbed, average="micro")}, step=self.pgd_eps)
        self.logger.log_metrics({f"test/adv/recall": recall_score(y_trues, y_hats_perturbed, average="micro")}, step=self.pgd_eps)
        self.logger.log_metrics({f"test/adv/f1_score": f1_score(y_trues, y_hats_perturbed, average="micro")}, step=self.pgd_eps)
        self.y_hats = []
        self.y_trues = []
        self.y_hats_perturbed = []
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.mel(x)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val/clean/loss", loss)
        self.y_hats.append(torch.softmax(y_hat, 1).cpu())
        self.y_trues.append(y.cpu())
        
        with torch.inference_mode(False):
            pgd_res = run_pgd_batched(self,
                                    samples=x,
                                    labels=y,
                                    alpha=self.pgd_alpha,
                                    eps=self.pgd_eps,
                                    max_iters=self.pgd_steps,
                                    restarts=self.pgd_restarts_val,
                                    verbose=False)
            x_adv = pgd_res["perturbed_inputs"]
            y_hat_adv = self.forward(x_adv)
            loss_adv = self.loss(y_hat_adv, y)
            self.log("val/adv/loss", loss_adv)
            self.y_hats_perturbed.append(y_hat_adv.detach().cpu().numpy())
        return {"y_hat": y_hat, "y": y, "loss": loss}

    def on_validation_epoch_end(self):
        y_hats = np.vstack(self.y_hats).argmax(axis=1)
        y_trues = np.vstack(self.y_trues).argmax(axis=1)
        y_hats_perturbed = np.vstack(self.y_hats_perturbed).argmax(axis=1)
        self.log(f"val/clean/accuracy", accuracy_score(y_trues, y_hats), on_epoch=True)
        self.log(f"val/clean/precision", precision_score(y_trues, y_hats, average="micro"), on_epoch=True)
        self.log(f"val/clean/recall", recall_score(y_trues, y_hats, average="micro"), on_epoch=True)
        self.log(f"val/clean/f1_score", f1_score(y_trues, y_hats, average="micro"), on_epoch=True)
        
        self.log(f"val/adv/accuracy", accuracy_score(y_trues, y_hats_perturbed), on_epoch=True)
        self.log(f"val/adv/precision", precision_score(y_trues, y_hats_perturbed, average="micro"), on_epoch=True)
        self.log(f"val/adv/recall", recall_score(y_trues, y_hats_perturbed, average="micro"), on_epoch=True)
        self.log(f"val/adv/f1_score", f1_score(y_trues, y_hats_perturbed, average="micro"), on_epoch=True)
        self.y_hats = []
        self.y_trues = []
        self.y_hats_perturbed = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
